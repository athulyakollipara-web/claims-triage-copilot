import json

import pytest
from fastapi.testclient import TestClient

from claims_copilot.audit import FileAuditLog
from claims_copilot.evidence import check_evidence
from claims_copilot.llm.narrative import check_faithfulness
from claims_copilot.pii import redact_local
from claims_copilot.policy import recommend


def test_pii_redaction():
    text, found = redact_local("Claimant Maya Hart (SSN 912-34-5678, phone 617-555-0101) emailed maya@example.com", ["Maya Hart"])
    assert "912-34-5678" not in text and "Maya Hart" not in text and "maya@example.com" not in text
    assert set(found) == {"SSN", "PHONE", "EMAIL", "NAME"}


def test_evidence_finds_missing_docs_and_flags():
    ev = check_evidence("disability", [{"doc_type": "claimant_statement", "received_date": "2025-01-02"},
                                       {"doc_type": "employer_statement", "received_date": None}],
                        "Payment requested to a third-party account, not the named beneficiary.")
    assert ev.missing_documents == ["attending_physician_statement", "employer_statement"]
    assert ev.note_flags == ["third_party_payment"]


def test_policy_tiers():
    expl = {"risk_drivers": [{"feature": "beneficiary_change_90d"}], "mitigating": []}
    clean = {"missing_documents": [], "note_flags": []}
    assert recommend(0.5, expl, clean, 0.05, 0.2)["action"] == "refer_to_siu"
    assert recommend(0.1, expl, clean, 0.05, 0.2)["action"] == "enhanced_review"
    assert recommend(0.01, expl, {"missing_documents": ["plan_of_care"], "note_flags": []}, 0.05, 0.2)["action"] == "standard_review"
    assert recommend(0.01, expl, clean, 0.05, 0.2)["action"] == "fast_track"


def test_faithfulness_check_rejects_invented_facts():
    facts = {"D1": "beneficiary changed within 90 days before the loss", "S1": "model risk score 0.42 (high risk tier)"}
    assert check_faithfulness("Score is high [S1]. Beneficiary changed recently [D1].", facts)["faithful"]
    bad = check_faithfulness("Score is high [S1]. The claimant has 3 prior fraud convictions [D9].", facts)
    assert not bad["faithful"] and bad["unknown_citations"] == ["D9"] and "3" in bad["new_numbers"]
    assert not check_faithfulness("This looks like fraud.", facts)["faithful"]


def test_unfaithful_llm_output_falls_back_to_template(data_dir):
    from claims_copilot.service import CopilotService

    class Hallucinating:
        def write(self, case, facts):
            return "The claimant was arrested twice in 2019 [D7].", "fake-model"

    svc = CopilotService(narrator=Hallucinating())
    cid = svc.store.queue(limit=1)[0]["claim_id"]
    r = svc.triage(cid, "inv_test")
    assert r["faithfulness"]["faithful"]
    assert "rejected fake-model" in r["narrative_model"]


def test_human_in_the_loop_and_audit_chain(service):
    cid = service.store.queue(limit=3)[2]["claim_id"]
    r = service.triage(cid, "inv_a")
    assert r["status"] == "awaiting_review"
    assert [t["step"] for t in r["trace"]][-1] == "narrate"  # paused before any decision
    d = service.decide(cid, "inv_a", "standard_review", reason="claimant known to adjuster", note="Called claimant 11/14")
    assert d["status"] == "decided" and d["agreed_with_copilot"] is (r["recommendation"]["action"] == "standard_review")
    events = [e["event"] for e in service.audit.for_claim(cid)]
    assert events[-2:] == ["recommendation", "decision"]
    assert service.audit.verify() == (True, None)
    with pytest.raises(LookupError):
        service.decide(cid, "inv_a", "fast_track")  # nothing pending any more


def test_case_memory_is_scoped(service):
    cid = service.store.queue(limit=5)[4]["claim_id"]
    service.triage(cid, "inv_a")
    service.decide(cid, "inv_a", "enhanced_review", note="private hunch", share_note=False)
    service.triage(cid, "inv_b")
    service.decide(cid, "inv_b", "enhanced_review", note="shared update")
    assert [m["note"] for m in service.memory.recall(cid, "inv_b")] == ["shared update"]
    assert {m["note"] for m in service.memory.recall(cid, "inv_a")} == {"private hunch", "shared update"}


def test_tampering_breaks_audit_chain(tmp_path):
    log = FileAuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.append(f"C{i}", "recommendation", "copilot", {"score": i})
    lines = log.path.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["payload"]["score"] = 99
    lines[1] = json.dumps(entry)
    log.path.write_text("\n".join(lines) + "\n")
    assert log.verify() == (False, 2)


def test_api_flow(service):
    import claims_copilot.service as svc_module
    from claims_copilot.api.app import app

    svc_module.default_service.cache_clear()
    client = TestClient(app)
    cid = client.get("/queue", params={"limit": 7}).json()[6]["claim_id"]
    r = client.post(f"/claims/{cid}/triage", json={"investigator": "inv_api"}).json()
    assert r["status"] == "awaiting_review" and "trace" not in r
    assert client.post(f"/claims/{cid}/decision", json={"investigator": "inv_api", "action": "bogus"}).status_code == 422
    assert client.post(f"/claims/{cid}/decision", json={"investigator": "inv_api", "action": "fast_track"}).json()["status"] == "decided"
    assert len(client.get(f"/claims/{cid}/audit").json()) >= 2
    assert client.post("/claims/NOPE/triage", json={"investigator": "inv_api"}).status_code == 404


def test_decision_survives_lost_graph_state(service):
    """Simulates a stateless deployment: a different process records the decision."""
    cid = service.store.queue(limit=9)[8]["claim_id"]
    r = service.triage(cid, "inv_a")
    service.pending.clear()  # in-process graph state is gone
    d = service.decide(cid, "inv_a", r["recommendation"]["action"])
    assert d["agreed_with_copilot"] and d["trace"][0]["resumed_from"] == "audit_log"
    with pytest.raises(LookupError):
        service.decide(cid, "inv_a", "fast_track")
