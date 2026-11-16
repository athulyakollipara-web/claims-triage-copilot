"""Claims triage workflow as a LangGraph state machine with a human in the loop.

  load -> redact -> score -> explain -> evidence -> recall -> recommend -> narrate -> human_review -> record
                                                                                        (interrupt)

The graph pauses at ``human_review`` and waits for an investigator's decision.
Nothing is paid, denied or referred automatically: the copilot prepares the
case, the person decides, and both are written to the audit log.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from claims_copilot.evidence import check_evidence
from claims_copilot.llm.narrative import check_faithfulness, facts_from, template_narrative
from claims_copilot.model.explain import Explainer
from claims_copilot.policy import recommend


class CaseState(TypedDict, total=False):
    claim_id: str
    investigator: str
    claim: dict
    redacted_notes: str
    pii_found: list[str]
    score: float
    explanation: dict
    evidence: dict
    memory: list[dict]
    recommendation: dict
    narrative: str
    narrative_model: str
    faithfulness: dict
    decision: dict
    trace: list[dict]


def _t(state: CaseState, step: str, started: float, **info: Any) -> list[dict]:
    return state.get("trace", []) + [{"step": step, "ms": round((time.perf_counter() - started) * 1000, 1), **info}]


def build_graph(store, redactor, narrator, audit, memory, checkpointer=None):
    explainer = Explainer(store.model, list(store.feature_row(store.features.index[0]).columns))
    review_t, siu_t = store.metrics["threshold"], store.metrics["siu_threshold"]

    def load(state: CaseState) -> CaseState:
        t = time.perf_counter()
        return {"claim": store.claim(state["claim_id"]), "trace": _t(state, "load", t)}

    def redact(state: CaseState) -> CaseState:
        t = time.perf_counter()
        text, found = redactor(state["claim"]["notes"], [state["claim"]["claimant_name"]])
        return {"redacted_notes": text, "pii_found": found, "trace": _t(state, "redact", t, pii=found)}

    def score(state: CaseState) -> CaseState:
        t = time.perf_counter()
        s = store.score(store.feature_row(state["claim_id"]))[0]
        return {"score": round(s, 4), "trace": _t(state, "score", t, score=round(s, 4))}

    def explain(state: CaseState) -> CaseState:
        t = time.perf_counter()
        e = explainer.explain(store.feature_row(state["claim_id"]))[0]
        return {"explanation": e, "trace": _t(state, "explain", t, drivers=[d["feature"] for d in e["risk_drivers"]])}

    def evidence(state: CaseState) -> CaseState:
        t = time.perf_counter()
        ev = check_evidence(state["claim"]["claim_type"], state["claim"]["documents"], state["redacted_notes"])
        return {"evidence": ev.__dict__, "trace": _t(state, "evidence", t, missing=ev.missing_documents, flags=ev.note_flags)}

    def recall(state: CaseState) -> CaseState:
        t = time.perf_counter()
        mem = memory.recall(state["claim_id"], state.get("investigator", "unknown"))
        return {"memory": mem, "trace": _t(state, "recall", t, items=len(mem))}

    def recommend_node(state: CaseState) -> CaseState:
        t = time.perf_counter()
        rec = recommend(state["score"], state["explanation"], state["evidence"], review_t, siu_t)
        audit.append(state["claim_id"], "recommendation", "copilot", {
            "score": state["score"], "recommendation": rec,
            "drivers": [d["feature"] for d in state["explanation"]["risk_drivers"]],
        })
        return {"recommendation": rec, "trace": _t(state, "recommend", t, **rec)}

    def narrate(state: CaseState) -> CaseState:
        t = time.perf_counter()
        case = {"score": state["score"], "explanation": state["explanation"], "evidence": state["evidence"],
                "recommendation": state["recommendation"]}
        facts = facts_from(case)
        try:
            text, model = narrator.write(case, facts)
        except Exception as e:  # model outage must not block the queue
            text, model = template_narrative(case, facts), f"template (fallback: {type(e).__name__})"
        check = check_faithfulness(text, facts)
        if not check["faithful"]:
            text, model = template_narrative(case, facts), f"template (rejected {model}: unfaithful)"
            check = check_faithfulness(text, facts)
        return {"narrative": text, "narrative_model": model, "faithfulness": check,
                "trace": _t(state, "narrate", t, model=model, faithful=check["faithful"])}

    def human_review(state: CaseState) -> CaseState:
        decision = interrupt({
            "claim_id": state["claim_id"], "score": state["score"], "recommendation": state["recommendation"],
            "narrative": state["narrative"], "prompt": "Approve the recommended action or override it.",
        })
        return {"decision": decision}

    def record(state: CaseState) -> CaseState:
        t = time.perf_counter()
        d = state["decision"]
        agreed = d["action"] == state["recommendation"]["action"]
        audit.append(state["claim_id"], "decision", d["investigator"], {
            "action": d["action"], "agreed_with_copilot": agreed, "reason": d.get("reason", ""),
        })
        if d.get("note"):
            memory.remember(state["claim_id"], d["investigator"], d["note"], shared=d.get("share_note", True))
        return {"trace": _t(state, "record", t, agreed=agreed)}

    g = StateGraph(CaseState)
    for name, fn in [("load", load), ("redact", redact), ("score", score), ("explain", explain), ("evidence", evidence),
                     ("recall", recall), ("recommend", recommend_node), ("narrate", narrate),
                     ("human_review", human_review), ("record", record)]:
        g.add_node(name, fn)
    g.set_entry_point("load")
    for a, b in [("load", "redact"), ("redact", "score"), ("score", "explain"), ("explain", "evidence"),
                 ("evidence", "recall"), ("recall", "recommend"), ("recommend", "narrate"),
                 ("narrate", "human_review"), ("human_review", "record")]:
        g.add_edge(a, b)
    g.add_edge("record", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
