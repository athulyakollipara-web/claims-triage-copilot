"""Service layer shared by the API and CLI: builds components for the configured mode."""
from __future__ import annotations

import uuid
from functools import lru_cache

from langgraph.types import Command

from claims_copilot.config import Settings, get_settings
from claims_copilot.store import ClaimStore

ACTIONS = {"refer_to_siu", "enhanced_review", "standard_review", "fast_track"}


class CopilotService:
    def __init__(self, settings: Settings | None = None, narrator=None):
        from claims_copilot.agent.graph import build_graph

        s = self.settings = settings or get_settings()
        self.store = ClaimStore(s.data_dir, s.artifacts_dir)
        if s.mode == "aws":
            from claims_copilot.agent.memory import DynamoCaseMemory
            from claims_copilot.audit import DynamoAuditLog
            from claims_copilot.pii import redact_comprehend

            self.audit = DynamoAuditLog(s.audit_table, s.aws_region)
            self.memory = DynamoCaseMemory(s.memory_table, s.aws_region)

            def redactor(text, names):
                return redact_comprehend(text, s.aws_region)
        else:
            from claims_copilot.agent.memory import FileCaseMemory
            from claims_copilot.audit import FileAuditLog
            from claims_copilot.pii import redact_local as redactor

            self.audit = FileAuditLog(s.data_dir / "audit" / "audit.jsonl")
            self.memory = FileCaseMemory(s.data_dir / "audit" / "case_memory.json")
        if narrator is None:
            if s.llm == "bedrock":
                from claims_copilot.llm.narrative import BedrockNarrator

                narrator = BedrockNarrator([s.narrative_model, s.fallback_model], s.aws_region)
            else:
                from claims_copilot.llm.narrative import TemplateNarrator

                narrator = TemplateNarrator()
        self.graph = build_graph(self.store, redactor, narrator, self.audit, self.memory)
        self.pending: dict[str, str] = {}  # claim_id -> thread id awaiting a decision

    def triage(self, claim_id: str, investigator: str) -> dict:
        if not self.store.exists(claim_id):
            raise KeyError(claim_id)
        thread = f"{claim_id}:{uuid.uuid4().hex[:8]}"
        state = self.graph.invoke({"claim_id": claim_id, "investigator": investigator}, {"configurable": {"thread_id": thread}})
        self.pending[claim_id] = thread
        return {
            "claim_id": claim_id, "status": "awaiting_review", "score": state["score"],
            "recommendation": state["recommendation"], "narrative": state["narrative"],
            "narrative_model": state["narrative_model"], "faithfulness": state["faithfulness"],
            "risk_drivers": state["explanation"]["risk_drivers"], "mitigating": state["explanation"]["mitigating"],
            "evidence": state["evidence"], "redacted_notes": state["redacted_notes"], "pii_found": state["pii_found"],
            "case_memory": state["memory"], "trace": state["trace"],
        }

    def decide(self, claim_id: str, investigator: str, action: str, reason: str = "", note: str = "", share_note: bool = True) -> dict:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {sorted(ACTIONS)}")
        thread = self.pending.pop(claim_id, None)
        if thread is None:
            # Stateless deployments (e.g. Lambda) lose in-process graph state between requests.
            # The recommendation is already in the audit log, so the decision can be recorded from it.
            return self._decide_from_audit(claim_id, investigator, action, reason, note, share_note)
        state = self.graph.invoke(
            Command(resume={"investigator": investigator, "action": action, "reason": reason, "note": note, "share_note": share_note}),
            {"configurable": {"thread_id": thread}},
        )
        return {"claim_id": claim_id, "status": "decided", "action": action,
                "agreed_with_copilot": action == state["recommendation"]["action"], "trace": state["trace"]}


    def _decide_from_audit(self, claim_id, investigator, action, reason, note, share_note) -> dict:
        history = self.audit.for_claim(claim_id)
        if not history or history[-1]["event"] != "recommendation":
            raise LookupError(f"No triage awaiting review for {claim_id}")
        recommended = history[-1]["payload"]["recommendation"]["action"]
        agreed = action == recommended
        self.audit.append(claim_id, "decision", investigator, {"action": action, "agreed_with_copilot": agreed, "reason": reason})
        if note:
            self.memory.remember(claim_id, investigator, note, shared=share_note)
        return {"claim_id": claim_id, "status": "decided", "action": action, "agreed_with_copilot": agreed,
                "trace": [{"step": "record", "resumed_from": "audit_log"}]}


@lru_cache(maxsize=1)
def default_service() -> CopilotService:
    return CopilotService()
