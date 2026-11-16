"""FastAPI service for the investigator workflow. Docs at http://localhost:8000/docs."""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from claims_copilot.service import default_service

app = FastAPI(
    title="Claims Triage Copilot",
    description="Scores claims for fraud risk, explains why, checks the file for gaps, and records the investigator's decision.",
    version="1.0.0",
)


class TriageRequest(BaseModel):
    investigator: str = Field(..., min_length=2, max_length=64, examples=["inv_jlee"])
    include_trace: bool = False


class DecisionRequest(BaseModel):
    investigator: str = Field(..., min_length=2, max_length=64)
    action: Literal["refer_to_siu", "enhanced_review", "standard_review", "fast_track"]
    reason: str = Field("", max_length=1000)
    note: str = Field("", max_length=2000, description="Saved to case memory for the next person who opens this claim.")
    share_note: bool = True


@app.get("/health")
def health() -> dict:
    s = default_service().settings
    return {"status": "ok", "mode": s.mode, "llm": s.llm}


@app.get("/queue")
def queue(limit: int = 20) -> list[dict]:
    """Highest-risk claims in the current scoring batch."""
    return default_service().store.queue(limit=min(limit, 200))


@app.get("/model")
def model_card() -> dict:
    return default_service().store.metrics


@app.post("/claims/{claim_id}/triage")
def triage(claim_id: str, req: TriageRequest) -> dict:
    try:
        out = default_service().triage(claim_id, req.investigator)
    except KeyError:
        raise HTTPException(404, f"Unknown claim {claim_id}")
    if not req.include_trace:
        out.pop("trace")
    return out


@app.post("/claims/{claim_id}/decision")
def decision(claim_id: str, req: DecisionRequest) -> dict:
    try:
        out = default_service().decide(claim_id, req.investigator, req.action, req.reason, req.note, req.share_note)
    except LookupError as e:
        raise HTTPException(409, str(e))
    out.pop("trace")
    return out


@app.get("/claims/{claim_id}/audit")
def audit(claim_id: str) -> list[dict]:
    return default_service().audit.for_claim(claim_id)
