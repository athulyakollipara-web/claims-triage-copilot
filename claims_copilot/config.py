"""Runtime settings from environment variables.

* ``CLAIMS_MODE`` - ``local`` (files on disk) or ``aws`` (S3, DynamoDB, Comprehend)
* ``CLAIMS_LLM``  - ``template`` (offline, deterministic narrative) or ``bedrock``
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    mode: str
    llm: str
    data_dir: Path
    artifacts_dir: Path
    aws_region: str
    narrative_model: str
    fallback_model: str
    guardrail_id: str | None
    guardrail_version: str
    audit_table: str
    memory_table: str
    review_capacity: float  # share of claims investigators can review


def get_settings() -> Settings:
    data_dir = Path(os.getenv("CLAIMS_DATA_DIR", ROOT / "data_store"))
    return Settings(
        mode=os.getenv("CLAIMS_MODE", "local"),
        llm=os.getenv("CLAIMS_LLM", "template"),
        data_dir=data_dir,
        artifacts_dir=Path(os.getenv("CLAIMS_ARTIFACTS_DIR", data_dir / "artifacts")),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        narrative_model=os.getenv("BEDROCK_NARRATIVE_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        fallback_model=os.getenv("BEDROCK_FALLBACK_MODEL", "us.amazon.nova-pro-v1:0"),
        guardrail_id=os.getenv("BEDROCK_GUARDRAIL_ID"),
        guardrail_version=os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT"),
        audit_table=os.getenv("AUDIT_TABLE", "claims-copilot-audit"),
        memory_table=os.getenv("MEMORY_TABLE", "claims-copilot-case-memory"),
        review_capacity=float(os.getenv("REVIEW_CAPACITY", "0.10")),
    )
