"""PII redaction for adjuster notes before any text reaches a model or a log.

AWS mode uses Amazon Comprehend ``DetectPiiEntities``. Local mode uses patterns
for the identifiers that appear in claim notes, plus the claimant's known name.
"""
from __future__ import annotations

import re

PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PHONE": r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
    "EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "BANK_ACCOUNT": r"\bACCT\d{6,}\b",
}


def redact_local(text: str, known_names: list[str] | None = None) -> tuple[str, list[str]]:
    found = []
    for label, pattern in PATTERNS.items():
        if re.search(pattern, text):
            found.append(label)
            text = re.sub(pattern, f"[{label}]", text)
    for name in known_names or []:
        if name and name in text:
            found.append("NAME")
            text = text.replace(name, "[NAME]")
    return text, sorted(set(found))


def redact_comprehend(text: str, region: str, min_score: float = 0.8) -> tuple[str, list[str]]:
    import boto3

    resp = boto3.client("comprehend", region_name=region).detect_pii_entities(Text=text, LanguageCode="en")
    entities = sorted((e for e in resp["Entities"] if e["Score"] >= min_score), key=lambda e: e["BeginOffset"], reverse=True)
    for e in entities:  # replace from the end so offsets stay valid
        text = text[: e["BeginOffset"]] + f"[{e['Type']}]" + text[e["EndOffset"]:]
    return text, sorted({e["Type"] for e in entities})
