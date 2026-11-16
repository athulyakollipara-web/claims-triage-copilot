"""Evidence check: what does the claim file say, and what is missing?

Combines the document checklist for the product with red-flag statements in the
(redacted) adjuster notes. The model score says *how risky*; this says *what to
do next*, e.g. "request the attending physician statement".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

REQUIRED_DOCS = {
    "term_life": ["claimant_statement", "death_certificate", "beneficiary_id"],
    "whole_life": ["claimant_statement", "death_certificate", "beneficiary_id"],
    "disability": ["claimant_statement", "attending_physician_statement", "employer_statement"],
    "long_term_care": ["claimant_statement", "plan_of_care", "facility_invoice"],
}
NOTE_FLAGS = {
    "recent_beneficiary_change": r"beneficiary was changed \d+ days before",
    "third_party_payment": r"third-party account",
    "unexplained_late_report": r"could not explain the delay",
    "cause_not_established": r"cause of loss not yet established",
    "document_inconsistency": r"do not match the claim form|signature .* differs",
    "evasive_claimant": r"evasive",
}


@dataclass
class Evidence:
    missing_documents: list[str] = field(default_factory=list)
    received_documents: list[str] = field(default_factory=list)
    note_flags: list[str] = field(default_factory=list)


def check_evidence(claim_type: str, documents: list[dict], notes: str) -> Evidence:
    received = {d["doc_type"] for d in documents if d.get("received_date")}
    required = REQUIRED_DOCS.get(claim_type, [])
    flags = [name for name, pattern in NOTE_FLAGS.items() if re.search(pattern, notes, re.I)]
    return Evidence(
        missing_documents=[d for d in required if d not in received],
        received_documents=sorted(received),
        note_flags=flags,
    )
