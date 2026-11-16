"""Governed triage policy: score + evidence -> action. Deterministic and auditable.

Actions:
  refer_to_siu     top ~2% risk: special investigations unit
  enhanced_review  within investigator capacity (top ~10%)
  standard_review  low score but missing documents or red-flag notes
  fast_track       low score, complete file, no flags
"""
from __future__ import annotations

NEXT_STEPS = {
    "beneficiary_change_90d": "confirm the beneficiary change with the policyholder's signed request",
    "bank_change_60d": "verify ownership of the new payout account",
    "payee_third_party": "confirm the third-party payee's relationship and authorization",
    "payee_account_claims": "check other claims paid to the same account",
    "loss_before_issue": "reconcile the date of loss with the policy issue date",
    "policy_age_days": "review the application for material misrepresentation (contestable claim)",
    "contestable": "review the application for material misrepresentation (contestable claim)",
    "provider_flag_rate": "review the provider's recent claim history",
    "late_report": "document the reason for the late report",
    "report_lag_days": "document the reason for the late report",
    "coverage_increase_180d": "review underwriting of the recent coverage increase",
}


def recommend(score: float, explanation: dict, evidence: dict, review_threshold: float, siu_threshold: float) -> dict:
    if score >= siu_threshold:
        tier, action = "high", "refer_to_siu"
    elif score >= review_threshold:
        tier, action = "elevated", "enhanced_review"
    elif evidence["missing_documents"] or evidence["note_flags"]:
        tier, action = "low", "standard_review"
    else:
        tier, action = "low", "fast_track"

    steps = [f"request the {d.replace('_', ' ').replace(' id', ' ID')}" for d in evidence["missing_documents"]]
    if action != "fast_track":
        for d in explanation["risk_drivers"]:
            step = NEXT_STEPS.get(d["feature"])
            if step and step not in steps:
                steps.append(step)
    if not steps:
        steps = ["approve for payment after standard verification"]
    return {"tier": tier, "action": action, "next_steps": steps[:5]}
