"""SHAP explanations turned into reasons an investigator can read.

Each claim gets its top risk drivers (features pushing the score up) and top
mitigating factors (pushing it down), with the actual value and a plain-English
description. These structured drivers, not free text, are what the LLM is
allowed to talk about later, which is how the narrative stays faithful.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

DESCRIPTIONS = {
    "policy_age_days": lambda v: (f"policy was {int(v)} days old at the date of loss" if v >= 0
                                  else f"loss occurred {abs(int(v))} days before the policy was issued"),
    "contestable": lambda v: "loss occurred inside the 2-year contestability period" if v else "policy is past the contestability period",
    "loss_before_issue": lambda v: "date of loss is before the policy issue date" if v else "loss date is after policy issue",
    "insured_age": lambda v: f"insured was {int(v)} at the date of loss",
    "face_amount": lambda v: f"face amount is ${v:,.0f}",
    "claim_amount": lambda v: f"claim amount is ${v:,.0f}",
    "claim_to_face": lambda v: f"claim is {v:.0%} of the face amount",
    "premium_to_face": lambda v: f"annual premium is {v:.2%} of face amount",
    "annual_premium": lambda v: f"annual premium is ${v:,.0f}",
    "report_lag_days": lambda v: f"claim was reported {int(v)} days after the loss",
    "late_report": lambda v: "claim was reported more than 90 days after the loss" if v else "claim was reported promptly",
    "beneficiary_change_90d": lambda v: "beneficiary changed within 90 days before the loss" if v else "no recent beneficiary change",
    "coverage_increase_180d": lambda v: "coverage increased within 180 days before the loss" if v else "no recent coverage increase",
    "address_change_90d": lambda v: "address changed within 90 days before the loss" if v else "no recent address change",
    "bank_change_60d": lambda v: "payout bank account changed within 60 days before the loss" if v else "no recent bank account change",
    "policy_changes_365d": lambda v: f"{int(v)} policy changes in the year before the loss",
    "provider_prior_claims": lambda v: f"provider has {int(v)} prior claims",
    "provider_flag_rate": lambda v: f"{v:.0%} of this provider's prior claims were flagged",
    "payee_third_party": lambda v: "payment requested to a third party" if v else "payment goes to the named beneficiary/insured",
    "payment_by_check": lambda v: "payment requested by check" if v else "payment by direct deposit",
    "payee_account_claims": lambda v: f"payout account has received {int(v)} claims" if v > 1 else "payout account is not shared with other claims",
    "docs_required": lambda v: f"{int(v)} documents required",
    "docs_received_at_report": lambda v: f"only {int(v)} required document{'s' if int(v) != 1 else ''} received within 14 days of reporting",
    "docs_missing_ratio": lambda v: f"{v:.0%} of required documents missing at intake",
    "customer_prior_claims": lambda v: f"customer has {int(v)} earlier claims",
}


def describe(feature: str, value: float) -> str:
    if feature in DESCRIPTIONS:
        return DESCRIPTIONS[feature](value)
    prefix, _, level = feature.rpartition("_")
    for cat in ("claim_type", "cause", "channel", "provider_type"):
        if feature.startswith(cat + "_"):
            level = feature[len(cat) + 1:].replace("_", " ")
            return f"{cat.replace('_', ' ')} is {level}" if value else f"{cat.replace('_', ' ')} is not {level}"
    return f"{feature} = {value}"


@dataclass
class Driver:
    feature: str
    value: float
    shap: float
    description: str


class Explainer:
    def __init__(self, booster: xgb.Booster, features: list[str]):
        self.features = features
        self.explainer = shap.TreeExplainer(booster)

    def explain(self, rows: pd.DataFrame, top: int = 4) -> list[dict]:
        values = self.explainer.shap_values(rows[self.features])
        out = []
        for i in range(len(rows)):
            contrib = values[i]
            order = np.argsort(-contrib)
            row = rows.iloc[i]
            risk = [Driver(self.features[j], float(row[self.features[j]]), round(float(contrib[j]), 4),
                           describe(self.features[j], float(row[self.features[j]]))) for j in order[:top] if contrib[j] > 0.01]
            mitigating = [Driver(self.features[j], float(row[self.features[j]]), round(float(contrib[j]), 4),
                                 describe(self.features[j], float(row[self.features[j]]))) for j in order[::-1][:2] if contrib[j] < -0.01]
            out.append({"risk_drivers": [d.__dict__ for d in risk], "mitigating": [d.__dict__ for d in mitigating]})
        return out
