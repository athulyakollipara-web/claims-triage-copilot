"""Synthetic claims data from six upstream systems.

There is no public dataset of life/disability claims with fraud labels and the
documents around them, so this generator builds one. It is seeded and
reproducible. Everything is invented: names, SSN-like numbers (900-series,
which are never issued) and amounts. Fraud labels come from a hidden logistic
model over known risk signals plus noise, so a model has to learn them; nothing
in the features is a copy of the label.

Systems (one folder each under ``raw/``):
  policy_admin/      policies.csv, customers.csv
  claims/            claims.csv
  servicing/         policy_changes.csv
  provider_registry/ providers.csv
  payments/          payments.csv
  documents/         documents.csv, adjuster_notes.csv

``generate(..., drifted=True)`` produces a later scoring batch whose
distribution has shifted (bigger claims, more online filing) for drift monitoring.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCTS = ["term_life", "whole_life", "disability", "long_term_care"]
REQUIRED_DOCS = {
    "term_life": ["claimant_statement", "death_certificate", "beneficiary_id"],
    "whole_life": ["claimant_statement", "death_certificate", "beneficiary_id"],
    "disability": ["claimant_statement", "attending_physician_statement", "employer_statement"],
    "long_term_care": ["claimant_statement", "plan_of_care", "facility_invoice"],
}
FIRST = ["Maya", "Arjun", "Elena", "Marcus", "Priya", "Daniel", "Sofia", "Kenji", "Grace", "Omar", "Lena", "Victor"]
LAST = ["Hart", "Iyer", "Novak", "Reyes", "Okafor", "Lindqvist", "Brennan", "Sato", "Mendez", "Kowalski", "Adler", "Quinn"]
CHANNEL_TEXT = {"agent": "through their agent", "online": "online", "phone": "by phone"}
STATES = ["MA", "NY", "CA", "TX", "FL", "IL", "PA", "OH", "GA", "NC"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(out_dir: Path, n_claims: int = 20000, seed: int = 11, drifted: bool = False,
             start: str = "2023-01-01", end: str = "2025-06-30") -> dict[str, int]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "C" if not drifted else "D"

    # ---------------- providers ----------------
    n_prov = 400
    prov_type = rng.choice(["physician", "funeral_home", "care_facility"], n_prov, p=[0.5, 0.3, 0.2])
    prior_claims = rng.integers(5, 400, n_prov)
    shady = rng.random(n_prov) < 0.06  # a few providers are involved in many bad claims
    flag_rate = np.where(shady, rng.uniform(0.15, 0.4, n_prov), rng.uniform(0.0, 0.05, n_prov))
    providers = pd.DataFrame({
        "provider_id": [f"P{i:04d}" for i in range(n_prov)], "provider_type": prov_type,
        "state": rng.choice(STATES, n_prov), "prior_claims": prior_claims,
        "prior_flagged": np.round(prior_claims * flag_rate).astype(int),
    })

    # ---------------- customers ----------------
    n_cust = int(n_claims * 1.1)
    fn, ln = rng.choice(FIRST, n_cust), rng.choice(LAST, n_cust)
    customers = pd.DataFrame({
        "customer_id": [f"{prefix}U{i:06d}" for i in range(n_cust)], "first_name": fn, "last_name": ln,
        "dob": (pd.Timestamp("1940-01-01") + pd.to_timedelta(rng.integers(0, 45 * 365, n_cust), unit="D")).date,
        "ssn": [f"9{rng.integers(10, 99)}-{rng.integers(10, 99)}-{rng.integers(1000, 9999)}" for _ in range(n_cust)],
        "phone": [f"617-555-{rng.integers(1000, 9999)}" for _ in range(n_cust)],
        "email": [f"{a.lower()}.{b.lower()}{i}@example.com" for i, (a, b) in enumerate(zip(fn, ln))],
    })

    # ---------------- claims (dates first) + one policy per claim ----------------
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    loss = pd.Series(t0 + pd.to_timedelta(rng.integers(0, (t1 - t0).days, n_claims), unit="D"))
    early = rng.random(n_claims) < 0.12                 # loss inside the 2-year contestability window
    loss_before_issue = rng.random(n_claims) < 0.004    # loss date earlier than the policy issue date
    age_days = np.where(early, rng.integers(30, 730, n_claims), rng.integers(730, 6000, n_claims))
    age_days = np.where(loss_before_issue, -rng.integers(5, 200, n_claims), age_days)
    claim_issue = loss - pd.to_timedelta(age_days, unit="D")

    n_extra = int(n_claims * 0.3)                       # policies with no claim
    n_pol = n_claims + n_extra
    product = rng.choice(PRODUCTS, n_pol, p=[0.4, 0.2, 0.25, 0.15])
    issue = pd.concat([claim_issue, pd.Series(pd.Timestamp("2010-01-01") + pd.to_timedelta(rng.integers(0, 12 * 365, n_extra), unit="D"))], ignore_index=True)
    face = np.round(np.exp(rng.normal(12.2, 0.8, n_pol)), -3).clip(25_000, 3_000_000)
    face = np.where(np.isin(product, ["disability", "long_term_care"]), np.round(face / 10, -2), face)
    cust_of_policy = rng.integers(0, n_cust, n_pol)     # some customers hold several policies
    policies = pd.DataFrame({
        "policy_id": [f"{prefix}L{i:06d}" for i in range(n_pol)], "customer_id": customers.customer_id.to_numpy()[cust_of_policy],
        "product": product, "issue_date": issue.dt.date, "face_amount": face,
        "annual_premium": np.round(face * rng.uniform(0.004, 0.02, n_pol), 2),
        "state": rng.choice(STATES, n_pol), "payment_mode": rng.choice(["monthly", "annual", "quarterly"], n_pol),
    })
    pol = policies.iloc[:n_claims].reset_index(drop=True)
    issue_ts = pd.to_datetime(pol["issue_date"])
    lag = rng.gamma(2.0, 12 if not drifted else 15, n_claims).astype(int)
    late = rng.random(n_claims) < 0.05
    lag = np.where(late, lag + rng.integers(90, 400, n_claims), lag)
    report = loss + pd.to_timedelta(lag, unit="D")

    is_life = pol["product"].isin(["term_life", "whole_life"]).to_numpy()
    cause = np.where(is_life, rng.choice(["natural", "accident", "illness", "unknown"], n_claims, p=[0.55, 0.15, 0.25, 0.05]),
                     rng.choice(["illness", "accident", "natural", "unknown"], n_claims, p=[0.6, 0.25, 0.1, 0.05]))
    amount_share = np.where(is_life, 1.0, rng.uniform(0.2, 1.1, n_claims))
    amount = np.round(pol["face_amount"].to_numpy() * amount_share * (1.15 if drifted else 1.0), 2)
    channel = rng.choice(["agent", "online", "phone"], n_claims, p=[0.35, 0.45 if drifted else 0.3, 0.2 if drifted else 0.35])
    prov_for_type = {
        "life": providers.index[providers.provider_type.isin(["physician", "funeral_home"])].to_numpy(),
        "care": providers.index[providers.provider_type.isin(["physician", "care_facility"])].to_numpy(),
    }
    prov_idx = np.where(is_life, rng.choice(prov_for_type["life"], n_claims), rng.choice(prov_for_type["care"], n_claims))
    claim_ids = [f"{prefix}{i:06d}" for i in range(n_claims)]

    # ---------------- servicing: policy changes before the loss ----------------
    def recent_change(p: float, max_days: int) -> np.ndarray:
        return np.where(rng.random(n_claims) < p, rng.integers(1, max_days, n_claims), -1)

    benef_days = recent_change(0.05, 120)
    cover_days = recent_change(0.04, 240)
    addr_days = recent_change(0.08, 120)
    bank_days = recent_change(0.04, 90)
    changes = []
    for kind, days in (("beneficiary", benef_days), ("coverage_increase", cover_days), ("address", addr_days), ("bank_account", bank_days)):
        mask = days > 0
        changes.append(pd.DataFrame({
            "policy_id": pol.loc[mask, "policy_id"].to_numpy(), "change_type": kind,
            "change_date": (loss[mask] - pd.to_timedelta(days[mask], unit="D")).dt.date.to_numpy(),
        }))
    background = rng.choice(n_pol, n_pol // 3)
    changes.append(pd.DataFrame({
        "policy_id": policies.policy_id.iloc[background].to_numpy(),
        "change_type": rng.choice(["address", "beneficiary", "bank_account"], len(background)),
        "change_date": (pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 2900, len(background)), unit="D")).date,
    }))
    policy_changes = pd.concat(changes, ignore_index=True)

    # ---------------- payments ----------------
    third_party = rng.random(n_claims) < 0.07
    accounts = np.array([f"ACCT{rng.integers(10**7, 10**8)}" for _ in range(n_claims)])
    ring = rng.random(n_claims) < 0.015  # a few accounts receive several claims
    ring_accounts = np.array([f"ACCT{rng.integers(10**7, 10**8)}" for _ in range(25)])
    accounts = np.where(ring, rng.choice(ring_accounts, n_claims), accounts)
    payments = pd.DataFrame({
        "claim_id": claim_ids, "payee_account_id": accounts,
        "payment_method": rng.choice(["ach", "check"], n_claims, p=[0.8, 0.2]),
        "payee_relationship": np.where(third_party, "third_party", np.where(is_life, "beneficiary", "insured")),
    })

    # ---------------- hidden fraud model ----------------
    pol_age_days = (loss - issue_ts).dt.days.to_numpy()
    prov_rate = (providers.prior_flagged / providers.prior_claims.clip(lower=1)).to_numpy()[prov_idx]
    contestable = pol_age_days < 730
    benef_recent = (benef_days > 0) & (benef_days < 90)
    z = (
        -4.9
        + 0.9 * contestable + 1.8 * loss_before_issue
        + 0.8 * benef_recent + 1.2 * ((cover_days > 0) & (cover_days < 180))
        + 1.4 * ((bank_days > 0) & (bank_days < 60)) + 1.2 * third_party + 2.0 * ring
        + 1.8 * (prov_rate > 0.12) + 0.8 * late + 0.6 * (cause == "unknown") + 0.4 * (cause == "accident")
        + 0.5 * (amount > 750_000) + 0.25 * (channel == "online")
        # Interactions: red flags that are much stronger together than apart.
        + 1.6 * (contestable & benef_recent)
        + 1.2 * (third_party & ((bank_days > 0) & (bank_days < 60)))
        + 1.0 * (late & (cause == "unknown"))
        + rng.normal(0, 0.5, n_claims)
    )
    fraud = rng.random(n_claims) < _sigmoid(z)

    # ---------------- documents + adjuster notes ----------------
    docs, notes = [], []
    for i, cid in enumerate(claim_ids):
        required = REQUIRED_DOCS[pol.at[i, "product"]]
        p_missing = 0.25 if fraud[i] else 0.07
        for d in required:
            received = rng.random() > p_missing
            docs.append({"claim_id": cid, "doc_type": d,
                         "received_date": (report.iloc[i].date() + timedelta(days=int(rng.integers(0, 10)))) if received else None})
        c = customers.iloc[cust_of_policy[i]]
        bits = [f"Claimant {c.first_name} {c.last_name} (SSN {c.ssn}, phone {c.phone}) reported the claim {CHANNEL_TEXT[channel[i]]}."]
        if (benef_days[i] > 0) and benef_days[i] < 90:
            bits.append(f"Beneficiary was changed {benef_days[i]} days before the date of loss.")
        if third_party[i]:
            bits.append("Payment requested to a third-party account, not the named beneficiary.")
        if late[i]:
            bits.append(f"Claim reported {lag[i]} days after the loss; claimant could not explain the delay.")
        if cause[i] == "unknown":
            bits.append("Cause of loss not yet established.")
        if fraud[i] and rng.random() < 0.3:
            bits.append(rng.choice(["Dates on the physician statement do not match the claim form.",
                                    "Claimant was evasive about the circumstances of the loss.",
                                    "Signature on the claimant statement differs from the policy application."]))
        notes.append({"claim_id": cid, "note_text": " ".join(bits)})

    claims = pd.DataFrame({
        "claim_id": claim_ids, "policy_id": pol["policy_id"], "provider_id": providers.provider_id.to_numpy()[prov_idx],
        "claim_type": pol["product"], "loss_date": loss.dt.date, "report_date": report.dt.date,
        "claim_amount": amount, "cause": cause, "channel": channel,
        "is_fraud": fraud.astype(int),  # label: known only after investigation closes
    })

    tables = {
        "policy_admin/policies.csv": policies, "policy_admin/customers.csv": customers,
        "claims/claims.csv": claims, "servicing/policy_changes.csv": policy_changes,
        "provider_registry/providers.csv": providers, "payments/payments.csv": payments,
        "documents/documents.csv": pd.DataFrame(docs), "documents/adjuster_notes.csv": pd.DataFrame(notes),
    }
    for rel, df in tables.items():
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return {"claims": n_claims, "fraud_rate": round(float(fraud.mean()), 4), "policies": n_pol}


if __name__ == "__main__":
    from claims_copilot.config import get_settings

    s = get_settings()
    print("history:", generate(s.data_dir / "raw", n_claims=60000))
    print("current batch:", generate(s.data_dir / "raw_current", n_claims=3000, seed=99, drifted=True, start="2025-07-01", end="2025-09-30"))
