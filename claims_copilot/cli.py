"""Command line entry point.

    python -m claims_copilot.cli pipeline        # generate data -> features (PySpark) -> train -> drift report
    python -m claims_copilot.cli queue           # top-risk claims in the current batch
    python -m claims_copilot.cli triage D001871  # full copilot run for one claim
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from claims_copilot.config import get_settings


def pipeline() -> int:
    from dataclasses import asdict

    from claims_copilot.data.generate import generate
    from claims_copilot.features.build import FEATURES, run
    from claims_copilot.model.drift import drift_report
    from claims_copilot.model.train import SPLIT_DATE, save, train

    s = get_settings()
    print("data (history):", generate(s.data_dir / "raw", n_claims=60000))
    print("data (current):", generate(s.data_dir / "raw_current", n_claims=3000, seed=99, drifted=True, start="2025-07-01", end="2025-09-30"))
    print("features (history):", run(s.data_dir / "raw", s.data_dir / "features" / "history"))
    print("features (current):", run(s.data_dir / "raw_current", s.data_dir / "features" / "current"))
    feats = pd.read_parquet(s.data_dir / "features" / "history")
    booster, report = train(feats, s.review_capacity)
    save(booster, report, feats[feats.report_date < pd.Timestamp(SPLIT_DATE).date()], s.artifacts_dir)
    print("model:", json.dumps(asdict(report), indent=2))
    drift = drift_report(pd.read_parquet(s.artifacts_dir / "reference_sample.parquet"),
                         pd.read_parquet(s.data_dir / "features" / "current"), FEATURES)
    drift.to_csv(s.artifacts_dir / "drift_report.csv", index=False)
    print("drift (top 5):\n" + drift.head(5).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="claims-copilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pipeline")
    q = sub.add_parser("queue")
    q.add_argument("--limit", type=int, default=10)
    t = sub.add_parser("triage")
    t.add_argument("claim_id")
    t.add_argument("--investigator", default="inv_demo")
    args = p.parse_args(argv)
    if args.cmd == "pipeline":
        return pipeline()
    from claims_copilot.service import CopilotService

    svc = CopilotService()
    if args.cmd == "queue":
        for row in svc.store.queue(limit=args.limit):
            print(f"{row['claim_id']}  score={row['score']:.3f}  {row['claim_type']:<15} ${row['amount']:>12,.0f}")
        return 0
    r = svc.triage(args.claim_id, args.investigator)
    print(f"CLAIM {r['claim_id']}  score={r['score']}  tier={r['recommendation']['tier']}  action={r['recommendation']['action']}\n")
    print("WHY (SHAP drivers)")
    for d in r["risk_drivers"]:
        print(f"  + {d['description']}  (shap {d['shap']:+.2f})")
    for d in r["mitigating"]:
        print(f"  - {d['description']}  (shap {d['shap']:+.2f})")
    print("\nEVIDENCE")
    print(f"  missing documents: {', '.join(r['evidence']['missing_documents']) or 'none'}")
    print(f"  note flags: {', '.join(r['evidence']['note_flags']) or 'none'}")
    print(f"  notes (PII redacted: {', '.join(r['pii_found']) or 'none'}): {r['redacted_notes']}")
    print("\nNEXT STEPS")
    for step in r["recommendation"]["next_steps"]:
        print(f"  - {step}")
    print(f"\nSUMMARY ({r['narrative_model']}, faithful={r['faithfulness']['faithful']})\n  {r['narrative']}")
    print("\nstatus: awaiting investigator decision")
    return 0


if __name__ == "__main__":
    sys.exit(main())
