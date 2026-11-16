"""End-to-end evaluation, logged to MLflow.

1. Model: PR-AUC / ROC-AUC on the 2025 hold-out, vs. logistic regression and the legacy rules.
2. Explanations: are the SHAP drivers consistent with the model? (top driver's SHAP
   sign agrees with the score moving when the feature is reset to its training median)
3. Narratives: faithfulness of summaries for a sample of high-risk claims
   (template offline; set CLAIMS_LLM=bedrock to evaluate a real model).
4. Drift: PSI of the current batch against training.

    python eval/run_eval.py [--cases 50]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from claims_copilot.config import get_settings
from claims_copilot.features.build import FEATURES
from claims_copilot.model.drift import drift_report
from claims_copilot.service import CopilotService

HERE = Path(__file__).parent


def explanation_consistency(svc: CopilotService, claim_ids: list[str]) -> float:
    """Resetting a claim's top risk driver to the training median should lower its score."""
    ref = pd.read_parquet(svc.settings.artifacts_dir / "reference_sample.parquet")
    medians = ref[FEATURES].median()
    from claims_copilot.model.explain import Explainer

    explainer = Explainer(svc.store.model, FEATURES)
    agree = 0
    for cid in claim_ids:
        row = svc.store.feature_row(cid)
        drivers = explainer.explain(row)[0]["risk_drivers"]
        if not drivers:
            continue
        top = drivers[0]["feature"]
        changed = row.copy()
        changed[top] = medians[top]
        agree += svc.store.score(changed)[0] < svc.store.score(row)[0]
    return agree / max(1, len(claim_ids))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=50)
    args = p.parse_args()
    s = get_settings()
    svc = CopilotService(s)

    model = svc.store.metrics
    top = [r["claim_id"] for r in svc.store.queue(limit=args.cases)]
    consistency = explanation_consistency(svc, top)

    faithful, latencies, actions = [], [], {}
    for cid in top:
        t = time.perf_counter()
        r = svc.triage(cid, "eval")
        latencies.append(time.perf_counter() - t)
        # "first try": the model's own text passed, without falling back to the template
        faithful.append(r["faithfulness"]["faithful"] and "rejected" not in r["narrative_model"] and "fallback" not in r["narrative_model"])
        actions[r["recommendation"]["action"]] = actions.get(r["recommendation"]["action"], 0) + 1
        svc.pending.pop(cid, None)

    drift = drift_report(pd.read_parquet(s.artifacts_dir / "reference_sample.parquet"),
                         pd.read_parquet(s.data_dir / "features" / "current"), FEATURES)
    summary = {
        "model": {k: model[k] for k in ("pr_auc", "roc_auc", "logistic_pr_auc", "logistic_roc_auc", "rules_precision",
                                        "model_precision_at_rules_volume", "referrals_to_match_rules_catches", "rules_referrals",
                                        "false_positive_reduction_at_rules_recall", "model_recall_at_capacity")},
        "explanations": {"top_driver_consistency": round(consistency, 4), "cases": len(top)},
        "narratives": {"llm": s.llm, "faithful_first_try_rate": round(float(np.mean(faithful)), 4), "cases": len(top),
                       "mean_triage_latency_s": round(float(np.mean(latencies)), 3)},
        "actions_for_top_cases": actions,
        "drift": {"features_watch": int((drift.status == "watch").sum()), "features_drift": int((drift.status == "drift").sum()),
                  "max_psi_feature": drift.iloc[0].feature, "max_psi": float(drift.iloc[0].psi)},
    }
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"eval-{s.llm}.json").write_text(json.dumps(summary, indent=2))

    import mlflow

    mlflow.set_experiment("claims-copilot-eval")
    with mlflow.start_run(run_name=f"eval-{s.llm}"):
        flat = {f"{g}.{k}": v for g, d in summary.items() if isinstance(d, dict) for k, v in d.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(flat)
        mlflow.log_param("llm", s.llm)
        mlflow.log_artifact(str(out / f"eval-{s.llm}.json"))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
