"""Train the XGBoost fraud-risk model with a time-based split and compare it with the rules it replaces.

Why time-based: the model scores *future* claims, so it's trained on 2023-2024
and evaluated on 2025 claims it never saw. A random split would leak patterns
from the future and overstate accuracy.

Why compare with rules: claims teams already refer claims to investigators by
simple rules (contestable period, large amount, late report). The question the
business asks is "with the same number of referrals, how many more real fraud
cases do we catch, and how many fewer innocent claimants get investigated?"
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from claims_copilot.features.build import FEATURES

SPLIT_DATE = "2025-01-01"
# No scale_pos_weight: it inflated scores and overfit the rare class in testing, and
# unweighted training keeps the scores usable as probabilities for thresholds.
PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.03,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10, "lambda": 1.0, "seed": 7,
}


@dataclass
class TrainReport:
    train_rows: int
    test_rows: int
    test_fraud_rate: float
    pr_auc: float
    roc_auc: float
    referrals: int
    model_precision_at_capacity: float
    model_recall_at_capacity: float
    rules_referrals: int
    rules_precision: float
    rules_recall: float
    model_precision_at_rules_volume: float
    referrals_to_match_rules_catches: int
    logistic_pr_auc: float
    logistic_roc_auc: float
    false_positive_reduction_at_rules_recall: float
    threshold: float
    siu_threshold: float
    best_iteration: int


def rules_baseline(df: pd.DataFrame) -> np.ndarray:
    """The legacy referral rules this model is meant to improve on."""
    return ((df["contestable"] == 1) | (df["claim_amount"] > 750_000) | (df["late_report"] == 1)).to_numpy()


def logistic_baseline(fit: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """A simpler model to beat: if XGBoost can't outperform this, ship the simpler one."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    lr.fit(fit[FEATURES].fillna(0), fit.is_fraud)
    return lr.predict_proba(test[FEATURES].fillna(0))[:, 1]


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-scores)[:k]] = True
    return flagged


def train(features: pd.DataFrame, capacity: float = 0.10) -> tuple[xgb.Booster, TrainReport]:
    df = features.sort_values("report_date").reset_index(drop=True)
    train_df = df[df.report_date < pd.Timestamp(SPLIT_DATE).date()]
    test_df = df[df.report_date >= pd.Timestamp(SPLIT_DATE).date()]
    cut = int(len(train_df) * 0.85)  # last 15% of the training period for early stopping
    fit, val = train_df.iloc[:cut], train_df.iloc[cut:]

    dfit = xgb.DMatrix(fit[FEATURES], label=fit.is_fraud, feature_names=FEATURES)
    dval = xgb.DMatrix(val[FEATURES], label=val.is_fraud, feature_names=FEATURES)
    booster = xgb.train(PARAMS, dfit, num_boost_round=1500, evals=[(dval, "val")],
                        early_stopping_rounds=100, verbose_eval=False)
    best = booster.best_iteration
    booster = booster[: best + 1]  # drop trees after the best round so scoring and SHAP use the same model

    y = test_df.is_fraud.to_numpy()
    p = booster.predict(xgb.DMatrix(test_df[FEATURES], feature_names=FEATURES))
    lr_p = logistic_baseline(fit, test_df)
    k = int(len(test_df) * capacity)
    flagged = _top_k(p, k)
    rules = rules_baseline(test_df)
    model_same_volume = _top_k(p, int(rules.sum()))

    def precision(mask): return float(y[mask].mean()) if mask.any() else 0.0
    def recall(mask): return float(y[mask].sum() / max(1, y.sum()))
    # Same number of fraud cases caught as the rules: how many referrals does the model need?
    order = np.argsort(-p)
    caught = np.cumsum(y[order])
    rules_tp = int((rules & (y == 1)).sum())
    k_match = int(np.searchsorted(caught, rules_tp) + 1) if rules_tp else 0
    fp_rules = int((rules & (y == 0)).sum())
    fp_model = int(k_match - caught[k_match - 1]) if k_match else 0

    report = TrainReport(
        train_rows=len(train_df), test_rows=len(test_df), test_fraud_rate=round(float(y.mean()), 4),
        pr_auc=round(float(average_precision_score(y, p)), 4), roc_auc=round(float(roc_auc_score(y, p)), 4),
        referrals=k, model_precision_at_capacity=round(precision(flagged), 4), model_recall_at_capacity=round(recall(flagged), 4),
        rules_referrals=int(rules.sum()), rules_precision=round(precision(rules), 4), rules_recall=round(recall(rules), 4),
        model_precision_at_rules_volume=round(precision(model_same_volume), 4),
        referrals_to_match_rules_catches=k_match,
        logistic_pr_auc=round(float(average_precision_score(y, lr_p)), 4),
        logistic_roc_auc=round(float(roc_auc_score(y, lr_p)), 4),
        false_positive_reduction_at_rules_recall=round((fp_rules - fp_model) / max(1, fp_rules), 4),
        threshold=round(float(np.sort(p)[::-1][k - 1]), 6),
        siu_threshold=round(float(np.sort(p)[::-1][max(0, int(len(p) * 0.02) - 1)]), 6),  # top 2% -> special investigations
        best_iteration=int(best),
    )
    return booster, report


def save(booster: xgb.Booster, report: TrainReport, reference: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(out / "model.json")
    (out / "metrics.json").write_text(json.dumps(asdict(report), indent=2))
    (out / "features.json").write_text(json.dumps(FEATURES))
    reference[FEATURES].sample(min(5000, len(reference)), random_state=1).to_parquet(out / "reference_sample.parquet")


def log_to_mlflow(booster: xgb.Booster, report: TrainReport, out: Path) -> None:
    import mlflow

    mlflow.set_experiment("claims-fraud-risk")
    with mlflow.start_run(run_name="xgboost-time-split"):
        mlflow.log_params({**PARAMS, "split_date": SPLIT_DATE, "n_features": len(FEATURES)})
        mlflow.log_metrics({k: v for k, v in asdict(report).items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(out / "metrics.json"))
        try:
            import mlflow.xgboost

            mlflow.xgboost.log_model(booster, name="model", registered_model_name="claims-fraud-risk")
        except Exception as e:  # registry needs a database-backed tracking server
            print(f"model registry skipped: {e}")


if __name__ == "__main__":
    from claims_copilot.config import get_settings

    s = get_settings()
    feats = pd.read_parquet(s.data_dir / "features" / "history")
    model, rep = train(feats, s.review_capacity)
    save(model, rep, feats[feats.report_date < pd.Timestamp(SPLIT_DATE).date()], s.artifacts_dir)
    print(json.dumps(asdict(rep), indent=2))
