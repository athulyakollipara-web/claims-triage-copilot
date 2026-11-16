import json

import numpy as np
import pandas as pd

from claims_copilot.model.drift import psi, status


def test_model_beats_legacy_rules(data_dir):
    m = json.loads((data_dir / "artifacts" / "metrics.json").read_text())
    assert m["roc_auc"] > 0.7
    assert m["model_precision_at_rules_volume"] > m["rules_precision"]
    assert m["referrals_to_match_rules_catches"] < m["rules_referrals"]
    assert m["siu_threshold"] > m["threshold"]


def test_split_is_time_based(data_dir):
    m = json.loads((data_dir / "artifacts" / "metrics.json").read_text())
    feats = pd.read_parquet(data_dir / "features" / "history")
    assert m["test_rows"] == (feats.report_date >= pd.Timestamp("2025-01-01").date()).sum()


def test_shap_drivers_explain_the_score(service):
    from claims_copilot.model.explain import Explainer
    from claims_copilot.features.build import FEATURES

    cid = service.store.queue(limit=1)[0]["claim_id"]
    row = service.store.feature_row(cid)
    exp = Explainer(service.store.model, FEATURES)
    values = exp.explainer.shap_values(row)
    margin = service.store.model.predict(__import__("xgboost").DMatrix(row, feature_names=FEATURES), output_margin=True)[0]
    assert abs(values.sum() + exp.explainer.expected_value - margin) < 1e-3  # SHAP additivity
    drivers = exp.explain(row)[0]["risk_drivers"]
    assert drivers and all(d["shap"] > 0 for d in drivers)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert status(psi(ref, pd.Series(rng.normal(0, 1, 5000)))) == "stable"
    assert status(psi(ref, pd.Series(rng.normal(1.0, 1, 5000)))) == "drift"
    assert psi(pd.Series([0] * 900 + [1] * 100), pd.Series([0] * 500 + [1] * 500)) > 0.25
