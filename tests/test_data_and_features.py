import pandas as pd

from claims_copilot.features.build import FEATURES


def test_six_source_systems_written(data_dir):
    systems = {p.name for p in (data_dir / "raw").iterdir() if p.is_dir()}
    assert systems == {"policy_admin", "claims", "servicing", "provider_registry", "payments", "documents"}


def test_one_feature_row_per_claim(data_dir):
    feats = pd.read_parquet(data_dir / "features" / "history")
    claims = pd.read_csv(data_dir / "raw" / "claims" / "claims.csv")
    assert len(feats) == len(claims) and feats.claim_id.is_unique
    assert len(FEATURES) >= 35
    assert not feats[FEATURES].isna().any().any()


def test_features_have_expected_meaning(data_dir):
    feats = pd.read_parquet(data_dir / "features" / "history")
    assert ((feats.policy_age_days < 0) == (feats.loss_before_issue == 1)).all()
    assert ((feats.policy_age_days < 730) == (feats.contestable == 1)).all()
    assert feats.docs_missing_ratio.between(0, 1).all()
    # one-hot groups sum to exactly one
    for prefix in ("claim_type_", "cause_", "channel_"):
        cols = [c for c in FEATURES if c.startswith(prefix)]
        assert (feats[cols].sum(axis=1) == 1).all()


def test_risk_signals_raise_fraud_rate(data_dir):
    feats = pd.read_parquet(data_dir / "features" / "history")
    for signal in ("beneficiary_change_90d", "payee_third_party", "contestable"):
        assert feats[feats[signal] == 1].is_fraud.mean() > feats[feats[signal] == 0].is_fraud.mean()


def test_synthetic_pii_is_clearly_fake(data_dir):
    customers = pd.read_csv(data_dir / "raw" / "policy_admin" / "customers.csv")
    assert customers.ssn.str.startswith("9").all()  # 900-series SSNs are never issued
    assert customers.email.str.endswith("@example.com").all()
