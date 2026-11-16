"""Population Stability Index (PSI): has the incoming claim mix drifted from training?

PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 significant drift (retraining review).
In AWS mode the per-feature PSI values are published as CloudWatch metrics so an
alarm can open a retraining ticket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref, cur = reference.dropna().to_numpy(), current.dropna().to_numpy()
    if len(np.unique(ref)) <= 2:  # binary / one-hot: compare category shares
        edges = np.array([-np.inf, 0.5, np.inf])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / max(1, len(ref))
    c = np.histogram(cur, edges)[0] / max(1, len(cur))
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def status(value: float) -> str:
    return "stable" if value < 0.10 else ("watch" if value < 0.25 else "drift")


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "psi": round(psi(reference[f], current[f]), 4)} for f in features]
    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    df["status"] = df.psi.map(status)
    return df


def publish_to_cloudwatch(report: pd.DataFrame, region: str, namespace: str = "ClaimsCopilot/Drift") -> None:
    import boto3

    cw = boto3.client("cloudwatch", region_name=region)
    data = [{"MetricName": "PSI", "Dimensions": [{"Name": "Feature", "Value": r.feature}], "Value": float(r.psi)}
            for r in report.itertuples()]
    for i in range(0, len(data), 20):
        cw.put_metric_data(Namespace=namespace, MetricData=data[i:i + 20])
