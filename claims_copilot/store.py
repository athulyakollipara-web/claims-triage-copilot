"""Read access to claim files and model artifacts (local folder; S3 paths in AWS mode via the same layout)."""
from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

import pandas as pd
import xgboost as xgb

from claims_copilot.features.build import FEATURES


def sync_from_s3(bucket: str, prefix: str, dest: Path) -> int:
    """Lambda cold start: copy artifacts, features and claim files from S3 into /tmp."""
    import boto3

    s3 = boto3.client("s3")
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            target = dest / obj["Key"][len(prefix):].lstrip("/")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, obj["Key"], str(target))
                count += 1
    return count


class ClaimStore:
    def __init__(self, data_dir: Path, artifacts_dir: Path):
        self.data_dir, self.artifacts_dir = data_dir, artifacts_dir
        bucket = os.getenv("CLAIMS_S3_BUCKET")
        if bucket and not (artifacts_dir / "model.json").exists():
            sync_from_s3(bucket, os.getenv("CLAIMS_S3_PREFIX", "data_store"), data_dir)

    @cached_property
    def features(self) -> pd.DataFrame:
        frames = [pd.read_parquet(p) for p in (self.data_dir / "features" / "history", self.data_dir / "features" / "current") if p.exists()]
        return pd.concat(frames, ignore_index=True).set_index("claim_id")

    def _raw(self, rel: str) -> pd.DataFrame:
        frames = [pd.read_csv(base / rel) for base in (self.data_dir / "raw", self.data_dir / "raw_current") if (base / rel).exists()]
        return pd.concat(frames, ignore_index=True)

    @cached_property
    def _claims(self) -> pd.DataFrame:
        return self._raw("claims/claims.csv").set_index("claim_id")

    @cached_property
    def _docs(self) -> pd.DataFrame:
        return self._raw("documents/documents.csv")

    @cached_property
    def _notes(self) -> pd.Series:
        return self._raw("documents/adjuster_notes.csv").set_index("claim_id")["note_text"]

    @cached_property
    def _names(self) -> pd.Series:
        pol = self._raw("policy_admin/policies.csv")[["policy_id", "customer_id"]]
        cust = self._raw("policy_admin/customers.csv")
        cust["name"] = cust.first_name + " " + cust.last_name
        return pol.merge(cust[["customer_id", "name"]], on="customer_id").set_index("policy_id")["name"]

    @cached_property
    def model(self) -> xgb.Booster:
        b = xgb.Booster()
        b.load_model(self.artifacts_dir / "model.json")
        return b

    @cached_property
    def metrics(self) -> dict:
        return json.loads((self.artifacts_dir / "metrics.json").read_text())

    def exists(self, claim_id: str) -> bool:
        return claim_id in self._claims.index

    def claim(self, claim_id: str) -> dict:
        c = self._claims.loc[claim_id]
        return {
            "claim_id": claim_id, "policy_id": c.policy_id, "claim_type": c.claim_type, "claim_amount": float(c.claim_amount),
            "loss_date": c.loss_date, "report_date": c.report_date, "cause": c.cause, "channel": c.channel,
            "documents": self._docs[self._docs.claim_id == claim_id].where(pd.notna, None).to_dict("records"),
            "notes": str(self._notes.get(claim_id, "")),
            "claimant_name": str(self._names.get(c.policy_id, "")),
        }

    def feature_row(self, claim_id: str) -> pd.DataFrame:
        return self.features.loc[[claim_id], FEATURES]

    def score(self, rows: pd.DataFrame) -> list[float]:
        return self.model.predict(xgb.DMatrix(rows[FEATURES], feature_names=FEATURES)).tolist()

    def queue(self, batch: str = "current", limit: int = 20) -> list[dict]:
        df = self.features[self.features.index.str.startswith("D" if batch == "current" else "C")]
        scores = self.score(df)
        out = pd.DataFrame({"claim_id": df.index, "score": scores, "claim_type": df.claim_type, "amount": df.claim_amount_raw})
        return out.sort_values("score", ascending=False).head(limit).round(4).to_dict("records")
