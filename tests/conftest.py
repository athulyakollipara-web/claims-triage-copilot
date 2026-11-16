import os
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    """Small end-to-end build: generate -> PySpark features -> train. Runs once per session."""
    d = tmp_path_factory.mktemp("claims")
    os.environ.update({"CLAIMS_DATA_DIR": str(d), "CLAIMS_MODE": "local", "CLAIMS_LLM": "template"})
    os.environ.pop("CLAIMS_ARTIFACTS_DIR", None)
    from claims_copilot.config import get_settings
    from claims_copilot.data.generate import generate
    from claims_copilot.features.build import run
    from claims_copilot.model.train import SPLIT_DATE, save, train

    s = get_settings()
    generate(d / "raw", n_claims=12000)
    generate(d / "raw_current", n_claims=800, seed=99, drifted=True, start="2025-07-01", end="2025-09-30")
    run(d / "raw", d / "features" / "history")
    run(d / "raw_current", d / "features" / "current")
    feats = pd.read_parquet(d / "features" / "history")
    booster, report = train(feats)
    save(booster, report, feats[feats.report_date < pd.Timestamp(SPLIT_DATE).date()], s.artifacts_dir)
    return d


@pytest.fixture(scope="session")
def service(data_dir):
    from claims_copilot.service import CopilotService

    return CopilotService()
