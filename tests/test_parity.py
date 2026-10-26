"""The exported model must give the same answer in Python, in the JSON evaluator and in the browser code."""

import json
import shutil
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from tripdur import export
from tripdur.features import FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "site" / "model" / "model.json"
FIXTURE = ROOT / "tests" / "fixtures" / "parity.json"


def test_compact_export_matches_lightgbm():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = X[:, 0] * 2 + np.sin(X[:, 1]) + rng.normal(scale=0.1, size=2000)
    booster = lgb.train({"objective": "regression", "num_leaves": 15, "verbose": -1, "seed": 0}, lgb.Dataset(X, y), 40)
    np.testing.assert_allclose(export.predict(export.compact(booster), X), booster.predict(X), rtol=0, atol=1e-12)


@pytest.fixture(scope="module")
def shipped():
    return json.loads(MODEL.read_text()), json.loads(FIXTURE.read_text())


def test_python_features_and_json_trees_reproduce_training(shipped):
    model, fx = shipped
    rows = pd.DataFrame(fx["rows"]).assign(pickup_datetime=lambda d: pd.to_datetime(d.pickup_datetime))
    X = FeatureBuilder.from_params(model["params"]).transform(rows).to_numpy()
    np.testing.assert_allclose(X, np.array(fx["features"]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(export.predict(model, X), fx["prediction_log"], rtol=0, atol=1e-9)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_browser_code_matches_python(shipped):
    _, fx = shipped
    out = subprocess.run(["node", str(ROOT / "tests" / "js_parity.js")], capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    np.testing.assert_allclose(np.array(js["features"]), np.array(fx["features"]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(js["prediction_log"], fx["prediction_log"], rtol=0, atol=1e-9)
