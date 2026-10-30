"""Checks that need only files in git (no raw/ or clean/): saved models, saved reports, dashboard.

These run in GitHub Actions on every push. They check that the committed models load and predict, that the
committed reports agree with each other, and that the dashboard starts without errors. They do not measure
accuracy (that needs the data: python3 run_all.py, numbers in reports/results.json).
Run from the project root: python3 -m pytest tests -q
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
INFO = json.loads((MODELS / "lgbm_final_v2_info.json").read_text())
RANGE_FILES = [m["file"] for m in INFO["quantile_models"].values()]


@pytest.mark.parametrize("file", ["lgbm_final_v2.txt", *RANGE_FILES])
def test_saved_model_loads_with_the_recorded_features(file):
    booster = lgb.Booster(model_file=str(MODELS / file))
    assert booster.feature_name() == INFO["features"]
    assert booster.num_trees() == INFO["n_trees"]


def test_saved_model_predicts_one_value_per_row():
    booster = lgb.Booster(model_file=str(MODELS / "lgbm_final_v2.txt"))
    X = np.zeros((5, len(INFO["features"])))          # plain array: category columns are read as codes
    pred = booster.predict(X)
    assert pred.shape == (5,) and np.isfinite(pred).all() and (pred >= 0).all()   # Tweedie: units, never negative


def test_saved_model_rejects_wrong_number_of_inputs():
    booster = lgb.Booster(model_file=str(MODELS / "lgbm_final_v2.txt"))
    with pytest.raises(lgb.basic.LightGBMError):
        booster.predict(np.zeros((5, len(INFO["features"]) - 1)))


def test_results_json_matches_saved_outputs():
    res = json.loads((REPORTS / "results.json").read_text())
    fc = pd.read_parquet(REPORTS / "forecast_next14.parquet")
    assert res["chosen_model"] == INFO["variant"] == "v2"
    assert res["forecast_2017-08-16_to_2017-08-29"]["total_units"] == round(fc.forecast.sum())
    plan = pd.read_parquet(REPORTS / "reorder_plan.parquet")
    counts = plan.status.value_counts()
    assert res["reorder_plan"]["rows"] == len(plan)
    assert all(res["reorder_plan"][s] == counts.get(s, 0) for s in ["Urgent", "Reorder", "OK"])


def test_dashboard_runs_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
