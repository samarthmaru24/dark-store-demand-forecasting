"""Retrain the chosen model (v2) and the prediction-range models.

1. Test-period check: train v2 on data before 2017-08-02 and score the 14 test days. The test period was
   ALREADY SEEN in Stage 5 (v1), so this number is reported next to v1 and is not a fresh unseen score.
   -> reports/test_predictions_v2.csv, reports/test_metrics.json (v1 and v2 side by side)
2. Final models on all data (2015-01-01 .. 2017-08-15) for the dashboard forecast:
   -> models/lgbm_final_v2.txt, lgbm_q10.txt, lgbm_q90.txt, lgbm_final_v2_info.json
The Stage 5 v1 files (models/lgbm_final.txt, reports/test_predictions.csv) are kept unchanged.
Run from the project root: python3 src/train.py (or python3 run_all.py --retrain)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (CHOSEN, MODELS, QUANTILES, RANGE_TARGET, REPORTS, TEST_START, TRAIN_START, baseline, feature_table,
                   fit_point, fit_quantile, metrics, predict_point, predict_quantile, variant, volume_wape, window)

FILES = {"forecast_low": "lgbm_q10.txt", "forecast_high": "lgbm_q90.txt"}


def test_metrics(df):
    # v1: from the Stage 5 predictions (unchanged). v2: retrained here on the same training days.
    v1 = pd.read_csv(REPORTS / "test_predictions.csv", parse_dates=["date"])
    v1 = v1.merge(df[["date", "store_nbr", "family", "is_perishable"]].assign(family=lambda x: x.family.astype(str)),
                  on=["date", "store_nbr", "family"], how="left")
    feats, obj, n_trees = variant(CHOSEN)
    train, test = window(df, TEST_START)
    test["pred"] = predict_point(fit_point(train, feats, n_trees, obj), test, feats, obj)
    test["baseline"] = baseline(df, test, TEST_START)
    for col, a in QUANTILES.items():
        test[col] = predict_quantile(fit_quantile(train, feats, n_trees, a), test, feats)
    lo, hi = np.minimum(test.forecast_low, test.pred), np.maximum(test.forecast_high, test.pred)
    out = test[["date", "store_nbr", "family", "sales", "pred", "baseline"]].assign(
        family=lambda x: x.family.astype(str), forecast_low=lo, forecast_high=hi)
    out.to_csv(REPORTS / "test_predictions_v2.csv", index=False)

    res = {"note": "Test period 2017-08-02..2017-08-15 was first used in Stage 5 (v1); v2 was chosen on "
                   "back-tests only and is scored here afterwards, so v2's test number is not a fresh unseen score.",
           "baseline": metrics(v1, v1.baseline),
           "v1": {**metrics(v1, v1.pred), "volume": volume_wape(v1, "pred")},
           "v2": {**metrics(test, test.pred), "volume": volume_wape(test, "pred"),
                  "range_coverage_pct": float(((test.sales >= lo) & (test.sales <= hi)).mean() * 100)}}
    with open(REPORTS / "test_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    for k in ["baseline", "v1", "v2"]:
        print(f"Test {k:8s} WAPE {res[k]['WAPE']:.2f}% | perishable {res[k]['WAPE_perishable']:.1f}% | "
              f"long-life {res[k]['WAPE_long_life']:.1f}% | bias {res[k]['bias_pct']:+.2f}%")
    print("Range coverage on test (v2):", round(res["v2"]["range_coverage_pct"], 1), "%")
    return res


def train_final(df):
    feats, obj, n_trees = variant(CHOSEN)
    data = df[(df.date >= TRAIN_START) & (df.is_closed == 0)]
    fit_point(data, feats, n_trees, obj).booster_.save_model(str(MODELS / "lgbm_final_v2.txt"))
    for col, a in QUANTILES.items():
        fit_quantile(data, feats, n_trees, a).booster_.save_model(str(MODELS / FILES[col]))
    info = {"variant": CHOSEN, "features": feats, "objective": obj, "tweedie_variance_power": 1.2,
            "range_target": RANGE_TARGET,
            "n_trees": n_trees, "trained_from": str(TRAIN_START.date()), "trained_until": str(df.date.max().date()),
            "quantile_models": {col: {"file": FILES[col], "alpha": a} for col, a in QUANTILES.items()}}
    with open(MODELS / "lgbm_final_v2_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Saved:", ["lgbm_final_v2.txt", *FILES.values(), "lgbm_final_v2_info.json"])


if __name__ == "__main__":
    df = feature_table()
    test_metrics(df)
    train_final(df)
