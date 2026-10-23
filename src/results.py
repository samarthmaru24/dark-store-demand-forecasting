"""Collect every reported number into reports/results.json (single source of truth for README, report, app).

Reads only files written by code: models/lgbm_v1_info.json (validation), reports/test_metrics.json (test v1/v2),
reports/backtest*.csv, reports/christmas_backtest.json, reports/forecast_next14.parquet, reports/reorder_plan.parquet.
Run from the project root: python3 src/results.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN, MODELS, REPORTS = ROOT / "clean", ROOT / "models", ROOT / "reports"


def load(name):
    with open(name) as f:
        return json.load(f)


def r(x, n=2):
    return round(float(x), n)


def build():
    valid = load(MODELS / "lgbm_v1_info.json")["results_valid"]
    test = load(REPORTS / "test_metrics.json")
    xmas = load(REPORTS / "christmas_backtest.json")
    bt = pd.read_csv(REPORTS / "backtest.csv")
    rg = pd.read_csv(REPORTS / "backtest_ranges.csv")
    ex = pd.read_csv(REPORTS / "backtest_experiment.csv")
    fc = pd.read_parquet(REPORTS / "forecast_next14.parquet")
    plan = pd.read_parquet(REPORTS / "reorder_plan.parquet")
    hist = pd.read_parquet(CLEAN / "store_sales_clean.parquet", columns=["date", "sales", "never_sold"])
    last14 = hist[(hist.never_sold == 0) & (hist.date > hist.date.max() - pd.Timedelta(days=14))].sales.sum()

    def m(d):
        return {k: r(d[k]) for k in ["WAPE", "WAPE_perishable", "WAPE_long_life", "bias_pct"]}

    exp = ex.groupby("variant", sort=False)[["WAPE", "WAPE_perishable", "WAPE_long_life", "bias_pct"]].mean()
    res = {
        "chosen_model": "v2",
        "model_versions": {
            "v1": "LightGBM, 31 features, log1p(sales), 140 trees (Stage 5)",
            "v2": "v1 + days_since_holiday + sales_lag_364, Tweedie objective (variance_power 1.2) on raw sales"},
        "validation_2017-07-19_to_2017-08-01": {k: r(v["WAPE"]) for k, v in valid.items()},
        "test_2017-08-02_to_2017-08-15": {
            "note": test["note"],
            "baseline": m(test["baseline"]),
            "v1": {**m(test["v1"]), "volume": {k: r(v, 1) for k, v in test["v1"]["volume"].items()}},
            "v2": {**m(test["v2"]), "volume": {k: r(v, 1) for k, v in test["v2"]["volume"].items()},
                   "range_coverage_pct": r(test["v2"]["range_coverage_pct"], 1)},
            "v1_improvement_vs_baseline_pct": r((1 - test["v1"]["WAPE"] / test["baseline"]["WAPE"]) * 100, 1),
            "v2_improvement_vs_baseline_pct": r((1 - test["v2"]["WAPE"] / test["baseline"]["WAPE"]) * 100, 1)},
        "backtest_6_windows": {
            "windows": [str(s) for s in bt.window_start],
            "experiment_mean": {v: {k: r(x) for k, x in row.items()} for v, row in exp.iterrows()},
            "chosen_v2": {"mean": r(bt.model_WAPE.mean()), "min": r(bt.model_WAPE.min()), "max": r(bt.model_WAPE.max()),
                          "baseline_mean": r(bt.baseline_WAPE.mean())},
            "range_coverage_pct": {"target": 80, "mean": r(rg.coverage_pct.mean(), 1), "min": r(rg.coverage_pct.min(), 1),
                                   "max": r(rg.coverage_pct.max(), 1),
                                   "perishable": r(rg.coverage_perishable_pct.mean(), 1),
                                   "long_life": r(rg.coverage_long_life_pct.mean(), 1)}},
        "christmas_backtest": {
            "window": xmas["window"], "WAPE_baseline": r(xmas["WAPE_baseline"], 1),
            "v1": {"WAPE": r(xmas["v1"]["WAPE"], 1), "units_vs_actual_pct": r((xmas["v1"]["units"] / xmas["units_actual"] - 1) * 100, 1)},
            "v2": {"WAPE": r(xmas["v2"]["WAPE"], 1), "units_vs_actual_pct": r((xmas["v2"]["units"] / xmas["units_actual"] - 1) * 100, 1)},
            "units_actual": round(xmas["units_actual"]),
            "categories": {f: {k: round(v) for k, v in d.items()} for f, d in xmas["categories"].items()}},
        "forecast_2017-08-16_to_2017-08-29": {
            "model": "v2", "total_units": round(fc.forecast.sum()), "last_14_days_actual_units": round(last14),
            "change_vs_last_14_days_pct": r((fc.forecast.sum() / last14 - 1) * 100, 1),
            "perishable_share_pct": r(fc[fc.is_perishable == 1].forecast.sum() / fc.forecast.sum() * 100, 1)},
        "reorder_plan": {"rows": len(plan), **{s: int((plan.status == s).sum()) for s in ["Urgent", "Reorder", "OK"]},
                         "stock": "SIMULATED"},
    }
    with open(REPORTS / "results.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    res = build()
    t = res["test_2017-08-02_to_2017-08-15"]
    b = res["backtest_6_windows"]
    print(f"Test WAPE: baseline {t['baseline']['WAPE']}% | v1 {t['v1']['WAPE']}% | v2 {t['v2']['WAPE']}% (test already seen)")
    print(f"Back-test v2: mean {b['chosen_v2']['mean']}% (min {b['chosen_v2']['min']}, max {b['chosen_v2']['max']}) | "
          f"range coverage {b['range_coverage_pct']['mean']}%")
    print("Saved:", REPORTS / "results.json")
