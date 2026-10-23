# Stage 6c: copy the small precomputed files the dashboard needs into app/data/,
# so the app can be deployed without raw/ or clean/. The app never runs the model.
# Run from the project root (after src/forecast.py and src/reorder.py): python3 src/build_app_data.py
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN, REPORTS, RULES = ROOT / "clean", ROOT / "reports", ROOT / "data_rules"
OUT = ROOT / "app" / "data"
HISTORY_DAYS = 60


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(CLEAN / "store_sales_clean.parquet",
                          columns=["date", "store_nbr", "family", "sales", "is_closed", "never_sold",
                                   "city", "state", "store_type"])
    raw = raw[raw.never_sold == 0]
    last = raw.date.max()
    hist = raw[raw.date > last - pd.Timedelta(days=HISTORY_DAYS)][["store_nbr", "family", "date", "sales", "is_closed"]]
    hist.assign(family=hist.family.astype(str)).to_parquet(OUT / "history_last60.parquet", index=False)
    stores = raw[raw.date == last][["store_nbr", "city", "state", "store_type"]].drop_duplicates().sort_values("store_nbr")
    stores.to_csv(OUT / "stores.csv", index=False)

    for f in ["forecast_next14.csv", "reorder_plan.csv", "category_errors.csv"]:
        shutil.copy(REPORTS / f, OUT / f)
    for f in sorted(RULES.glob("*.csv")):
        shutil.copy(f, OUT / f.name)

    # Test-period accuracy for the KPI row and the shelf-life panel (same numbers as 04_evaluation_run)
    info = json.load(open(ROOT / "models" / "lgbm_final_info.json"))
    test = pd.read_csv(REPORTS / "test_predictions.csv").merge(
        pd.read_csv(REPORTS / "category_errors.csv")[["family", "is_perishable"]], on="family")
    wape = lambda g: float(np.abs(g.sales - g.pred).sum() / g.sales.sum() * 100)
    summary = {"test_WAPE": info["test_WAPE"], "baseline_test_WAPE": info["baseline_test_WAPE"],
               "test_period": f"{test.date.min()} to {test.date.max()}",
               "WAPE_perishable": round(wape(test[test.is_perishable == 1]), 1),
               "WAPE_long_life": round(wape(test[test.is_perishable == 0]), 1)}
    json.dump(summary, open(OUT / "eval_summary.json", "w"), indent=2)
    print("eval_summary:", summary)

    total = 0
    for f in sorted(OUT.iterdir()):
        total += f.stat().st_size
        print(f"{f.name:28s} {f.stat().st_size / 1024:8.0f} KB")
    print(f"History: {hist.date.min().date()} to {hist.date.max().date()} | {len(hist):,} rows")
    print(f"Total app/data: {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build()
