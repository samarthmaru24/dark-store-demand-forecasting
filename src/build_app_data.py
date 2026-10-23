"""Copy the small precomputed files the dashboard needs into app/data/.

The app can then be deployed without raw/ or clean/, and it never runs the model.
-> history_last60.parquet, stores.csv, forecast_next14.parquet, reorder_plan.parquet, category_errors.csv,
   conditions_rules.csv, festival_name_map.csv, results.json
Run from the project root (after forecast, reorder and results): python3 src/build_app_data.py
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN, REPORTS, RULES = ROOT / "clean", ROOT / "reports", ROOT / "data_rules"
OUT = ROOT / "app" / "data"
HISTORY_DAYS = 60
COPIES = ["forecast_next14.parquet", "reorder_plan.parquet", "category_errors.csv", "results.json"]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in ["forecast_next14.csv", "reorder_plan.csv", "eval_summary.json"]:     # earlier formats
        (OUT / old).unlink(missing_ok=True)
    raw = pd.read_parquet(CLEAN / "store_sales_clean.parquet",
                          columns=["date", "store_nbr", "family", "sales", "is_closed", "never_sold",
                                   "city", "state", "store_type"])
    raw = raw[raw.never_sold == 0]
    last = raw.date.max()
    hist = raw[raw.date > last - pd.Timedelta(days=HISTORY_DAYS)][["store_nbr", "family", "date", "sales", "is_closed"]]
    hist.assign(family=hist.family.astype(str)).to_parquet(OUT / "history_last60.parquet", index=False)
    stores = raw[raw.date == last][["store_nbr", "city", "state", "store_type"]].drop_duplicates().sort_values("store_nbr")
    stores.to_csv(OUT / "stores.csv", index=False)

    for f in COPIES:
        shutil.copy(REPORTS / f, OUT / f)
    for f in sorted(RULES.glob("*.csv")):
        shutil.copy(f, OUT / f.name)

    total = 0
    for f in sorted(OUT.iterdir()):
        total += f.stat().st_size
        print(f"{f.name:28s} {f.stat().st_size / 1024:8.0f} KB")
    print(f"History: {hist.date.min().date()} to {hist.date.max().date()} | {len(hist):,} rows")
    print(f"Total app/data: {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build()
