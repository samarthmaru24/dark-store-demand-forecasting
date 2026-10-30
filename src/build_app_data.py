"""Copy the small precomputed files the dashboard needs into app/data/.

The app can then be deployed without raw/ or clean/, and it never runs the model.
-> history_last60.parquet, stores.csv, forecast_next14.parquet, reorder_plan.parquet, category_errors.csv,
   conditions_rules.csv, festival_name_map.csv, results.json, forecast_why.parquet (src/explain.py),
   monthly_pattern.csv, test_daily.parquet
monthly_pattern.csv: month-by-month pattern per category over the last 12 full months (Aug 2016 - Jul 2017),
   all stores together, open days only: each month's average units/day vs the 12-month average. Earlier years
   are not used: several categories have recording gaps there (e.g. PRODUCE ~200 units/day in some 2013-2015
   months, ~110,000 in others). Categories that did not sell in every month (BOOKS) are left out.
   One year of history = a past pattern, not a forecast.
test_daily.parquet: locked test days 2017-08-02..15, units per day x category (actual, v2, baseline), all stores.
holidays_window.parquet: Ecuador holidays and events per store from the history start to the forecast end, same
   rules as notebooks/01_cleaning.ipynb (no transferred=True, no "Work Day"; National = all stores, Regional =
   state, Local = city, "Event" = all stores). Used only to shade the dashboard charts.
Run from the project root (after forecast, explain, reorder and results): python3 src/build_app_data.py
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN, REPORTS, RULES = ROOT / "clean", ROOT / "reports", ROOT / "data_rules"
HOLIDAYS = ROOT / "raw" / "store_sales" / "holidays_events.csv"
OUT = ROOT / "app" / "data"
HISTORY_DAYS = 60
COPIES = ["forecast_next14.parquet", "reorder_plan.parquet", "category_errors.csv", "results.json",
          "forecast_why.parquet"]
PATTERN_START, PATTERN_END = pd.Timestamp("2016-08-01"), pd.Timestamp("2017-07-31")


def monthly_pattern(raw):
    # index_pct = how much a month sells above (+) or below (-) the category's average month
    open_ = raw[(raw.is_closed == 0) & raw.date.between(PATTERN_START, PATTERN_END)]
    daily = open_.groupby(["family", "date"], observed=True).sales.sum().reset_index()
    daily["month"] = daily.date.dt.month
    m = daily.groupby(["family", "month"], observed=True).sales.mean().rename("avg_units_day").reset_index()
    stats = m.groupby("family", observed=True).avg_units_day.agg(["mean", "min", "size"])
    keep = stats[(stats["size"] == 12) & (stats["min"] > 0)].index        # sold in every month
    m = m[m.family.isin(keep)].merge(stats["mean"].rename("avg_month").reset_index(), on="family")
    m["index_pct"] = ((m.avg_units_day / m.avg_month - 1) * 100).round(1)
    m["family"] = m.family.astype(str)
    return m[["family", "month", "index_pct", "avg_units_day"]].round({"avg_units_day": 1})


def holidays_window(stores, start, end):
    h = pd.read_csv(HOLIDAYS, parse_dates=["date"])
    h = h[(~h.transferred) & (h.type != "Work Day") & h.date.between(start, end)]
    rows = []
    for st in stores.itertuples():
        match = ((h.locale == "National") | ((h.locale == "Regional") & (h.locale_name == st.state)) |
                 ((h.locale == "Local") & (h.locale_name == st.city)))
        for r in h[match].itertuples():
            rows.append({"store_nbr": st.store_nbr, "date": r.date, "name": r.description, "type": r.type})
    return pd.DataFrame(rows, columns=["store_nbr", "date", "name", "type"]).drop_duplicates()


def test_daily():
    t = pd.read_csv(REPORTS / "test_predictions_v2.csv", parse_dates=["date"])
    return t.groupby(["date", "family"])[["sales", "pred", "baseline"]].sum().round(1).reset_index()


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
    pattern = monthly_pattern(raw)
    pattern.to_csv(OUT / "monthly_pattern.csv", index=False)
    test_daily().to_parquet(OUT / "test_daily.parquet", index=False)
    fc_end = pd.read_parquet(OUT / "forecast_next14.parquet", columns=["date"]).date.max()
    hol = holidays_window(stores, hist.date.min(), fc_end)
    hol.to_parquet(OUT / "holidays_window.parquet", index=False)

    total = 0
    for f in sorted(OUT.iterdir()):
        total += f.stat().st_size
        print(f"{f.name:28s} {f.stat().st_size / 1024:8.0f} KB")
    print(f"History: {hist.date.min().date()} to {hist.date.max().date()} | {len(hist):,} rows")
    print(f"Holidays {hist.date.min().date()} to {fc_end.date()}: {hol[['date', 'name']].drop_duplicates().shape[0]} "
          f"holiday/event days ({len(hol):,} store rows)")
    print(f"Monthly pattern {PATTERN_START.date()} to {PATTERN_END.date()}: {pattern.family.nunique()} categories")
    print(f"Total app/data: {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build()
