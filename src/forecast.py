# Stage 6a: forecast the next H = 14 days after the last day in the data, with models/lgbm_final.txt.
# Uses src/features.py unchanged. Every past-sales clue is at least 14 days old, so the 14 future
# days only need sales up to the last data day: no recursive forecasting.
# Assumptions (say so in the dashboard): onpromotion = planned promotions from raw/store_sales/test.csv
# (the store knows its promotion calendar; 0 where no plan), all stores open, no earthquake.
# Run from the project root: python3 src/forecast.py
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import build_features, H

CLEAN, MODELS, REPORTS = ROOT / "clean", ROOT / "models", ROOT / "reports"
RAW = ROOT / "raw" / "store_sales"
EXTRA_DAYS = 60  # calendar-only rows so days_to_holiday can see the next holiday (capped at 60)


def holiday_flags(rows, holidays):
    # Same rules as notebooks/01_cleaning.ipynb (cell 5). rows: date, state, city
    h = holidays[(~holidays.transferred) & (holidays.type != "Work Day")]
    ev = h[h.type == "Event"][["date"]].drop_duplicates().assign(is_event=1)
    h = h[h.type != "Event"]
    nat = h[h.locale == "National"][["date"]].drop_duplicates().assign(is_national_holiday=1)
    reg = (h[h.locale == "Regional"][["date", "locale_name"]].drop_duplicates()
           .rename(columns={"locale_name": "state"}).assign(is_regional_holiday=1))
    loc = (h[h.locale == "Local"][["date", "locale_name"]].drop_duplicates()
           .rename(columns={"locale_name": "city"}).assign(is_local_holiday=1))
    out = (rows.merge(ev, on="date", how="left").merge(nat, on="date", how="left")
           .merge(reg, on=["date", "state"], how="left").merge(loc, on=["date", "city"], how="left"))
    cols = ["is_event", "is_national_holiday", "is_regional_holiday", "is_local_holiday"]
    out[cols] = out[cols].fillna(0).astype(int)
    out["is_holiday"] = out[["is_national_holiday", "is_regional_holiday", "is_local_holiday"]].max(axis=1)
    return out


def future_rows(raw, holidays, promos, days):
    # One row per (store, category) pair and future day. sales / oil / transactions unknown (NaN).
    last = raw.date.max()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=days)
    static = ["store_nbr", "family", "city", "state", "store_type", "cluster", "is_perishable"]
    pairs = raw[raw.date == last][static]
    fut = pairs.merge(pd.DataFrame({"date": dates}), how="cross")
    fut = holiday_flags(fut, holidays)
    fut["sales"] = np.nan
    fut = fut.merge(promos, on=["date", "store_nbr", "family"], how="left")
    fut["onpromotion"] = fut["onpromotion"].fillna(0).astype(int)   # no plan -> 0
    fut["is_closed"] = 0
    fut["dcoilwtico"] = np.nan
    fut["transactions"] = np.nan
    fut["is_payday"] = ((fut.date.dt.day == 15) | fut.date.dt.is_month_end).astype(int)
    fut["is_quake"] = 0
    fut["never_sold"] = 0
    return fut[raw.columns]


def make_forecast():
    info = json.load(open(MODELS / "lgbm_final_info.json"))
    booster = lgb.Booster(model_file=str(MODELS / "lgbm_final.txt"))
    raw = pd.read_parquet(CLEAN / "store_sales_clean.parquet")
    raw = raw[raw.never_sold == 0].copy()
    holidays = pd.read_csv(RAW / "holidays_events.csv", parse_dates=["date"])
    weather = pd.read_csv(ROOT / "raw" / "weather_ecuador" / "weather.csv", parse_dates=["date"])
    # test.csv is used ONLY for planned onpromotion (it has no sales)
    promos = pd.read_csv(RAW / "test.csv", parse_dates=["date"])[["date", "store_nbr", "family", "onpromotion"]]

    last = raw.date.max()
    full = pd.concat([raw, future_rows(raw, holidays, promos, H + EXTRA_DAYS)], ignore_index=True)
    df = build_features(full, weather)
    fut = df[(df.date > last) & (df.date <= last + pd.Timedelta(days=H))].copy()
    fut["forecast"] = np.expm1(booster.predict(fut[info["features"]])).clip(0)
    return df, fut, info, last


if __name__ == "__main__":
    df, fut, info, last = make_forecast()
    print("Rows with a planned promotion value:", f"{fut.onpromotion.gt(0).sum():,}", "of", f"{len(fut):,}",
          "| avg planned onpromotion:", round(fut.onpromotion.mean(), 2))
    out = fut[["date", "store_nbr", "family", "is_perishable", "onpromotion", "forecast"]].copy()
    out["family"] = out.family.astype(str)
    out["forecast"] = out.forecast.round(2)
    out.to_csv(REPORTS / "forecast_next14.csv", index=False)
    print("Forecast:", out.date.min().date(), "to", out.date.max().date(), "|", f"{len(out):,} rows",
          "|", out.store_nbr.nunique(), "stores |", out.family.nunique(), "categories")
    print("Saved:", REPORTS / "forecast_next14.csv")
