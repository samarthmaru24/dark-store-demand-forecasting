"""Reorder plan per store x category, from reports/forecast_next14.parquet.

order = forecast over window + buffer - current stock, minimum 0 (rounded up to whole units).
Window: perishables 3 days, long-life 14 days.
Buffer = sum over the window of max(forecast_high - forecast, 0) (90% quantile model), capped at 50% of the
         window forecast; tiny categories (< 1 unit/day per store) get a buffer of at least 1 unit.
Stock is SIMULATED (the dataset has no stock): recent 14-day average x U(0.3, 1.5) x window_days, seed 42.
Status: "Urgent" if days_of_cover < 2 (only when the average daily forecast is >= 1 unit),
        else "Reorder" if order_qty > 0, else "OK".
-> reports/reorder_plan.parquet (+ .csv copy for download)
Run from the project root (after src/forecast.py): python3 src/reorder.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN, REPORTS = ROOT / "clean", ROOT / "reports"
WINDOW = {1: 3, 0: 14}      # is_perishable -> window_days
BUFFER_CAP = 0.50
LOW_SELLER = 1.0            # units/day per store
URGENT_DAYS = 2
SEED = 42
KEYS = ["store_nbr", "family"]


def simulated_stock(pairs, last):
    # Recent average = mean sales on open days in the last 14 days. pairs: store_nbr, family, window_days
    raw = pd.read_parquet(CLEAN / "store_sales_clean.parquet",
                          columns=["date", "store_nbr", "family", "sales", "is_closed", "never_sold"])
    recent = raw[(raw.date > last - pd.Timedelta(days=14)) & (raw.is_closed == 0) & (raw.never_sold == 0)]
    avg = recent.groupby(KEYS)["sales"].mean().rename("recent_avg_14d").reset_index()
    out = pairs.merge(avg, on=KEYS, how="left").fillna({"recent_avg_14d": 0})
    rng = np.random.default_rng(SEED)
    out["stock_factor"] = rng.uniform(0.3, 1.5, size=len(out))
    out["stock_SIMULATED"] = np.round(out.recent_avg_14d * out.stock_factor * out.window_days)
    return out[KEYS + ["stock_SIMULATED"]]


def windows(fc):
    # One row per store x category: is_perishable, window_days (sorted, so the stock seed order is stable)
    pairs = fc[KEYS + ["is_perishable"]].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    pairs["window_days"] = pairs.is_perishable.map(WINDOW)
    return pairs


def build_plan(fc, stock, tiny_families):
    # fc: date, store_nbr, family, is_perishable, forecast, forecast_high. stock: store_nbr, family, stock_SIMULATED
    fc = fc.copy()
    fc["day_n"] = (fc.date - fc.date.min()).dt.days + 1                 # 1..14
    pairs = windows(fc)
    fc = fc.merge(pairs[KEYS + ["window_days"]], on=KEYS)
    fc["spread"] = (fc.forecast_high - fc.forecast).clip(lower=0)
    inwin = fc[fc.day_n <= fc.window_days].groupby(KEYS)[["forecast", "spread"]].sum()
    inwin.columns = ["forecast_window", "spread_window"]
    daily = fc.groupby(KEYS)["forecast"].mean().rename("avg_daily_forecast")
    plan = pairs.merge(inwin.reset_index(), on=KEYS).merge(daily.reset_index(), on=KEYS)

    plan["buffer"] = np.minimum(plan.spread_window, BUFFER_CAP * plan.forecast_window)
    tiny = plan.family.isin(tiny_families)
    plan.loc[tiny, "buffer"] = np.maximum(plan.loc[tiny, "buffer"], 1.0)

    plan = plan.merge(stock, on=KEYS, how="left")
    plan["order_qty"] = np.ceil((plan.forecast_window + plan.buffer - plan.stock_SIMULATED).clip(lower=0))
    # Rounded before the status, so the shown number and the status always agree; empty if no demand
    plan["days_of_cover"] = (plan.stock_SIMULATED / plan.avg_daily_forecast).where(plan.avg_daily_forecast > 0).round(1)
    urgent = (plan.days_of_cover < URGENT_DAYS) & (plan.avg_daily_forecast >= LOW_SELLER)
    plan["status"] = np.select([urgent, plan.order_qty > 0], ["Urgent", "Reorder"], "OK")
    cols = KEYS + ["is_perishable", "window_days", "forecast_window", "buffer",
                   "stock_SIMULATED", "order_qty", "days_of_cover", "status"]
    return plan[cols].round({"forecast_window": 1, "buffer": 1, "days_of_cover": 1})


def make_plan():
    fc = pd.read_parquet(REPORTS / "forecast_next14.parquet")
    cat = pd.read_csv(REPORTS / "category_errors.csv")
    tiny = sorted(cat.loc[cat.avg_units_per_day < LOW_SELLER, "family"])
    stock = simulated_stock(windows(fc), fc.date.min() - pd.Timedelta(days=1))
    return build_plan(fc, stock, tiny), tiny


if __name__ == "__main__":
    plan, tiny = make_plan()
    plan.to_parquet(REPORTS / "reorder_plan.parquet", index=False)
    plan.to_csv(REPORTS / "reorder_plan.csv", index=False)
    print("Buffer of at least 1 unit (< 1 unit/day per store):", tiny)
    capped = plan.buffer >= BUFFER_CAP * plan.forecast_window - 0.05
    print(f"Buffer capped at 50% of the window forecast: {int((capped & ~plan.family.isin(tiny)).sum())} rows")
    print(f"\nRows: {len(plan):,} |", " | ".join(f"{s}: {(plan.status == s).sum():,}" for s in ["Urgent", "Reorder", "OK"]))
    counts = pd.crosstab(plan.is_perishable.map({0: "long-life", 1: "perishable"}), plan.status)
    print(counts[[c for c in ["Urgent", "Reorder", "OK"] if c in counts]].to_string())
    print("\nTop 10 orders for store 44 (stock is SIMULATED):")
    print(plan[plan.store_nbr == 44].sort_values("order_qty", ascending=False).head(10).to_string(index=False))
    print("\nSaved:", REPORTS / "reorder_plan.parquet", "(+ .csv copy)")
