# Stage 6b: reorder plan per store x category, from reports/forecast_next14.csv.
# order = forecast over window + buffer - current stock, minimum 0 (rounded up to whole units).
# Window: perishables 3 days, long-life 14 days.
# Buffer = forecast over window x category test WAPE (reports/category_errors.csv), capped at 50%;
#          categories selling < 1 unit/day per store get a fixed buffer of 1 unit.
# Stock is SIMULATED (the dataset has no stock): recent 14-day average x U(0.3, 1.5) x window_days, seed 42.
# Status: "Urgent" if days_of_cover < 2, else "Reorder" if order_qty > 0, else "OK".
# Run from the project root (after src/forecast.py): python3 src/reorder.py
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN, REPORTS = ROOT / "clean", ROOT / "reports"
WINDOW = {1: 3, 0: 14}      # is_perishable -> window_days
BUFFER_CAP = 0.50
LOW_SELLER = 1.0            # units/day per store
SEED = 42


def simulated_stock(pairs, last):
    # Recent average = mean sales on open days in the last 14 days
    raw = pd.read_parquet(CLEAN / "store_sales_clean.parquet",
                          columns=["date", "store_nbr", "family", "sales", "is_closed", "never_sold"])
    recent = raw[(raw.date > last - pd.Timedelta(days=14)) & (raw.is_closed == 0) & (raw.never_sold == 0)]
    avg = recent.groupby(["store_nbr", "family"])["sales"].mean().rename("recent_avg_14d").reset_index()
    out = pairs.merge(avg, on=["store_nbr", "family"], how="left").fillna({"recent_avg_14d": 0})
    rng = np.random.default_rng(SEED)
    out["stock_factor"] = rng.uniform(0.3, 1.5, size=len(out))
    out["stock_SIMULATED"] = np.round(out.recent_avg_14d * out.stock_factor * out.window_days)
    return out


def make_plan():
    fc = pd.read_csv(REPORTS / "forecast_next14.csv", parse_dates=["date"])
    cat = pd.read_csv(REPORTS / "category_errors.csv")
    last = fc.date.min() - pd.Timedelta(days=1)
    fc["day_n"] = (fc.date - last).dt.days                       # 1..14

    pairs = fc[["store_nbr", "family", "is_perishable"]].drop_duplicates().sort_values(["store_nbr", "family"])
    pairs["window_days"] = pairs.is_perishable.map(WINDOW)
    fc = fc.merge(pairs[["store_nbr", "family", "window_days"]], on=["store_nbr", "family"])
    win = fc[fc.day_n <= fc.window_days].groupby(["store_nbr", "family"])["forecast"].sum().rename("forecast_window")
    daily = fc.groupby(["store_nbr", "family"])["forecast"].mean().rename("avg_daily_forecast")
    plan = pairs.merge(win.reset_index(), on=["store_nbr", "family"]).merge(daily.reset_index(), on=["store_nbr", "family"])

    plan = plan.merge(cat[["family", "WAPE_model", "avg_units_per_day"]], on="family", how="left")
    rate = (plan.WAPE_model / 100).clip(upper=BUFFER_CAP)
    plan["buffer"] = np.where(plan.avg_units_per_day < LOW_SELLER, 1.0, plan.forecast_window * rate)

    plan = simulated_stock(plan, last)
    plan["order_qty"] = np.ceil((plan.forecast_window + plan.buffer - plan.stock_SIMULATED).clip(lower=0))
    # No forecast demand -> days_of_cover left empty (never Urgent)
    # Rounded before the status, so the shown number and the status always agree
    plan["days_of_cover"] = (plan.stock_SIMULATED / plan.avg_daily_forecast).where(plan.avg_daily_forecast > 0).round(1)
    plan["status"] = np.select([plan.days_of_cover < 2, plan.order_qty > 0], ["Urgent", "Reorder"], "OK")

    cols = ["store_nbr", "family", "is_perishable", "window_days", "forecast_window", "buffer",
            "stock_SIMULATED", "order_qty", "days_of_cover", "status"]
    plan = plan[cols].round({"forecast_window": 1, "buffer": 1, "days_of_cover": 1})
    return plan, cat


if __name__ == "__main__":
    plan, cat = make_plan()
    plan.to_csv(REPORTS / "reorder_plan.csv", index=False)
    low = sorted(cat.loc[cat.avg_units_per_day < LOW_SELLER, "family"])
    print("Fixed 1-unit buffer (< 1 unit/day per store):", low)
    print("Buffer capped at 50%:", sorted(set(cat.loc[cat.WAPE_model > 50, "family"]) - set(low)))
    print(f"\nRows: {len(plan):,} |", " | ".join(f"{s}: {(plan.status == s).sum():,}" for s in ["Urgent", "Reorder", "OK"]))
    counts = pd.crosstab(plan.is_perishable.map({0: "long-life", 1: "perishable"}), plan.status)
    print(counts[[c for c in ["Urgent", "Reorder", "OK"] if c in counts]].to_string())
    print("\nTop 10 orders for store 44 (stock is SIMULATED):")
    print(plan[plan.store_nbr == 44].sort_values("order_qty", ascending=False).head(10).to_string(index=False))
    print("\nSaved:", REPORTS / "reorder_plan.csv")
