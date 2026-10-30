"""Why does the forecast move? Split every v2 forecast into the effect of each input (LightGBM pred_contrib).

The v2 model (Tweedie) predicts log(units) as a sum: base + one contribution per feature, so
exp(sum) = the forecast. Features are grouped into plain reasons (REASONS). For each store x category we
compare the average contribution over the next 14 days (2017-08-16..29) with the same model's view of the
last 14 days (2017-08-02..15, open days): effect_pct = exp(difference) - 1, i.e. how much that reason pushes
the next 14 days up or down. The last 14 days are inside the final model's training data, so this explains
the model, it is not an accuracy check. The reasons multiply, so the effects do not add up exactly.
-> reports/forecast_why.parquet (store_nbr, family, reason, effect_pct)
Run from the project root (after src/forecast.py): python3 src/explain.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import H
from forecast import make_forecast

MODELS, REPORTS = ROOT / "models", ROOT / "reports"
KEYS = ["store_nbr", "family"]
REASONS = {
    "Recent sales level": ["lag_14", "lag_21", "lag_28", "lag_35", "lag_42", "mean_7", "mean_28", "mean_56", "std_28"],
    "Same time last year": ["sales_lag_364"],
    "Calendar (weekday, payday)": ["dayofweek", "day", "month", "weekofyear", "dayofyear", "is_payday"],
    "Holidays & events": ["is_holiday", "is_national_holiday", "is_regional_holiday", "is_local_holiday",
                          "is_event", "is_quake", "days_to_holiday", "days_since_holiday"],
    "Promotions": ["onpromotion"],
    "Shoppers & oil price": ["trans_lag14", "oil_lag14"],
    "Store & category": ["store_id", "family", "city", "store_type", "cluster", "is_perishable"],
}


def contributions(booster, part, feats):
    # One column per reason (log scale) + "total" = log of the forecast
    c = booster.predict(part[feats], pred_contrib=True)
    out = part[KEYS].copy()
    for reason, cols in REASONS.items():
        out[reason] = c[:, [feats.index(f) for f in cols]].sum(axis=1)
    out["total"] = c.sum(axis=1)
    return out


def make_why():
    df, fut, info, last, _ = make_forecast()
    feats = info["features"]
    missing = set(feats) - {f for cols in REASONS.values() for f in cols}
    assert not missing, f"features without a reason: {missing}"
    booster = lgb.Booster(model_file=str(MODELS / "lgbm_final_v2.txt"))
    past = df[(df.date > last - pd.Timedelta(days=H)) & (df.date <= last) & (df.is_closed == 0)]
    cf, cp = contributions(booster, fut, feats), contributions(booster, past, feats)
    # Check: the contributions add up to the saved forecast
    gap = np.abs(np.exp(cf.total.to_numpy()) - fut.forecast.to_numpy()).max()
    diff = cf.groupby(KEYS, observed=True).mean() - cp.groupby(KEYS, observed=True).mean()
    why = (diff.drop(columns="total").stack().rename("log_diff").reset_index()
           .rename(columns={"level_2": "reason"}))
    why["effect_pct"] = (np.exp(why.log_diff) - 1) * 100
    why["family"] = why.family.astype(str)
    return why.drop(columns="log_diff").dropna(), gap


if __name__ == "__main__":
    why, gap = make_why()
    print(f"Check: exp(sum of contributions) vs saved forecast, largest gap {gap:.6f} units")
    why["effect_pct"] = why.effect_pct.round(2)
    why.to_parquet(REPORTS / "forecast_why.parquet", index=False)
    print(f"Rows: {len(why):,} ({why[KEYS].drop_duplicates().shape[0]:,} store x category pairs x {len(REASONS)} reasons)")
    print("\nMedian effect per reason, all pairs (%):")
    print(why.groupby("reason").effect_pct.median().round(1).sort_values().to_string())
    print("\nStore 44, GROCERY I:")
    print(why[(why.store_nbr == 44) & (why.family == "GROCERY I")].to_string(index=False))
    print("\nSaved:", REPORTS / "forecast_why.parquet")
