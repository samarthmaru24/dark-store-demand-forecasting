"""Model settings and training helpers shared by back-tests, retraining, forecasting and tests.

Base settings are the ones agreed in Stages 4-5 (notebooks/04_evaluation.ipynb): LightGBM, 140 trees,
no weather, every past-sales clue at least 14 days old (src/features.py).
  v1 = 31 features, target log1p(sales)                         (Stage 5 model, models/lgbm_final.txt)
  v2 = v1 + days_since_holiday + sales_lag_364, Tweedie on raw sales (kept after the back-test experiment)
  (6i accuracy experiments, src/tune.py: settings, start year, holiday + promotion features, 3 seeds - none
   passed the rule, so v2 stays. load_holidays() and window(train_start=...) are used there.)
Prediction ranges: quantile models (10% and 90%) with the chosen feature set.
  "level"    = quantile of log1p(sales)                                   (6d, A7)
  "relative" = quantile of log1p(sales) - log1p(mean_28), i.e. relative to the category's recent level
               (mean_28 is 14+ days old). Tested with python3 src/backtest.py --ranges-experiment.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import V2_EXTRA, build_features_v2, wape

CLEAN, MODELS, REPORTS = ROOT / "clean", ROOT / "models", ROOT / "reports"
TRAIN_START = pd.Timestamp("2015-01-01")
VALID_START = pd.Timestamp("2017-07-19")
TEST_START = pd.Timestamp("2017-08-02")            # locked test period: 2017-08-02 .. 2017-08-15
BACKTEST_STARTS = pd.to_datetime(["2017-01-16", "2017-02-27", "2017-04-10",
                                  "2017-05-22", "2017-06-19", "2017-07-19"])
XMAS_START = pd.Timestamp("2016-12-12")              # Christmas back-test (Stage 5)
QUANTILES = {"forecast_low": 0.1, "forecast_high": 0.9}
PARAMS = dict(learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, verbose=-1)
CHOSEN = "v2"                                        # decided on back-tests only (see CLAUDE.md)
RANGE_TARGET = "relative"                           # "level" or "relative" (see docstring)
RANGE_REF = "mean_28"


def final_info():
    # Stage 5 (v1) model info: features and number of trees
    with open(MODELS / "lgbm_final_info.json") as f:
        return json.load(f)


def variant(name):
    # (features, objective, n_trees) for a model variant
    info = final_info()
    if name == "v1":
        return info["features"], "log1p", info["n_trees"]
    if name == "v2":
        return info["features"] + V2_EXTRA, "tweedie", info["n_trees"]
    raise ValueError(name)


def load_clean():
    raw = pd.read_parquet(CLEAN / "store_sales_clean.parquet")
    return raw[raw.never_sold == 0].copy()


def load_weather():
    return pd.read_csv(ROOT / "raw" / "weather_ecuador" / "weather.csv", parse_dates=["date"])


def load_holidays():
    return pd.read_csv(ROOT / "raw" / "store_sales" / "holidays_events.csv", parse_dates=["date"])


def feature_table(raw=None):
    # v2 table: has every v1 column plus the two v2 columns, so both variants can use it
    raw = load_clean() if raw is None else raw
    return build_features_v2(raw, load_weather())


def fit_point(train, feats, n_trees, objective="log1p"):
    if objective == "tweedie":
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, n_estimators=n_trees, **PARAMS)
        return m.fit(train[feats], train.sales)
    m = lgb.LGBMRegressor(objective="regression", n_estimators=n_trees, **PARAMS)
    return m.fit(train[feats], np.log1p(train.sales))


def predict_point(model, part, feats, objective="log1p"):
    # Works for LGBMRegressor and a Booster loaded from file (Tweedie predictions are already in units)
    p = model.predict(part[feats])
    return (p if objective == "tweedie" else np.expm1(p)).clip(0)


def range_ref(part, target):
    # log1p of the reference level the range model works relative to (0 for "level")
    if target == "relative":
        return np.log1p(part[RANGE_REF].fillna(0).to_numpy())
    return 0.0


def fit_quantile(train, feats, n_trees, alpha, target=None):
    # Quantiles survive the log1p transform (and a shift by a known value), so they map back with expm1
    target = target or RANGE_TARGET
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=n_trees, **PARAMS)
    return m.fit(train[feats], np.log1p(train.sales) - range_ref(train, target))


def predict_quantile(model, part, feats, target=None):
    target = target or RANGE_TARGET
    return np.expm1(model.predict(part[feats]) + range_ref(part, target)).clip(0)


def baseline(df, part, start):
    # Average units/day of each store x category over the 14 open days before `start`
    hist = df[(df.date >= start - pd.Timedelta(days=14)) & (df.date < start) & (df.is_closed == 0)]
    b = hist.groupby(["store_nbr", "family"], observed=True)["sales"].mean().rename("baseline").reset_index()
    return part.merge(b, on=["store_nbr", "family"], how="left")["baseline"].fillna(0).to_numpy()


def metrics(part, pred):
    y, p = part.sales.to_numpy(), np.asarray(pred)
    per = part.is_perishable.to_numpy() == 1
    return {"WAPE": wape(y, p), "WAPE_perishable": wape(y[per], p[per]),
            "WAPE_long_life": wape(y[~per], p[~per]), "bias_pct": float((p.sum() / y.sum() - 1) * 100)}


def volume_wape(part, pred_col):
    # Low / medium / top sellers, same grouping as notebooks/04_evaluation.ipynb (row order matters for ties)
    avg = part.groupby(["store_nbr", "family"], observed=True)["sales"].transform("mean")
    vol = pd.qcut(avg.rank(method="first"), 3, labels=["low sellers", "medium sellers", "top sellers"])
    return {str(k): wape(g.sales, g[pred_col]) for k, g in part.groupby(vol, observed=True)}


def window(df, start, days=14, train_start=TRAIN_START):
    # (train, part) for a window starting at `start`; open days only, train from train_start
    open_ = df.is_closed == 0
    train = df[(df.date >= train_start) & (df.date < start) & open_]
    part = df[(df.date >= start) & (df.date < start + pd.Timedelta(days=days)) & open_].copy()
    return train, part
