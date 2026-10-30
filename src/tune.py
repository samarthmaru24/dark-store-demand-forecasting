"""Accuracy experiments (6i): model settings, training start year, holiday features, promotion feature,
average of 3 seeds. Every decision follows rules written BEFORE the run (here and in CLAUDE.md 6i).
The locked test days (2017-08-02..15) are never used here.

Windows (14 days each)
  tuning:    4 new windows in 2016 (never used before), always trained from 2015-01-01:
             2016-02-01 (Carnaval), 2016-06-06, 2016-08-01 (moved independence holiday), 2016-10-31 (Difuntos)
  back-test: the 6 agreed windows in 2017 (model.BACKTEST_STARTS)
Steps (each starts from the model kept by the step before; start = v2)
  1 settings  step-by-step search on the tuning windows only: num_leaves -> Tweedie power -> min_child_samples
              -> colsample_bytree, trees checked at CHECKPOINTS. Lowest mean tuning WAPE wins, then it must
              pass RULE on the back-tests.
  2 start     train from 2014-01-01 or 2016-01-01 instead of 2015-01-01 (back-tests only). Must pass RULE;
              if both pass, the lower mean wins.
  3 holiday   + HOLIDAY_EXTRA. Must pass RULE and not make the tuning mean worse.
  4 promo     + PROMO_EXTRA. Same as 3.
  5 seeds     average of 3 seeds (geometric mean, so the "why" parts still add up). Same as 3.
RULE: back-test mean WAPE improves by >= 0.3 points AND at most 1 of the 6 windows gets worse.
-> reports/tune_settings.csv, reports/tune_steps.csv, reports/tune_decisions.json
Run from the project root: python3 src/tune.py   (about 1-2 hours)
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import HOLIDAY_EXTRA, PROMO_EXTRA, add_v3_features, wape
from model import (BACKTEST_STARTS, PARAMS, REPORTS, TEST_START, TRAIN_START, feature_table, load_holidays,
                   variant, window)

TUNE_STARTS = pd.to_datetime(["2016-02-01", "2016-06-06", "2016-08-01", "2016-10-31"])
CHECKPOINTS = [140, 300, 500, 800, 1200, 1600, 2000]
SEARCH = [("num_leaves", [63, 127, 255]), ("power", [1.1, 1.2, 1.3, 1.5]),
          ("min_child_samples", [20, 50, 200]), ("colsample_bytree", [0.5, 0.8])]
START_YEARS = ["2014-01-01", "2016-01-01"]
SEEDS = [42, 43, 44]
MIN_GAIN = 0.3
V2_SETTINGS = {"num_leaves": 127, "power": 1.2, "min_child_samples": 50, "colsample_bytree": 0.8, "n_trees": 140}


def fit(train, feats, s, seed=42):
    p = {**PARAMS, "num_leaves": s["num_leaves"], "min_child_samples": s["min_child_samples"],
         "colsample_bytree": s["colsample_bytree"], "random_state": seed}
    m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=s["power"], n_estimators=s["n_trees"], **p)
    return m.fit(train[feats], train.sales)


def score(df, starts, feats, s, train_start=TRAIN_START, seeds=(42,)):
    # WAPE per window. Several seeds: average on the log scale (geometric mean of the forecasts)
    out = {}
    for st in starts:
        train, part = window(df, st, train_start=train_start)
        raw = np.mean([fit(train, feats, s, seed).predict(part[feats], raw_score=True) for seed in seeds], axis=0)
        out[str(st.date())] = wape(part.sales, np.exp(raw))
    return pd.Series(out)


def curve(df, feats, s):
    # Tuning-window WAPE at every checkpoint from one fit per window (rows: window x trees)
    rows = []
    for st in TUNE_STARTS:
        train, part = window(df, st)
        m = fit(train, feats, {**s, "n_trees": max(CHECKPOINTS)})
        for k in CHECKPOINTS:
            rows.append({**{n: s[n] for n, _ in SEARCH}, "window_start": str(st.date()), "trees": k,
                         "WAPE": wape(part.sales, m.predict(part[feats], num_iteration=k))})
    return pd.DataFrame(rows)


def search(df, feats):
    # One setting at a time; each value keeps its own best number of trees
    best, done = dict(V2_SETTINGS), {}
    for name, values in SEARCH:
        means = {}
        for v in values:
            s = {**best, name: v}
            key = tuple(s[n] for n, _ in SEARCH)
            if key not in done:
                t0 = time.time()
                done[key] = curve(df, feats, s)
                print(f"  {key}  best {done[key].groupby('trees').WAPE.mean().min():.3f}  ({time.time() - t0:.0f}s)",
                      flush=True)
            means[v] = done[key].groupby("trees").WAPE.mean()
        v = min(means, key=lambda x: means[x].min())
        best.update({name: v, "n_trees": int(means[v].idxmin())})
        print(f"  -> {name} = {v}, trees = {best['n_trees']}, tuning mean {means[v].min():.3f}", flush=True)
    pd.concat(done.values()).to_csv(REPORTS / "tune_settings.csv", index=False)
    return best


def rule(new, old, tune_new=None, tune_old=None):
    gain, worse = old.mean() - new.mean(), int((new > old).sum())
    ok = gain >= MIN_GAIN and worse <= 1
    if tune_new is not None:
        ok = ok and tune_new.mean() <= tune_old.mean()
    return {"backtest_mean_old": round(old.mean(), 3), "backtest_mean_new": round(new.mean(), 3),
            "gain_points": round(gain, 3), "windows_worse": worse,
            **({"tuning_mean_old": round(tune_old.mean(), 3), "tuning_mean_new": round(tune_new.mean(), 3)}
               if tune_new is not None else {}), "kept": bool(ok)}


def main():
    assert all(s + pd.Timedelta(days=14) <= TEST_START for s in list(TUNE_STARTS) + list(BACKTEST_STARTS))
    df = add_v3_features(feature_table(), load_holidays())
    feats, _, _ = variant("v2")
    rows, decisions = [], {}

    def log(step, cand, bt, tu=None):
        for st, w in bt.items():
            rows.append({"step": step, "candidate": cand, "windows": "back-test", "window_start": st, "WAPE": w})
        for st, w in (tu.items() if tu is not None else []):
            rows.append({"step": step, "candidate": cand, "windows": "tuning", "window_start": st, "WAPE": w})
        pd.DataFrame(rows).to_csv(REPORTS / "tune_steps.csv", index=False)

    def decide(step, d, extra=None):
        decisions[step] = {**d, **(extra or {})}
        with open(REPORTS / "tune_decisions.json", "w") as f:
            json.dump(decisions, f, indent=2)
        print(f"STEP {step}: {decisions[step]}\n", flush=True)

    # Kept model so far: v2
    kept = {"settings": dict(V2_SETTINGS), "train_start": str(TRAIN_START.date()), "features": list(feats), "seeds": [42]}
    bt_kept = score(df, BACKTEST_STARTS, feats, kept["settings"])
    tu_kept = score(df, TUNE_STARTS, feats, kept["settings"])
    log("0 v2", "v2", bt_kept, tu_kept)
    print(f"v2: back-test mean {bt_kept.mean():.3f} | tuning mean {tu_kept.mean():.3f}\n", flush=True)

    print("STEP 1 settings (tuning windows only)", flush=True)
    s = search(df, feats)
    bt, tu = score(df, BACKTEST_STARTS, feats, s), score(df, TUNE_STARTS, feats, s)
    log("1 settings", str(s), bt, tu)
    d = rule(bt, bt_kept)
    decide("1 settings", d, {"candidate": s})
    if d["kept"]:
        kept["settings"], bt_kept, tu_kept = s, bt, tu

    print("STEP 2 training start year (back-tests only)", flush=True)
    best = None
    for start in START_YEARS:
        bt = score(df, BACKTEST_STARTS, feats, kept["settings"], train_start=pd.Timestamp(start))
        log("2 start", start, bt)
        d = rule(bt, bt_kept)
        print(f"  start {start}: {d}", flush=True)
        if d["kept"] and (best is None or bt.mean() < best[1].mean()):
            best = (start, bt, d)
    decide("2 start", best[2] if best else {"kept": False}, {"candidate": best[0] if best else START_YEARS})
    if best:
        kept["train_start"], bt_kept = best[0], best[1]
    ts = pd.Timestamp(kept["train_start"])

    for step, extra in [("3 holiday", HOLIDAY_EXTRA), ("4 promo", PROMO_EXTRA)]:
        print(f"STEP {step}", flush=True)
        f = kept["features"] + extra
        bt = score(df, BACKTEST_STARTS, f, kept["settings"], train_start=ts)
        tu = score(df, TUNE_STARTS, f, kept["settings"])
        log(step, "+" + ",".join(extra), bt, tu)
        d = rule(bt, bt_kept, tu, tu_kept)
        decide(step, d, {"candidate": extra})
        if d["kept"]:
            kept["features"], bt_kept, tu_kept = f, bt, tu

    print("STEP 5 seeds", flush=True)
    bt = score(df, BACKTEST_STARTS, kept["features"], kept["settings"], train_start=ts, seeds=SEEDS)
    tu = score(df, TUNE_STARTS, kept["features"], kept["settings"], seeds=SEEDS)
    log("5 seeds", str(SEEDS), bt, tu)
    d = rule(bt, bt_kept, tu, tu_kept)
    decide("5 seeds", d, {"candidate": SEEDS})
    if d["kept"]:
        kept["seeds"], bt_kept, tu_kept = SEEDS, bt, tu

    decide("final", {"kept_model": kept, "backtest_mean": round(bt_kept.mean(), 3),
                     "backtest_windows": {k: round(v, 3) for k, v in bt_kept.items()},
                     "tuning_mean": round(tu_kept.mean(), 3)})


if __name__ == "__main__":
    main()
