"""Rolling back-tests: 6 windows of 14 days, all before the locked test period (2017-08-02).

For each window: train from 2015-01-01 up to the window start, predict the next 14 days, and score the
model and the 14-day-average baseline (WAPE). Model changes are accepted or rejected ONLY on these windows.

  python3 src/backtest.py               chosen model (v2) + baseline + 80% prediction ranges + Christmas check
                                        -> reports/backtest.csv, backtest_ranges.csv, christmas_backtest.json,
                                           q12_backtest.png
  python3 src/backtest.py --experiment  v1 vs v1+A4 features vs v1+A4+Tweedie -> reports/backtest_experiment.csv
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import wape
from model import (BACKTEST_STARTS, CHOSEN, QUANTILES, REPORTS, V2_EXTRA, XMAS_START, baseline, feature_table,
                   final_info, fit_point, fit_quantile, metrics, predict_point, predict_quantile, variant, window)


def experiment(df, feats_v1, n_trees):
    variants = {"v1": (feats_v1, "log1p"),
                "v1+A4": (feats_v1 + V2_EXTRA, "log1p"),
                "v1+A4+Tweedie": (feats_v1 + V2_EXTRA, "tweedie")}
    rows = []
    for s in BACKTEST_STARTS:
        train, part = window(df, s)
        rows.append({"window_start": s.date(), "variant": "baseline", **metrics(part, baseline(df, part, s))})
        for name, (feats, obj) in variants.items():
            m = fit_point(train, feats, n_trees, obj)
            rows.append({"window_start": s.date(), "variant": name, **metrics(part, predict_point(m, part, feats, obj))})
            print(f"  {s.date()} {name:14s} WAPE {rows[-1]['WAPE']:.2f}%")
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "backtest_experiment.csv", index=False)
    summary = res.groupby("variant", sort=False)[["WAPE", "WAPE_perishable", "WAPE_long_life", "bias_pct"]].mean()
    print("\nMean over the 6 back-test windows (simple average of window WAPEs):")
    print(summary.round(2).to_string())
    return summary


def chosen(df):
    feats, obj, n_trees = variant(CHOSEN)
    rows, ranges = [], []
    for s in BACKTEST_STARTS:
        train, part = window(df, s)
        part["pred"] = predict_point(fit_point(train, feats, n_trees, obj), part, feats, obj)
        part["baseline"] = baseline(df, part, s)
        rows.append({"window_start": s.date(), **{f"model_{k}": v for k, v in metrics(part, part.pred).items()},
                     "baseline_WAPE": wape(part.sales, part.baseline)})
        for col, a in QUANTILES.items():
            part[col] = predict_quantile(fit_quantile(train, feats, n_trees, a), part, feats)
        lo, hi = np.minimum(part.forecast_low, part.pred), np.maximum(part.forecast_high, part.pred)
        inside = (part.sales >= lo) & (part.sales <= hi)
        per = part.is_perishable == 1
        ranges.append({"window_start": s.date(), "coverage_pct": inside.mean() * 100,
                       "coverage_perishable_pct": inside[per].mean() * 100,
                       "coverage_long_life_pct": inside[~per].mean() * 100,
                       "clipped_rows_pct": ((part.forecast_low > part.pred) | (part.forecast_high < part.pred)).mean() * 100})
        print(f"  {s.date()}  model {rows[-1]['model_WAPE']:.2f}%  baseline {rows[-1]['baseline_WAPE']:.2f}%  "
              f"range coverage {ranges[-1]['coverage_pct']:.1f}%")
    bt, rg = pd.DataFrame(rows), pd.DataFrame(ranges)
    bt.to_csv(REPORTS / "backtest.csv", index=False)
    rg.to_csv(REPORTS / "backtest_ranges.csv", index=False)

    w = bt.model_WAPE
    print(f"\nModel WAPE over 6 windows: mean {w.mean():.2f}% | min {w.min():.2f}% | max {w.max():.2f}%"
          f"   (baseline mean {bt.baseline_WAPE.mean():.2f}%)")
    print(f"80% range coverage: mean {rg.coverage_pct.mean():.1f}% | min {rg.coverage_pct.min():.1f}% | "
          f"max {rg.coverage_pct.max():.1f}%  (perishable {rg.coverage_perishable_pct.mean():.1f}%, "
          f"long-life {rg.coverage_long_life_pct.mean():.1f}%)")
    plot(bt)

    christmas(df)
    return bt, rg


def christmas(df):
    # Christmas back-test (Extra 1), same window as Stage 5, for v1 and v2 -> reports/christmas_backtest.json
    train, xmas = window(df, XMAS_START)
    xmas["baseline"] = baseline(df, xmas, XMAS_START)
    out = {"window": f"{xmas.date.min().date()} to {xmas.date.max().date()}",
           "WAPE_baseline": wape(xmas.sales, xmas.baseline),
           "units_actual": float(xmas.sales.sum()), "units_baseline": float(xmas.baseline.sum())}
    for name in ["v1", "v2"]:
        feats, obj, n_trees = variant(name)
        xmas[name] = predict_point(fit_point(train, feats, n_trees, obj), xmas, feats, obj)
        out[name] = {"WAPE": wape(xmas.sales, xmas[name]), "units": float(xmas[name].sum())}
        print(f"Christmas back-test {name}: WAPE {out[name]['WAPE']:.1f}% (baseline {out['WAPE_baseline']:.1f}%) | "
              f"units {out[name]['units']:,.0f} vs actual {out['units_actual']:,.0f} "
              f"({(out[name]['units'] / out['units_actual'] - 1) * 100:+.1f}%)")
    out["categories"] = {f: {"actual": float(g.sales.sum()), "v1": float(g.v1.sum()), "v2": float(g.v2.sum()),
                             "baseline": float(g.baseline.sum())}
                         for f, g in xmas.groupby(xmas.family.astype(str))
                         if f in ("FROZEN FOODS", "LIQUOR,WINE,BEER", "GROCERY I", "BEVERAGES")}
    with open(REPORTS / "christmas_backtest.json", "w") as f:
        json.dump(out, f, indent=2)


def plot(bt):
    x = np.arange(len(bt))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, bt.model_WAPE, 0.4, color="#2a78d6", label=f"LightGBM ({CHOSEN})")
    ax.bar(x + 0.2, bt.baseline_WAPE, 0.4, color="#c3c2b7", label="Baseline (14-day average)")
    for i, (m, b) in enumerate(zip(bt.model_WAPE, bt.baseline_WAPE)):
        ax.text(i - 0.2, m + 0.3, f"{m:.1f}", ha="center", fontsize=9, color="#52514e")
        ax.text(i + 0.2, b + 0.3, f"{b:.1f}", ha="center", fontsize=9, color="#52514e")
    ax.set_xticks(x, [str(s) for s in bt.window_start])
    ax.set_ylabel("WAPE % (lower = better)")
    ax.set_title("Rolling back-tests: 6 windows of 14 days before the test period")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "q12_backtest.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", action="store_true")
    args = ap.parse_args()
    df = feature_table()
    if args.experiment:
        info = final_info()
        experiment(df, info["features"], info["n_trees"])
    else:
        chosen(df)
