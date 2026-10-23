import json
cells=[]
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.strip()})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip()})

md("""# 03 · Build the model
**Goal:** a model that clearly beats a simple guess, predicting units sold per store × category for the next **14 days**.

Steps: weather → features → time split → baseline → LightGBM → save.

⚠️ **The one rule:** every past-sales clue is at least 14 days old. The test period (last 14 days) is **not touched** here — that is Stage 5.""")

code("""import time, json
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN = Path("../clean")
WEATHER = Path("../raw/weather_ecuador")
MODELS = Path("../models")
REPORTS = Path("../reports")
for p in (WEATHER, MODELS, REPORTS):
    p.mkdir(exist_ok=True, parents=True)

H = 14  # forecast horizon in days

df = pd.read_parquet(CLEAN / "store_sales_clean.parquet")
df = df[df.never_sold == 0].copy()          # pairs that never sell -> always predict 0
df["family"] = df["family"].astype(str)
print(df.shape, df.date.min().date(), "to", df.date.max().date())""")

md("""## 1 · Weather (rain + temperature) for each store's city
Free [Open-Meteo](https://open-meteo.com/) API, no key needed. Downloaded once and saved to `raw/weather_ecuador/weather.csv`; later runs just load the file.
The city's coordinates are found by name and checked against the store's state.""")
code("""WFILE = WEATHER / "weather.csv"
start, end = df.date.min().date().isoformat(), df.date.max().date().isoformat()

NEEDED = {"date", "city", "rain_mm", "temp_max"}
weather = pd.read_csv(WFILE, parse_dates=["date"]) if WFILE.exists() else None
if weather is not None and NEEDED.issubset(weather.columns):
    print("Loaded saved weather:", weather.shape)
else:
    if weather is not None:
        print("Saved weather file has a different format -> downloading again")
    places = df[["city", "state"]].drop_duplicates().sort_values("city")
    frames = []
    for city, state in places.itertuples(index=False):
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": city, "count": 10, "countryCode": "EC"}, timeout=30).json()
        results = geo.get("results", [])
        match = next((r for r in results if r.get("admin1", "").lower() == state.lower()), results[0] if results else None)
        if match is None:
            print(f"!! {city}: not found, skipped")
            continue
        print(f"{city:15} ({state:30}) -> {match['name']}, {match.get('admin1')}  {match['latitude']:.2f}, {match['longitude']:.2f}")
        w = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": match["latitude"], "longitude": match["longitude"],
            "start_date": start, "end_date": end,
            "daily": "precipitation_sum,temperature_2m_max", "timezone": "auto"}, timeout=60).json()["daily"]
        frames.append(pd.DataFrame({"date": pd.to_datetime(w["time"]), "city": city,
                                    "rain_mm": w["precipitation_sum"], "temp_max": w["temperature_2m_max"]}))
        time.sleep(1)
    weather = pd.concat(frames, ignore_index=True)
    weather.to_csv(WFILE, index=False)
    print("Saved:", WFILE, weather.shape)

print("Missing weather values:", weather[["rain_mm", "temp_max"]].isna().sum().to_dict())
weather.head()""")

md("""**Check the printed list:** each city should match its own state. If one looks wrong, tell your assistant before going on.

Note: in real use, weather comes from a forecast (up to 16 days ahead), so real-world accuracy will be slightly lower than here.""")

md("## 2 · Features (the clues)")
code("""df = df.merge(weather, on=["date", "city"], how="left")
df["rain_mm"] = df["rain_mm"].fillna(0)
df["temp_max"] = df.groupby("city")["temp_max"].transform(lambda s: s.ffill().bfill())
df["rainy_day"] = (df.rain_mm >= 5).astype(int)

# Calendar
df["dayofweek"] = df.date.dt.dayofweek
df["day"] = df.date.dt.day
df["month"] = df.date.dt.month
df["weekofyear"] = df.date.dt.isocalendar().week.astype(int)
df["dayofyear"] = df.date.dt.dayofyear

# Days until the next national holiday (the festival build-up clue)
hol_dates = pd.DatetimeIndex(sorted(df.loc[df.is_national_holiday == 1, "date"].unique()))
cal = pd.DataFrame({"date": pd.date_range(df.date.min(), df.date.max())})
pos = hol_dates.searchsorted(cal.date)
nxt = pd.Series(list(hol_dates) + [pd.NaT])[pos].to_numpy()
cal["days_to_holiday"] = ((nxt - cal.date.to_numpy()) / np.timedelta64(1, "D"))
cal["days_to_holiday"] = cal["days_to_holiday"].fillna(60).clip(upper=60)

# Oil price: only the value from 14+ days ago is known when forecasting
oil = df.groupby("date")["dcoilwtico"].first().reindex(cal.date).ffill()
cal["oil_lag14"] = oil.shift(H).to_numpy()
df = df.merge(cal, on="date", how="left")

df.filter(["date", "dayofweek", "days_to_holiday", "oil_lag14", "rain_mm", "temp_max"]).head()""")

code("""# Past sales clues — ALL at least 14 days old. Closed days are blanked so they don't look like zero demand.
df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
df["sales_open"] = df.sales.where(df.is_closed == 0)
keys = [df.store_nbr, df.family]
g = df.groupby(keys)["sales_open"]

for lag in [14, 21, 28, 35, 42]:
    df[f"lag_{lag}"] = g.shift(lag)

old = g.shift(H)
for w in [7, 28, 56]:
    df[f"mean_{w}"] = old.groupby(keys).transform(lambda s: s.rolling(w, min_periods=1).mean())
df["std_28"] = old.groupby(keys).transform(lambda s: s.rolling(28, min_periods=7).std())

# Store traffic, also 14+ days old
tr = df.groupby(["store_nbr", "date"])["transactions"].first().groupby(level=0).shift(H).rename("trans_lag14")
df = df.merge(tr.reset_index(), on=["store_nbr", "date"], how="left")

# Categories LightGBM should treat as labels, not numbers
for c in ["family", "city", "store_type"]:
    df[c] = df[c].astype("category")
df["store_id"] = df.store_nbr.astype("category")

FEATURES = [
    "store_id", "family", "city", "store_type", "cluster", "is_perishable",
    "dayofweek", "day", "month", "weekofyear", "dayofyear", "is_payday",
    "is_holiday", "is_national_holiday", "is_regional_holiday", "is_local_holiday", "is_event", "is_quake",
    "days_to_holiday", "onpromotion",
    "lag_14", "lag_21", "lag_28", "lag_35", "lag_42", "mean_7", "mean_28", "mean_56", "std_28",
    "trans_lag14", "oil_lag14", "rain_mm", "temp_max", "rainy_day",
]
print(len(FEATURES), "features")""")

md("""## 3 · Split by time (never randomly)
- **Test** = last 14 days → kept locked for Stage 5
- **Validation** = the 14 days before that
- **Training** = from 2015 up to validation (older years are used only to build the clues)""")
code("""END = df.date.max()
TEST_START = END - pd.Timedelta(days=H - 1)
VALID_START = TEST_START - pd.Timedelta(days=H)
TRAIN_START = pd.Timestamp("2015-01-01")

open_ = df.is_closed == 0
train = df[(df.date >= TRAIN_START) & (df.date < VALID_START) & open_]
valid = df[(df.date >= VALID_START) & (df.date < TEST_START) & open_].copy()

print("Train:", train.date.min().date(), "to", train.date.max().date(), f"({len(train):,} rows)")
print("Valid:", valid.date.min().date(), "to", valid.date.max().date(), f"({len(valid):,} rows)")
print("Test (locked):", TEST_START.date(), "to", END.date())
assert valid.date.max() < TEST_START""")

md("## 4 · Baseline: average of the last 14 days")
code("""def mae(y, p):  return float(np.mean(np.abs(y - p)))
def wape(y, p): return float(np.sum(np.abs(y - p)) / np.sum(y) * 100)

hist = df[(df.date >= VALID_START - pd.Timedelta(days=H)) & (df.date < VALID_START) & open_]
base = hist.groupby(["store_nbr", "family"], observed=True)["sales"].mean().rename("baseline").reset_index()
base["family"] = base["family"].astype(str)
valid["family_str"] = valid["family"].astype(str)
valid = valid.merge(base, left_on=["store_nbr", "family_str"], right_on=["store_nbr", "family"],
                    how="left", suffixes=("", "_b")).drop(columns=["family_b", "family_str"])
valid["baseline"] = valid["baseline"].fillna(0)

results = {"Baseline (14-day average)": (mae(valid.sales, valid.baseline), wape(valid.sales, valid.baseline))}
print(f"Baseline  MAE = {results['Baseline (14-day average)'][0]:.2f} units   WAPE = {results['Baseline (14-day average)'][1]:.2f}%")""")

md("""## 5 · Train LightGBM
Trained on log(1 + sales) so huge and tiny categories work together. Early stopping picks the best number of trees on the validation days.""")
code("""def train_lgbm(features):
    model = lgb.LGBMRegressor(
        objective="regression", learning_rate=0.05, n_estimators=3000,
        num_leaves=127, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(
        train[features], np.log1p(train.sales),
        eval_set=[(valid[features], np.log1p(valid.sales))],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    pred = np.expm1(model.predict(valid[features])).clip(0)
    return model, pred

t0 = time.time()
model, valid["pred"] = train_lgbm(FEATURES)
print(f"Trained in {time.time() - t0:.0f}s, best number of trees: {model.best_iteration_}")

results["LightGBM (all features)"] = (mae(valid.sales, valid.pred), wape(valid.sales, valid.pred))""")

md("## 6 · Does it beat the baseline?")
code("""table = pd.DataFrame(results, index=["MAE (units)", "WAPE (%)"]).T.round(2)
print(table.to_string())

b, m = results["Baseline (14-day average)"][1], results["LightGBM (all features)"][1]
print(f"\\nLightGBM error is {(1 - m / b) * 100:.1f}% lower than the baseline." if m < b
      else "\\n⚠️ LightGBM does NOT beat the baseline — stop and check the features.")""")

md("## 7 · Which clues helped most?")
code("""imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).sort_values()
imp = imp / imp.sum() * 100
plt.figure(figsize=(10, 7))
plt.barh(imp.index[-15:], imp.values[-15:], color="#4C72B0")
plt.title("Top 15 clues by importance (% of total gain)")
plt.xlabel("% of total importance")
plt.tight_layout(); plt.savefig(REPORTS / "q8_feature_importance.png", dpi=150); plt.show(); plt.close("all")
print(imp.sort_values(ascending=False).round(1).head(15).to_string())""")

md("""## 8 · Does weather help? (for your report)
Same model, trained **without** the 3 weather clues. If the error goes up without them, weather helps.""")
code("""WEATHER_COLS = ["rain_mm", "temp_max", "rainy_day"]
_, pred_nw = train_lgbm([f for f in FEATURES if f not in WEATHER_COLS])
results["LightGBM (no weather)"] = (mae(valid.sales, pred_nw), wape(valid.sales, pred_nw))
print(pd.DataFrame(results, index=["MAE (units)", "WAPE (%)"]).T.round(2).to_string())""")

md("## 9 · Save the model")
code("""model.booster_.save_model(str(MODELS / "lgbm_v1.txt"))
json.dump({"features": FEATURES, "best_iteration": int(model.best_iteration_), "horizon_days": H,
           "train_start": str(TRAIN_START.date()), "valid_start": str(VALID_START.date()),
           "test_start": str(TEST_START.date()),
           "results_valid": {k: {"MAE": round(v[0], 3), "WAPE": round(v[1], 3)} for k, v in results.items()}},
          open(MODELS / "lgbm_v1_info.json", "w"), indent=2)
print("Saved:", sorted(p.name for p in MODELS.iterdir()))""")

md("""## ✅ Done when
- LightGBM's WAPE is clearly lower than the baseline's
- `models/lgbm_v1.txt` and `models/lgbm_v1_info.json` exist

**Write down for your report:** baseline vs LightGBM error, the top 5 clues, and whether weather helped.""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
for i,c in enumerate(cells): c["id"]=f"cell-{i:02d}"
json.dump(nb,open("notebooks/03_model.ipynb","w"),indent=1)
print("Written notebooks/03_model.ipynb")

