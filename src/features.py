# Feature building shared by Stage 5 (evaluation) and Stage 6 (dashboard).
# Same logic as notebooks/03_model.ipynb. Every past-sales clue is at least H = 14 days old.
import numpy as np
import pandas as pd

H = 14
WEATHER_COLS = ["rain_mm", "temp_max", "rainy_day"]
FEATURES = [
    "store_id", "family", "city", "store_type", "cluster", "is_perishable",
    "dayofweek", "day", "month", "weekofyear", "dayofyear", "is_payday",
    "is_holiday", "is_national_holiday", "is_regional_holiday", "is_local_holiday", "is_event", "is_quake",
    "days_to_holiday", "onpromotion",
    "lag_14", "lag_21", "lag_28", "lag_35", "lag_42", "mean_7", "mean_28", "mean_56", "std_28",
    "trans_lag14", "oil_lag14", "rain_mm", "temp_max", "rainy_day",
]


def build_features(df, weather):
    # df: clean table with never_sold pairs removed. weather: date, city, rain_mm, temp_max
    df = df.copy()
    df["family"] = df["family"].astype(str)
    df = df.merge(weather, on=["date", "city"], how="left")
    df["rain_mm"] = df["rain_mm"].fillna(0)
    df["temp_max"] = df.groupby("city")["temp_max"].transform(lambda s: s.ffill().bfill())
    df["rainy_day"] = (df.rain_mm >= 5).astype(int)

    df["dayofweek"] = df.date.dt.dayofweek
    df["day"] = df.date.dt.day
    df["month"] = df.date.dt.month
    df["weekofyear"] = df.date.dt.isocalendar().week.astype(int)
    df["dayofyear"] = df.date.dt.dayofyear

    hol_dates = pd.DatetimeIndex(sorted(df.loc[df.is_national_holiday == 1, "date"].unique()))
    cal = pd.DataFrame({"date": pd.date_range(df.date.min(), df.date.max())})
    pos = hol_dates.searchsorted(cal.date)
    nxt = pd.Series(list(hol_dates) + [pd.NaT])[pos].to_numpy()
    cal["days_to_holiday"] = (nxt - cal.date.to_numpy()) / np.timedelta64(1, "D")
    cal["days_to_holiday"] = cal["days_to_holiday"].fillna(60).clip(upper=60)
    oil = df.groupby("date")["dcoilwtico"].first().reindex(cal.date).ffill()
    cal["oil_lag14"] = oil.shift(H).to_numpy()
    df = df.merge(cal, on="date", how="left")

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

    tr = df.groupby(["store_nbr", "date"])["transactions"].first().groupby(level=0).shift(H).rename("trans_lag14")
    df = df.merge(tr.reset_index(), on=["store_nbr", "date"], how="left")

    for c in ["family", "city", "store_type"]:
        df[c] = df[c].astype("category")
    df["store_id"] = df.store_nbr.astype("category")
    return df


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def wape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.sum(np.abs(y - p)) / np.sum(y) * 100) if np.sum(y) > 0 else float("nan")
