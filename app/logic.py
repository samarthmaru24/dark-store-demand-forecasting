# Calculations for the dashboard. No Streamlit here, so every function can be tested directly.
# Reads only the precomputed files in app/data/ (built by src/build_app_data.py). Never runs the model.
import datetime as dt
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).resolve().parent / "data"
TODAY_DATA = pd.Timestamp("2017-08-15")      # the data's "today" (Ecuador demo)
SEASON_MONTHS = {"Summer": [3, 4, 5], "Monsoon": [6, 7, 8, 9], "Winter": [11, 12, 1, 2]}   # Monsoon wins in June
INDIA_CITIES = {  # lat, lon
    "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21), "Bengaluru": (12.97, 77.59), "Chennai": (13.08, 80.27),
    "Kolkata": (22.57, 88.36), "Hyderabad": (17.39, 78.49), "Pune": (18.52, 73.86), "Ahmedabad": (23.02, 72.57),
}
# Weather alert thresholds (assumptions, not model output)
RAINY_MM = 5          # daily rain >= 5 mm -> "Rainy day"
COLD_MAX_C = 15       # daily max temperature <= 15 °C -> "Cold day"
HOT_MAX_C = 40        # daily max temperature >= 40 °C -> "Hot day"


def load_data():
    d = {
        "history": pd.read_parquet(DATA / "history_last60.parquet"),
        "forecast": pd.read_csv(DATA / "forecast_next14.csv", parse_dates=["date"]),
        "plan": pd.read_csv(DATA / "reorder_plan.csv"),
        "cat_err": pd.read_csv(DATA / "category_errors.csv"),
        "rules": pd.read_csv(DATA / "conditions_rules.csv"),
        "name_map": pd.read_csv(DATA / "festival_name_map.csv"),
        "stores": pd.read_csv(DATA / "stores.csv"),
    }
    with open(DATA / "eval_summary.json") as f:
        d["eval"] = json.load(f)
    d["history"]["date"] = pd.to_datetime(d["history"]["date"])
    return d


def store_label(row):
    return f"{row.store_nbr} — {row.city}, type {row.store_type}"


def families_for(d, shelf):
    ce = d["cat_err"]
    if shelf == "Perishable":
        ce = ce[ce.is_perishable == 1]
    elif shelf == "Long-life":
        ce = ce[ce.is_perishable == 0]
    return sorted(ce.family)


def kpis(d, store, fams):
    fc = d["forecast"]
    fc = fc[(fc.store_nbr == store) & fc.family.isin(fams)]
    h = d["history"]
    h = h[(h.store_nbr == store) & h.family.isin(fams) & (h.date > TODAY_DATA - pd.Timedelta(days=14))]
    plan = d["plan"][(d["plan"].store_nbr == store) & d["plan"].family.isin(fams)]
    nxt, last = fc.forecast.sum(), h.sales.sum()
    return {"forecast_14d": float(nxt), "last_14d": float(last),
            "change_pct": float((nxt / last - 1) * 100) if last > 0 else float("nan"),
            "urgent": int((plan.status == "Urgent").sum()), "reorder": int((plan.status == "Reorder").sum())}


def demand_series(d, store, fams, days=30):
    # Long table: date, series, kind (actual / forecast), units, low, high
    h = d["history"]
    h = h[(h.store_nbr == store) & h.family.isin(fams) & (h.date > TODAY_DATA - pd.Timedelta(days=days))].copy()
    h.loc[h.is_closed == 1, "sales"] = np.nan
    fc = d["forecast"][(d["forecast"].store_nbr == store) & d["forecast"].family.isin(fams)].copy()
    fc = fc.merge(d["cat_err"][["family", "WAPE_model"]], on="family", how="left")
    fc["err"] = fc.forecast * fc.WAPE_model.fillna(0) / 100        # forecast ± category test WAPE
    by_cat = len(fams) <= 4
    key = "family" if by_cat else None
    if by_cat:
        a = h.groupby(["family", "date"]).sales.sum(min_count=1).reset_index()
        f = fc.groupby(["family", "date"])[["forecast", "err"]].sum().reset_index()
    else:
        a = h.groupby("date").sales.sum(min_count=1).reset_index().assign(family="Total (selected categories)")
        f = fc.groupby("date")[["forecast", "err"]].sum().reset_index().assign(family="Total (selected categories)")
    a = a.rename(columns={"family": "series", "sales": "units"}).assign(kind="actual")
    f = f.rename(columns={"family": "series", "forecast": "units"}).assign(kind="forecast")
    f["low"], f["high"] = (f.units - f.err).clip(lower=0), f.units + f.err
    # Start each forecast line at the last actual point so the two lines join
    join = a[a.date == TODAY_DATA].assign(kind="forecast", low=lambda x: x.units, high=lambda x: x.units)
    out = pd.concat([a, join, f.drop(columns="err")], ignore_index=True)
    return out.sort_values(["series", "kind", "date"]), by_cat


def trending(d, store, fams, min_base=1.0):
    # Last 7 days vs the previous 28 days (daily averages, open days only). Base < min_base units/day is skipped.
    h = d["history"]
    h = h[(h.store_nbr == store) & h.family.isin(fams) & (h.is_closed == 0)]
    recent = h[h.date > TODAY_DATA - pd.Timedelta(days=7)].groupby("family").sales.mean()
    prev = h[(h.date <= TODAY_DATA - pd.Timedelta(days=7)) &
             (h.date > TODAY_DATA - pd.Timedelta(days=35))].groupby("family").sales.mean()
    t = pd.DataFrame({"last_7d_avg": recent, "prev_28d_avg": prev}).dropna()
    t = t[t.prev_28d_avg >= min_base]
    t["change_pct"] = (t.last_7d_avg / t.prev_28d_avg - 1) * 100
    t = t.sort_values("change_pct", ascending=False).reset_index()
    up = t[t.change_pct > 0].head(5)
    down = t[t.change_pct < 0].tail(5)
    return up, down


def heatmap_table(d, store, fams):
    fc = d["forecast"][(d["forecast"].store_nbr == store) & d["forecast"].family.isin(fams)]
    return fc.pivot_table(index="family", columns="date", values="forecast", aggfunc="sum")


def current_season(day):
    for s, months in SEASON_MONTHS.items():
        if day.month in months:
            return s
    return None


def india_festivals(d, planning_date, within_days=45):
    # Festivals with a stocking rule from the `holidays` library, mapped via festival_name_map.csv
    lib = holidays.country_holidays("IN", years=[planning_date.year, planning_date.year + 1],
                                    categories=("public", "optional"))
    lib2rule = dict(zip(d["name_map"].library_name, d["name_map"].rule_name))
    rules = d["rules"].set_index("name")
    rows = []
    for day, names in lib.items():
        for n in names.split(";"):
            n = n.strip()
            base = n.replace(" (estimated)", "")
            away = (day - planning_date).days
            if base in lib2rule and 0 <= away <= within_days:
                r = rules.loc[lib2rule[base]]
                start = day - dt.timedelta(days=int(r.lead_days))
                rows.append({"festival": lib2rule[base], "library_name": n, "date": day, "days_away": away,
                             "start_stocking_by": start, "stock_now": start <= planning_date,
                             "india_products": r.india_products, "related_categories": r.related_categories,
                             "extra_stock_pct": int(r.extra_stock_pct), "season": r.season})
    cols = ["festival", "library_name", "date", "days_away", "start_stocking_by", "stock_now",
            "india_products", "related_categories", "extra_stock_pct", "season"]
    return pd.DataFrame(rows, columns=cols).sort_values("date").reset_index(drop=True)


def active_rules(d, planning_date, season_filter="All", weather_rules=()):
    # Rules that apply on the planning date: festivals whose stocking date has arrived, the current season,
    # and the weather rules the live forecast triggered. The season filter keeps only that season.
    rules = d["rules"]
    fest = india_festivals(d, planning_date)
    names = list(fest[fest.stock_now].festival)
    season = current_season(planning_date)
    if season:
        names.append(season)
    names += list(weather_rules)
    act = rules[rules.name.isin(names)]
    if season_filter != "All":
        act = act[act.season == season_filter]
    return act


def apply_boosts(plan, rules):
    # boosted_order = order + forecast_window x extra_stock_pct. If several rules match a family,
    # the largest extra_stock_pct is used (no stacking) and every matching rule is listed.
    plan = plan.copy()
    pct = pd.Series(0.0, index=plan.index)
    why = pd.Series("", index=plan.index)
    for r in rules.itertuples():
        cats = [c.strip() for c in r.related_categories.split(";")]
        hit = plan.family.isin(cats)
        pct = pct.where(~hit | (pct >= r.extra_stock_pct), float(r.extra_stock_pct))
        why = why.where(~hit, why.where(why == "", why + ", ") + f"{r.name} (+{r.extra_stock_pct}%)")
    plan["boost_pct"] = pct
    plan["boosted_order"] = np.ceil(plan.order_qty + plan.forecast_window * pct / 100)
    plan["boost_rule"] = why
    return plan


def weather_forecast(city):
    # Next 7 days of rain and temperature from the free Open-Meteo forecast API.
    # Returns (DataFrame, None) or (None, message).
    lat, lon = INDIA_CITIES[city]
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=8, params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "forecast_days": 7, "timezone": "Asia/Kolkata"})
        r.raise_for_status()
        day = r.json()["daily"]
        df = pd.DataFrame({"date": pd.to_datetime(day["time"]), "rain_mm": day["precipitation_sum"],
                           "temp_max": day["temperature_2m_max"], "temp_min": day["temperature_2m_min"]})
        df["rain_mm"] = df.rain_mm.fillna(0)
        return df, None
    except Exception:
        return None, "Weather forecast is not available right now (no internet or the free API is busy). Try again later."


def weather_alerts(wdf):
    # {rule name: list of dates} for every weather rule the 7-day forecast triggers
    checks = {"Rainy day": wdf.rain_mm >= RAINY_MM,
              "Cold day": wdf.temp_max <= COLD_MAX_C,
              "Hot day": wdf.temp_max >= HOT_MAX_C}
    return {name: list(wdf.date[hit]) for name, hit in checks.items() if hit.any()}


def shelf_life_split(d, store, fams):
    fc = d["forecast"][(d["forecast"].store_nbr == store) & d["forecast"].family.isin(fams)]
    s = fc.groupby("is_perishable").forecast.sum()
    total = s.sum()
    return pd.DataFrame({
        "group": ["Perishable", "Long-life"],
        "forecast_units": [float(s.get(1, 0)), float(s.get(0, 0))],
        "share_pct": [float(s.get(1, 0) / total * 100) if total else 0, float(s.get(0, 0) / total * 100) if total else 0],
        "test_WAPE": [d["eval"]["WAPE_perishable"], d["eval"]["WAPE_long_life"]],
    })
