"""Dark Store Demand Planner — Streamlit dashboard (minimal layout: filters, 4 numbers, 4 tabs).

Reads only precomputed files in app/data/ (src/build_app_data.py). The model never runs here.
All calculations live in app/logic.py. Run from the project root: streamlit run app/app.py
"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import logic as L

# Colours (validated reference palette): categorical slots, up/down, ink, blue ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
UP, DOWN = "#2a78d6", "#d03b3b"
INK, INK2, MUTED, GRID = "#1a1a18", "#52514e", "#8a887f", "#ebe9e3"
BLUES = ["#e4eefb", "#b9d3f5", "#86b2ec", "#4f8de0", "#2a6fc4", "#1b5299", "#0d366b"]
STATUS_LABEL = {"Urgent": "🔴 Urgent", "Reorder": "🟡 Reorder", "OK": "🟢 OK"}

st.set_page_config(page_title="Dark Store Planner", page_icon="📦", layout="wide")
st.html("""<style>
.block-container {max-width: 1180px; padding-top: 4.5rem}
.eyebrow {font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .14em;
          text-transform: uppercase; color: #8a887f}
.sub {font-size: 15px; color: #6b6a64; margin: -6px 0 18px}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 6px 0 10px}
.card {background: #fff; border: 1px solid #e8e6df; border-radius: 14px; padding: 16px 18px}
.card .v {font-family: 'Fraunces', serif; font-size: 34px; line-height: 1.15; color: #1a1a18; margin: 6px 0 2px}
.card .v.red {color: #c2352f}
.card .s {font-size: 13px; color: #6b6a64}
.row {display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #f0eee8; font-size: 14px}
.row b {font-family: 'JetBrains Mono', monospace; font-weight: 500}
.up {color: #2a78d6} .down {color: #d03b3b}
.fest {font-family: 'Fraunces', serif; font-size: 21px; color: #1a1a18}
.pill {display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 2px 8px;
       border-radius: 99px; background: #fdecea; color: #b42318; margin-left: 6px; vertical-align: middle}
.note {font-size: 12.5px; color: #8a887f}
</style>""")


@st.cache_data
def data():
    return L.load_data()


@st.cache_data(ttl=3600, show_spinner=False)
def weather(city):
    return L.weather_forecast(city)


@st.cache_data
def festivals(planning_date):
    return L.india_festivals(data(), planning_date)


@st.cache_data
def per_store(fn, store, fams):
    # Cached per-store calculations (fams as a tuple so it can be a cache key)
    return getattr(L, fn)(data(), store, list(fams))


def nice(family):
    # "GROCERY II" -> "Grocery II", "HOME AND KITCHEN I" -> "Home and Kitchen I"
    words = [w if w in ("I", "II") else w.title() for w in family.split()]
    return " ".join(words).replace(" And ", " and ")


def style(fig, height=340, ytitle=""):
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=INK2, size=12),
        legend=dict(orientation="h", y=1.1, x=0, title=None), hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, tickfont=dict(color=MUTED),
                     title=dict(text=ytitle, font=dict(color=MUTED)), separatethousands=True)
    return fig


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {a})"


def card(label, value, sub, red=False):
    return (f"<div class='card'><div class='eyebrow'>{label}</div>"
            f"<div class='v{' red' if red else ''}'>{value}</div><div class='s'>{sub}</div></div>")


d = data()
res = d["results"]
bt, test = res["backtest_6_windows"], res["test_2017-08-02_to_2017-08-15"]

# ---------------- Header + filters ----------------
st.markdown("<div class='eyebrow'>Dark store planner · Favorita data · today = 15 Aug 2017</div>",
            unsafe_allow_html=True)
st.title("Demand & reorder")
st.markdown("<div class='sub'>Units each category will sell in the next 14 days, and what to order.</div>",
            unsafe_allow_html=True)

stores = d["stores"]
labels = {r.store_nbr: L.store_label(r) for r in stores.itertuples()}
f1, f2, f3 = st.columns([1.2, 1, 1.8])
store = f1.selectbox("Store", list(labels), index=list(labels).index(44), format_func=labels.get)
shelf = f2.segmented_control("Shelf life", ["All", "Perishable", "Long-life"], default="All") or "All"
options = L.families_for(d, shelf)
picked = f3.multiselect("Categories", options, format_func=nice, placeholder="All categories",
                        key=f"cats_{shelf}")
fams = picked or options

# ---------------- 4 numbers ----------------
k = per_store("kpis", store, tuple(fams))
st.markdown("<div class='kpis'>" + "".join([
    card("Next 14 days", f"{k['forecast_14d']:,.0f}", f"units · {k['change_pct']:+.1f}% vs last 14 days"),
    card("Urgent", k["urgent"], "under 2 days of stock left", red=k["urgent"] > 0),
    card("To reorder", k["reorder"], "categories that need an order"),
    card("Model error", f"{bt['chosen_v2']['mean']:.1f}%",
         f"vs {bt['chosen_v2']['baseline_mean']:.1f}% for a simple average"),
]) + "</div>", unsafe_allow_html=True)

t_fc, t_ro, t_in, t_ab = st.tabs(["Forecast", "Reorder", "India planning", "About"])

# ---------------- India planning (built first: the Reorder tab uses its settings) ----------------
with t_in:
    i1, i2, i3 = st.columns([1, 1.6, 1])
    planning_date = i1.date_input("Planning date", value=dt.date.today())
    season_filter = i2.segmented_control("Season", ["All", "Summer", "Monsoon", "Winter", "Festival"],
                                         default="All") or "All"
    city = i3.selectbox("City (weather)", list(L.INDIA_CITIES))
    st.markdown("<div class='note'>Rule-based suggestions for India, not model output.</div>",
                unsafe_allow_html=True)
    rules = d["rules"].set_index("name")

    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.subheader("Upcoming festivals")
        fest = festivals(planning_date)
        if season_filter not in ("All", "Festival"):
            st.caption(f"Hidden by the season filter ({season_filter}).")
        elif fest.empty:
            st.caption("No festival in the next 45 days.")
        else:
            for r in fest.itertuples():
                pill = "<span class='pill'>stock now</span>" if r.stock_now else ""
                st.markdown(
                    f"<div class='card' style='margin-bottom:10px'><span class='fest'>{r.festival}</span>{pill}"
                    f"<div class='eyebrow' style='margin:4px 0 6px'>{r.date:%a %d %b} · in {r.days_away} days · "
                    f"stock by {r.start_stocking_by:%d %b} · +{r.extra_stock_pct}%</div>"
                    f"<div class='s'>{r.india_products}</div></div>", unsafe_allow_html=True)
    with right:
        st.subheader("Season")
        season = L.current_season(planning_date)
        if season and season_filter in ("All", season):
            st.markdown(card(planning_date.strftime("%B"), season, rules.loc[season, "india_products"]),
                        unsafe_allow_html=True)
        elif season:
            st.caption(f"{season_filter} is out of season, so its products are hidden.")
        else:
            st.caption(f"No season rule for {planning_date:%B}.")

    st.subheader(f"Weather in {city} · next 7 days")
    wdf, err = weather(city)
    alerts = {}
    if err:
        st.caption(err)
    else:
        days = wdf.date.dt.strftime("%a %d")
        w1, w2 = st.columns(2, gap="large")
        fig = go.Figure(go.Bar(x=days, y=wdf.rain_mm, marker=dict(
            color=[SERIES[0] if v >= L.RAINY_MM else BLUES[1] for v in wdf.rain_mm], cornerradius=4),
            hovertemplate="%{x}<br>%{y:.1f} mm<extra></extra>"))
        style(fig, 220, "rain (mm)").update_layout(hovermode="closest")
        fig.add_hline(y=L.RAINY_MM, line=dict(color=MUTED, width=1, dash="dot"))
        w1.plotly_chart(fig, width="stretch")
        fig = go.Figure(go.Scatter(x=days, y=wdf.temp_max, mode="lines+markers", line=dict(color=SERIES[1], width=2),
                                   marker=dict(size=8), customdata=wdf.temp_min,
                                   hovertemplate="%{x}<br>max %{y:.1f} °C · min %{customdata:.1f} °C<extra></extra>"))
        style(fig, 220, "max temp (°C)").update_layout(hovermode="closest")
        w2.plotly_chart(fig, width="stretch")

        alerts = L.weather_alerts(wdf)
        icons = {"Rainy day": "🌧", "Cold day": "🥶", "Hot day": "🥵"}
        shown = {n: v for n, v in alerts.items() if season_filter in ("All", rules.loc[n, "season"])}
        if not alerts:
            st.caption("No rain, cold or heat alert this week.")
        for name, dates in shown.items():
            when = ", ".join(pd.to_datetime(dates).strftime("%a %d"))
            st.markdown(f"<div class='row'><span>{icons[name]} <b>{name}</b> · {when}</span>"
                        f"<span class='s'>stock up: {rules.loc[name, 'india_products']}</span></div>",
                        unsafe_allow_html=True)
        st.markdown(f"<div class='note' style='margin-top:6px'>Rainy ≥ {L.RAINY_MM} mm · cold ≤ {L.COLD_MAX_C} °C "
                    f"· hot ≥ {L.HOT_MAX_C} °C (max temperature).</div>", unsafe_allow_html=True)

# ---------------- Forecast ----------------
with t_fc:
    st.subheader("Last 30 days → next 14 days")
    series, by_cat = per_store("demand_series", store, tuple(fams))
    fig = go.Figure()
    for i, (name, g) in enumerate(series.groupby("series", sort=False)):
        col = SERIES[i % len(SERIES)] if by_cat else SERIES[0]
        label = nice(name) if by_cat else "All selected"
        a, f = g[g.kind == "actual"], g[g.kind == "forecast"]
        if by_cat:
            fig.add_trace(go.Scatter(x=f.date, y=f.high, mode="lines", line=dict(width=0), showlegend=False,
                                     hoverinfo="skip", legendgroup=label))
            fig.add_trace(go.Scatter(x=f.date, y=f.low, mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=rgba(col, 0.13), hoverinfo="skip", showlegend=False, legendgroup=label))
        fig.add_trace(go.Scatter(x=a.date, y=a.units, mode="lines", name=label, legendgroup=label,
                                 line=dict(color=col, width=2), hovertemplate="%{y:,.0f} units"))
        fig.add_trace(go.Scatter(x=f.date, y=f.units, mode="lines", name=label, legendgroup=label, showlegend=False,
                                 line=dict(color=col, width=2, dash="dash"), hovertemplate="%{y:,.0f} forecast"))
    fig.add_vline(x=L.TODAY_DATA, line=dict(color=MUTED, width=1, dash="dot"))
    st.plotly_chart(style(fig, 380, "units / day"), width="stretch")
    st.markdown("<div class='note'>Solid = actual · dashed = forecast"
                + (" · band = likely range (80%)" if by_cat else " · pick up to 4 categories to see each one")
                + "</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Trending this week")
        up, down = per_store("trending", store, tuple(fams))
        rows = [(r.family, r.change_pct, "up", "▲") for r in up.itertuples()] + \
               [(r.family, r.change_pct, "down", "▼") for r in down.sort_values("change_pct").itertuples()]
        if not rows:
            st.caption("Not enough sales history.")
        st.markdown("".join(f"<div class='row'><span>{nice(f)}</span><b class='{c}'>{a} {v:+.0f}%</b></div>"
                            for f, v, c, a in rows), unsafe_allow_html=True)
        st.markdown("<div class='note' style='margin-top:6px'>Last 7 days vs the 4 weeks before.</div>",
                    unsafe_allow_html=True)
    with c2:
        st.subheader("Perishable vs long-life")
        sl = per_store("shelf_life_split", store, tuple(fams))
        st.markdown("<div class='kpis'>" + "".join(
            card(r.group, f"{r.share_pct:.0f}%", f"{r.forecast_units:,.0f} units · {r.backtest_WAPE:.1f}% error")
            for r in sl.itertuples()) + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='note'>Share of next-14-day units. Fresh food sells every day in high volume, "
                    "so it is easier to predict.</div>", unsafe_allow_html=True)

    st.subheader("Next 14 days by category")
    hm = per_store("heatmap_table", store, tuple(fams))
    hm = hm.loc[hm.sum(axis=1).sort_values().index]
    # Log colour scale: a few big categories would otherwise make all others look the same
    ticks = [t for t in [1, 10, 100, 1_000, 10_000, 100_000] if t <= max(hm.values.max(), 1) * 10]
    fig = go.Figure(go.Heatmap(
        z=np.log10(hm.values + 1), customdata=hm.values,
        x=[c.strftime("%a %d") for c in hm.columns], y=[nice(f) for f in hm.index],
        colorscale=[[i / (len(BLUES) - 1), c] for i, c in enumerate(BLUES)], xgap=2, ygap=2,
        colorbar=dict(title=dict(text="units"), thickness=10, outlinewidth=0,
                      tickvals=[np.log10(t + 1) for t in ticks], ticktext=[f"{t:,}" for t in ticks]),
        hovertemplate="%{y}<br>%{x}<br>%{customdata:,.0f} units<extra></extra>"))
    style(fig, max(300, 22 * len(hm) + 60)).update_layout(hovermode="closest")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

# ---------------- Reorder ----------------
with t_ro:
    plan = d["plan"][(d["plan"].store_nbr == store) & d["plan"].family.isin(fams)].copy()
    r1, r2 = st.columns([3, 1])
    boost_on = r1.toggle("Add India festival / season / weather extra stock",
                         help="Uses the India planning tab settings. Rule-based assumption, not model output.")
    if boost_on:
        plan = L.apply_boosts(plan, L.active_rules(d, planning_date, season_filter, alerts))
    plan["rank"] = plan.status.map({"Urgent": 0, "Reorder": 1, "OK": 2})
    plan = plan.sort_values(["rank", "order_qty"], ascending=[True, False])
    whole = lambda x: x.round().astype(int)
    table = pd.DataFrame({
        "Category": plan.family.map(nice), "Status": plan.status.map(STATUS_LABEL),
        "Order": whole(plan.boosted_order if boost_on else plan.order_qty),
        "Stock*": whole(plan.stock_SIMULATED), "Days of stock": plan.days_of_cover,
        "Forecast": whole(plan.forecast_window), "Safety buffer": whole(plan.buffer), "Window (days)": plan.window_days})
    if boost_on:
        table["Extra because of"] = plan.boost_rule
    r2.download_button("Download CSV", table.to_csv(index=False).encode(), width="stretch",
                       file_name=f"reorder_plan_store_{store}.csv", mime="text/csv")
    units = st.column_config.NumberColumn(format="localized")
    st.dataframe(table, hide_index=True, width="stretch", height=min(35 * len(table) + 38, 600),
                 column_config={"Order": units, "Stock*": units, "Forecast": units, "Safety buffer": units,
                                "Days of stock": st.column_config.NumberColumn(format="%.1f")})
    st.markdown("<div class='note'>Order = forecast for the window + safety buffer − stock. "
                "Window: 3 days for fresh food, 14 for long-life. "
                "*Stock is simulated — the dataset has no stock data.</div>", unsafe_allow_html=True)

# ---------------- About ----------------
with t_ab:
    st.markdown(f"""
**How it works.** A LightGBM model learns from 2015–2017 sales. Every past-sales clue is at least 14 days old,
so all 14 days can be forecast at once without peeking at the future.

**How good is it?** (WAPE = % of units predicted wrong, lower is better)
- 6 back-tests in 2017: **{bt['chosen_v2']['mean']:.1f}%** vs {bt['chosen_v2']['baseline_mean']:.1f}% for a simple 14-day average
- Locked test days (2–15 Aug 2017): first model **{test['v1']['WAPE']:.1f}%** (unseen score) ·
  current model {test['v2']['WAPE']:.1f}% (scored after the test was seen) · simple average {test['baseline']['WAPE']:.1f}%

**Limits.** Ecuador supermarket data stands in for India · stock is simulated · category level, not single items ·
India festival, season and weather rules are assumptions.
""")

st.markdown("<div class='note' style='margin-top:40px'>Data: "
            "<a href='https://www.kaggle.com/competitions/store-sales-time-series-forecasting'>Favorita store sales "
            "(Kaggle)</a> · festival dates: Python holidays library · "
            "<a href='https://open-meteo.com/'>Weather data by Open-Meteo.com</a> (CC BY 4.0) · all numbers in units</div>",
            unsafe_allow_html=True)
