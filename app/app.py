# Dark Store Demand Planner — Streamlit dashboard (Stage 6c).
# Reads only precomputed files in app/data/ (src/build_app_data.py). The model never runs here.
# Run from the project root: streamlit run app/app.py
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import logic as L

# Colours (validated reference palette): categorical slots, diverging blue/red, status, ink, blue ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
UP, DOWN = "#2a78d6", "#e34948"
STATUS = {"Urgent": ("#d03b3b", "#ffffff"), "Reorder": ("#fab219", "#0b0b0b"), "OK": ("#0ca30c", "#ffffff")}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.set_page_config(page_title="Dark Store Demand Planner", page_icon="📦", layout="wide")


@st.cache_data
def data():
    return L.load_data()


@st.cache_data(ttl=3600, show_spinner=False)
def weather(city):
    return L.weather_forecast(city)


@st.cache_data
def festivals(planning_date):
    return L.india_festivals(data(), planning_date)


def style(fig, height=360, ytitle="units"):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=13),
        legend=dict(orientation="h", y=1.08, x=0, title=None), hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     title=dict(text=ytitle, font=dict(color=MUTED)), separatethousands=True)
    return fig


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {a})"


d = data()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Filters")
    stores = d["stores"]
    labels = {r.store_nbr: L.store_label(r) for r in stores.itertuples()}
    store = st.selectbox("Store", list(labels), index=list(labels).index(44), format_func=labels.get)
    shelf = st.radio("Shelf life", ["All", "Perishable", "Long-life"], horizontal=True)
    options = L.families_for(d, shelf)
    if st.checkbox("All categories", value=True):
        fams = options
    else:
        fams = st.multiselect("Categories", options, default=options[:3], key=f"cats_{shelf}")
    season_filter = st.selectbox("Season filter", ["All", "Summer", "Monsoon", "Winter", "Festival"],
                                 help="Applies to the India panel and to the India rule boosts.")
    st.divider()
    planning_date = st.date_input("India planning date", value=dt.date.today(), help="India panel only.")
    st.caption("India panel only — separate from the Ecuador forecast, whose “today” is 2017-08-15.")
    boost_on = st.toggle("Apply India rule boosts to orders", value=False)
    st.caption("Assumption: rule-based extra stock, not model output.")

if not fams:
    st.info("Pick at least one category in the sidebar.")
    st.stop()

# ---------------- Header + KPIs ----------------
st.title("Dark Store Demand Planner")
st.caption(f"{labels[store]} · forecast for 2017-08-16 to 2017-08-29 · data “today” = 2017-08-15 · "
           f"{len(fams)} of 33 categories · all quantities in units")

k = L.kpis(d, store, fams)
ev = d["eval"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Next 14 days forecast", f"{k['forecast_14d']:,.0f} units",
          f"{k['change_pct']:+.1f}% vs last 14 days", delta_color="off")
c2.metric("Urgent items", k["urgent"], help="Days of cover below 2 (stock is SIMULATED).")
c3.metric("Items to reorder", k["reorder"], help="Order quantity above 0, not urgent.")
c4.metric("Model error on unseen days", f"{ev['test_WAPE']:.1f}%",
          help=f"WAPE on the locked test days {ev['test_period']}. Lower is better.")
c4.caption(f"vs {ev['baseline_test_WAPE']:.1f}% for a simple 14-day average")

# ---------------- Demand chart ----------------
st.subheader("Demand: last 30 days and next 14 days")
series, by_cat = L.demand_series(d, store, fams)
fig = go.Figure()
for i, (name, g) in enumerate(series.groupby("series", sort=False)):
    col = SERIES[i % len(SERIES)] if by_cat else SERIES[0]
    a, f = g[g.kind == "actual"], g[g.kind == "forecast"]
    fig.add_trace(go.Scatter(x=f.date, y=f.high, mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f.date, y=f.low, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(col, 0.15), hoverinfo="skip",
                             name=f"{name} — approximate range" if by_cat else "Approximate range"))
    fig.add_trace(go.Scatter(x=a.date, y=a.units, mode="lines", name=f"{name} — actual" if by_cat else "Actual",
                             line=dict(color=col, width=2), hovertemplate="%{y:,.0f} units"))
    fig.add_trace(go.Scatter(x=f.date, y=f.units, mode="lines", name=f"{name} — forecast" if by_cat else "Forecast",
                             line=dict(color=col, width=2, dash="dash"), hovertemplate="%{y:,.0f} units"))
fig.add_shape(type="line", x0=L.TODAY_DATA, x1=L.TODAY_DATA, y0=0, y1=1, yref="paper",
              line=dict(color=MUTED, width=1, dash="dot"))
fig.add_annotation(x=L.TODAY_DATA, y=1, yref="paper", text="today (2017-08-15)", showarrow=False,
                   xanchor="left", yanchor="bottom", font=dict(color=INK2, size=12))
st.plotly_chart(style(fig, 400), width="stretch")
st.caption("Solid = actual units sold, dashed = forecast. Shaded band = forecast ± each category's test error "
           "(approximate range, not a guarantee). "
           + ("One line per category." if by_cat else "Pick 4 or fewer categories to see one line each."))

# ---------------- Trending + heatmap ----------------
with st.container():
    st.subheader("Trending now")
    up, down = L.trending(d, store, fams)
    t = pd.concat([down, up]).sort_values("change_pct")
    if t.empty:
        st.info("Not enough sales history for the selected categories.")
    else:
        # The % value sits in the row label, so labels never collide with the bars
        fig = go.Figure(go.Bar(
            x=t.change_pct, y=[f"{f}  {v:+.0f}%" for f, v in zip(t.family, t.change_pct)], orientation="h",
            marker=dict(color=[UP if v > 0 else DOWN for v in t.change_pct], cornerradius=4),
            customdata=t[["last_7d_avg", "prev_28d_avg"]],
            hovertemplate="%{y}<br>%{x:+.1f}%<br>last 7 days: %{customdata[0]:,.1f} units/day"
                          "<br>previous 28 days: %{customdata[1]:,.1f} units/day<extra></extra>"))
        style(fig, 340, ytitle="")
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title=dict(text="% change in units/day", font=dict(color=MUTED)), zeroline=True,
                         zerolinecolor=AXIS, showgrid=True, gridcolor=GRID, ticksuffix="%")
        st.plotly_chart(fig, width="stretch")
        st.caption("Last 7 days vs previous 28 days, average units/day. Blue = up, red = down. "
                   "Categories selling under 1 unit/day are skipped.")

with st.container():
    st.subheader("Next 14 days by category")
    hm = L.heatmap_table(d, store, fams)
    hm = hm.loc[hm.sum(axis=1).sort_values().index]
    # Log colour scale: a few big categories would otherwise make all others look the same
    ticks = [t for t in [1, 10, 100, 1_000, 10_000, 100_000] if t <= max(hm.values.max(), 1) * 10]
    fig = go.Figure(go.Heatmap(
        z=np.log10(hm.values + 1), customdata=hm.values,
        x=[c.strftime("%a %d") for c in hm.columns], y=hm.index,
        colorscale=[[i / (len(BLUES) - 1), c] for i, c in enumerate(BLUES)], xgap=2, ygap=2,
        colorbar=dict(title=dict(text="units<br>(log scale)"), thickness=12,
                      tickvals=[np.log10(t + 1) for t in ticks], ticktext=[f"{t:,}" for t in ticks]),
        hovertemplate="%{y}<br>%{x}<br>%{customdata:,.0f} units<extra></extra>"))
    style(fig, max(320, 22 * len(hm) + 80), ytitle="")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickangle=0, title=dict(text="August 2017", font=dict(color=MUTED)))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Forecast units per category and day. Colour uses a log scale so small categories stay visible; "
               "hover for exact units.")

# ---------------- Reorder table ----------------
st.subheader("Reorder plan")
plan = d["plan"][(d["plan"].store_nbr == store) & d["plan"].family.isin(fams)].copy()
alerts = {}
weather_city = st.session_state.get("weather_city", "Mumbai")
if boost_on:
    wdf, _ = weather(weather_city)
    alerts = L.weather_alerts(wdf) if wdf is not None else {}
    plan = L.apply_boosts(plan, L.active_rules(d, planning_date, season_filter, alerts))
plan["status_rank"] = plan.status.map({"Urgent": 0, "Reorder": 1, "OK": 2})
plan = plan.sort_values(["status_rank", "order_qty"], ascending=[True, False])
table = pd.DataFrame({
    "Category": plan.family, "Shelf life": plan.is_perishable.map({1: "Perishable", 0: "Long-life"}),
    "Window (days)": plan.window_days, "Forecast over window": plan.forecast_window, "Buffer": plan.buffer,
    "Stock (SIMULATED)": plan.stock_SIMULATED, "Order qty": plan.order_qty,
    "Days of cover": plan.days_of_cover, "Status": plan.status})
if boost_on:
    table["Boosted order"] = plan.boosted_order
    table["Boost rule (assumption)"] = plan.boost_rule

def status_css(v):
    bg, fg = STATUS[v]
    return f"background-color: {bg}; color: {fg}; font-weight: 600"

num = {c: "{:,.0f}" for c in ["Forecast over window", "Buffer", "Stock (SIMULATED)", "Order qty", "Boosted order"]
       if c in table}
styled = table.style.format({**num, "Days of cover": "{:.1f}"}, na_rep="—").map(status_css, subset=["Status"])
st.dataframe(styled, hide_index=True, width="stretch", height=min(38 * len(table) + 40, 560))
st.caption("Order = forecast over window + safety buffer − stock, minimum 0. Window: perishables 3 days, "
           "long-life 14 days. Urgent = under 2 days of cover · Reorder = order above 0 · OK = enough stock. "
           "**Stock is SIMULATED** (the dataset has no stock data).")
if boost_on:
    st.caption(f"Boosts use the India planning date {planning_date}, season filter “{season_filter}”"
               f"{', weather alerts for ' + weather_city + ': ' + ', '.join(alerts) if alerts else ''}. "
               "If several rules match, the largest "
               "extra % is used. Rule-based assumption, not model output.")
st.download_button("Download reorder plan (CSV)", table.to_csv(index=False).encode(),
                   file_name=f"reorder_plan_store_{store}.csv", mime="text/csv")

# ---------------- India panel ----------------
st.divider()
st.header("India planning — rule-based, not model output")
st.caption(f"Planning date: {planning_date} · season filter: {season_filter} · from data_rules/conditions_rules.csv "
           "and the Python holidays library. Stock-up suggestions are assumptions.")
show_fest = season_filter in ("All", "Festival")
fest = festivals(planning_date)
st.subheader("Upcoming festivals (next 45 days)")
if not show_fest:
    st.info(f"Festivals hidden by the season filter (“{season_filter}”).")
elif fest.empty:
    st.info("No festival with a stocking rule in the next 45 days.")
else:
    for r in fest.itertuples():
        msg = (f"**{r.festival}** — {r.date:%a %d %b %Y} · in {r.days_away} days · start stocking by "
               f"{r.start_stocking_by:%d %b} · +{r.extra_stock_pct}% for {r.related_categories}  \n"
               f"Products: {r.india_products}")
        (st.warning if r.stock_now else st.info)(("⏰ **Start stocking now.** " if r.stock_now else "") + msg)
    st.caption("Library names: " + "; ".join(f"{r.festival} = “{r.library_name}”" for r in fest.itertuples()))

st.subheader("Season")
season = L.current_season(planning_date)
rules = d["rules"].set_index("name")
if season_filter in ("All", season):
    if season:
        st.success(f"**{season}** on {planning_date:%d %b} — in-season products: {rules.loc[season, 'india_products']} "
                   f"(+{rules.loc[season, 'extra_stock_pct']}% for {rules.loc[season, 'related_categories']})")
    else:
        st.info(f"No seasonal rule for {planning_date:%B} (Summer = Mar–May, Monsoon = Jun–Sep, Winter = Nov–Feb).")
elif season_filter == "Festival":
    st.info("Season rules hidden by the season filter (“Festival”).")
else:
    st.info(f"{season_filter} is out of season on {planning_date:%d %b}, so its products are hidden.")

st.subheader("Weather alerts (next 7 days)")
city = st.selectbox("City", list(L.INDIA_CITIES), key="weather_city")
wdf, err = weather(city)
if err:
    st.info(err)
else:
    days = wdf.date.dt.strftime("%a %d")
    wl, wr = st.columns(2)
    fig = go.Figure(go.Bar(x=days, y=wdf.rain_mm, marker=dict(
        color=[SERIES[0] if v >= L.RAINY_MM else BLUES[1] for v in wdf.rain_mm], cornerradius=4),
        hovertemplate="%{x}<br>%{y:.1f} mm<extra></extra>"))
    style(fig, 240, ytitle="rain (mm)")
    fig.update_layout(hovermode="closest", title=dict(text="Rain", font=dict(size=14, color=INK)))
    fig.add_hline(y=L.RAINY_MM, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"rainy day = {L.RAINY_MM} mm", annotation_font_color=INK2)
    wl.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Scatter(x=days, y=wdf.temp_max, mode="lines+markers", line=dict(color=SERIES[1], width=2),
                               marker=dict(size=8), customdata=wdf.temp_min,
                               hovertemplate="%{x}<br>max %{y:.1f} °C · min %{customdata:.1f} °C<extra></extra>"))
    style(fig, 240, ytitle="max temperature (°C)")
    fig.update_layout(hovermode="closest", title=dict(text="Max temperature", font=dict(size=14, color=INK)))
    for y, label in [(L.COLD_MAX_C, f"cold day ≤ {L.COLD_MAX_C} °C"), (L.HOT_MAX_C, f"hot day ≥ {L.HOT_MAX_C} °C")]:
        if wdf.temp_max.min() - 8 <= y <= wdf.temp_max.max() + 8:      # draw a threshold only when it's near
            fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dot"), annotation_text=label,
                          annotation_font_color=INK2)
    wr.plotly_chart(fig, width="stretch")

    alerts_now = L.weather_alerts(wdf)
    icons = {"Rainy day": "🌧", "Cold day": "🥶", "Hot day": "🥵"}
    if not alerts_now:
        st.success(f"No weather alert for {city} in the next 7 days (rain < {L.RAINY_MM} mm, "
                   f"max temperature between {L.COLD_MAX_C} and {L.HOT_MAX_C} °C).")
    for name, dates in alerts_now.items():
        r = rules.loc[name]
        when = ", ".join(pd.to_datetime(dates).strftime("%a %d %b"))
        if season_filter in ("All", r.season):
            st.warning(f"{icons[name]} **{name} expected in {city}** on {when}. Stock up: {r.india_products} "
                       f"(+{r.extra_stock_pct}% for {r.related_categories}).")
        else:
            st.info(f"{name} expected in {city}; this {r.season} rule is hidden by the season filter (“{season_filter}”).")
    st.caption(f"Live forecast from the free Open-Meteo API (real today, not the planning date). Thresholds are "
               f"assumptions: rainy ≥ {L.RAINY_MM} mm, cold ≤ {L.COLD_MAX_C} °C max, hot ≥ {L.HOT_MAX_C} °C max.")

# ---------------- Perishable vs long-life ----------------
st.divider()
st.subheader("Perishable vs long-life")
sl = L.shelf_life_split(d, store, fams)
cols = st.columns(2)
for col, r in zip(cols, sl.itertuples()):
    col.metric(f"{r.group}: share of next-14-day units", f"{r.share_pct:.1f}%")
    col.caption(f"{r.forecast_units:,.0f} units · test error (all stores): **{r.test_WAPE:.1f}%** WAPE")
st.caption("Perishables are easier to predict here mainly because they sell in high volume every day; "
           "low sellers are the hardest (52% error) whatever their shelf life.")

# ---------------- About ----------------
with st.expander("About & limits"):
    st.markdown("""
- **Ecuador data stands in for India.** The model is trained on Favorita store sales (Ecuador, 2013–2017).
- **Stock is simulated.** The dataset has no stock levels, so "Stock (SIMULATED)" is random (seed 42).
- **Category level, not item level** (e.g. all of DAIRY, not one brand of milk).
- **Sales ≠ demand during stock-outs**: if a shelf was empty, true demand was higher than the sales we see.
- **Weather did not help** the model at category level, so it is not used.
- **India rules are assumptions** (festival and season products, extra stock %, lead days), not model output.
- **Planned promotions** for the forecast days come from the dataset's test.csv (a store knows its promotion calendar).
- Model: LightGBM, 31 features, every past-sales clue at least 14 days old. Test error 15.5% vs 22.0% for a simple
  14-day average, on 14 days the model never saw.
""")
