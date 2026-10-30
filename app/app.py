"""Dark Store Demand Planner — Streamlit dashboard.

Layout: header, filters, alert strip, 4 numbers, 6 tabs (Overview, Category, Seasons & festivals, Reorder,
All stores, Model). Reads only precomputed files in app/data/ (src/build_app_data.py); the model never runs here.
All calculations live in app/logic.py. Run from the project root: streamlit run app/app.py
"""
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import logic as L

# Colours: validated categorical slots, up/down, ink, blue ramp, status (always shown with an icon + label)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
UP, DOWN = "#2a78d6", "#d03b3b"
INK, INK2, MUTED, GRID = "#0f172a", "#475569", "#94a3b8", "#e9edf2"
BLUES = ["#e4eefb", "#b9d3f5", "#86b2ec", "#4f8de0", "#2a6fc4", "#1b5299", "#0d366b"]
STATUS_LABEL = {"Urgent": "🔴 Urgent", "Reorder": "🟡 Reorder", "OK": "🟢 OK"}
TRUST_ICON = {"Reliable": "🟢", "Rough guide": "🟡", "Low confidence": "⚪"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

st.set_page_config(page_title="Dark Store Planner", page_icon="📦", layout="wide")
st.html("""<style>
.block-container {max-width: 1200px; padding-top: 3.2rem; padding-bottom: 2rem}
h2, h3 {letter-spacing: -0.01em}
.hero {background: linear-gradient(120deg, #0b1a33 0%, #143058 100%); border-radius: 16px; padding: 26px 30px;
       display: flex; justify-content: space-between; align-items: flex-end; gap: 18px; flex-wrap: wrap; color: #fff}
.hero .brand {font-size: 11.5px; letter-spacing: .16em; text-transform: uppercase; color: #8fb3e8; font-weight: 600}
.hero h1 {font-family: 'Plus Jakarta Sans', sans-serif; font-size: 30px; font-weight: 700; color: #fff;
          margin: 6px 0 4px; padding: 0; letter-spacing: -0.02em}
.hero p {color: #c3d2e8; font-size: 14.5px; margin: 0}
.hero .chips {display: flex; gap: 8px; flex-wrap: wrap}
.hero .chip {background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); border-radius: 99px;
             padding: 5px 12px; font-size: 12.5px; color: #dbe6f5; white-space: nowrap}
.alerts {display: flex; gap: 8px; flex-wrap: wrap; margin: 2px 0 4px}
.alert {border-radius: 10px; padding: 8px 14px; font-size: 13.5px; border: 1px solid}
.alert.red {background: #fdf0f0; border-color: #f6d4d4; color: #8f1d1d}
.alert.amber {background: #fff8e6; border-color: #f6e2ad; color: #7a4e00}
.alert.blue {background: #eef5fd; border-color: #cfe0f6; color: #17457e}
.alert.green {background: #eefaf0; border-color: #cdebd3; color: #1d5e2a}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 4px 0 6px}
.kpis.two {grid-template-columns: repeat(2, minmax(0, 1fr))}
.card {background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 18px;
       box-shadow: 0 1px 2px rgba(15,23,42,.04)}
.label {font-size: 11.5px; letter-spacing: .08em; text-transform: uppercase; color: #64748b; font-weight: 600}
.card .v {font-family: 'Plus Jakarta Sans', sans-serif; font-size: 30px; font-weight: 700; line-height: 1.2;
          color: #0f172a; margin: 6px 0 2px; font-variant-numeric: tabular-nums; letter-spacing: -0.02em}
.card .v.sm {font-size: 22px}
.card .v.red {color: #b42318}
.card .s {font-size: 13px; color: #64748b}
.list {background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 6px 16px}
.row {display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 9px 0;
      border-bottom: 1px solid #f1f4f8; font-size: 14px; color: #0f172a}
.row:last-child {border-bottom: 0}
.row .n {font-variant-numeric: tabular-nums; font-weight: 600; white-space: nowrap}
.row .m {color: #64748b; font-size: 12.5px; white-space: nowrap}
.up {color: #1f5fb0} .down {color: #b42318}
.fest {font-family: 'Plus Jakarta Sans', sans-serif; font-size: 18px; font-weight: 650; color: #0f172a}
.pill {display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 9px; border-radius: 99px;
       background: #fdf0f0; color: #b42318; margin-left: 8px; vertical-align: middle}
.tag {display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 9px; border-radius: 99px;
      background: #eef1f5; color: #475569; vertical-align: middle}
.note {font-size: 12.5px; color: #94a3b8; margin-top: 4px}
.foot {font-size: 12px; color: #94a3b8; margin-top: 36px; padding-top: 14px; border-top: 1px solid #e2e8f0}
.foot a {color: #64748b}
.card.alarm {background: #fff7f7; border-color: #f6d4d4}
.delta-up {color: #1f5fb0; font-weight: 600} .delta-down {color: #b42318; font-weight: 600}
[data-testid="stPlotlyChart"] {background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
                               padding: 12px 14px 6px; box-shadow: 0 1px 2px rgba(15,23,42,.04)}
.st-key-filters {background: #fff; box-shadow: 0 1px 2px rgba(15,23,42,.04)}
button[role="tab"] p {font-size: 15px; font-weight: 600}
button[role="tab"][aria-selected="false"] p {color: #64748b}
</style>""")


@st.cache_data
def data():
    return L.load_data()


@st.cache_data(ttl=3600, show_spinner=False)
def weather(city):
    return L.weather_forecast(city)


@st.cache_data
def festivals(planning_date, within_days=45):
    return L.india_festivals(data(), planning_date, within_days)


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
        height=height, margin=dict(l=0, r=14, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=INK2, size=12),
        legend=dict(orientation="h", y=1.12, x=0, title=None), hovermode="x unified",
        hoverlabel=dict(bgcolor="#fff", bordercolor=GRID, font=dict(color=INK)))
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, tickfont=dict(color=MUTED),
                     title=dict(text=ytitle, font=dict(color=MUTED)), separatethousands=True)
    return fig


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {a})"


def card(label, value, sub, red=False, small=False):
    cls = "v" + (" red" if red else "") + (" sm" if small else "")
    return (f"<div class='card{' alarm' if red else ''}'><div class='label'>{label}</div><div class='{cls}'>{value}</div>"
            f"<div class='s'>{sub}</div></div>")


def delta(pct, text="vs last 14 days"):
    # "+5.9% vs last 14 days", coloured blue (up) or red (down)
    if pd.isna(pct):
        return text
    return f"<span class='{'delta-up' if pct >= 0 else 'delta-down'}'>{'▲' if pct >= 0 else '▼'} {pct:+.1f}%</span> {text}"


def note(text):
    st.html(f"<div class='note'>{text}</div>")


def shade_holidays(fig, hol, start):
    # Light amber column on each Ecuador holiday / event day shown in the chart; returns the names for a note
    hol = hol[hol.date >= start]
    for day in hol.date:
        fig.add_vrect(x0=day - pd.Timedelta(hours=12), x1=day + pd.Timedelta(hours=12), fillcolor="#fde9b8",
                      opacity=0.55, line_width=0, layer="below")
    return " · ".join(f"{r.date:%d %b} {r.name}" for r in hol.itertuples())


def hbar(labels, values, colors, text, height):
    # Horizontal bars, biggest at the top
    fig = go.Figure(go.Bar(y=labels[::-1], x=values[::-1], orientation="h", text=text[::-1],
                           textposition="outside", cliponaxis=False,
                           marker=dict(color=colors[::-1], cornerradius=4),
                           hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
    style(fig, height).update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, max(values) * 1.45 if values else 1])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK2, size=12.5))
    return fig


d = data()
res = d["results"]
bt, test, xmas = res["backtest_6_windows"], res["test_2017-08-02_to_2017-08-15"], res["christmas_backtest"]
trust = L.trust(d)

# ---------------- Header + filters ----------------
hero_box = st.empty()      # filled once the store is chosen

stores = d["stores"]
labels = {r.store_nbr: L.store_label(r) for r in stores.itertuples()}
if "store" not in st.session_state:
    st.session_state.store = 44
with st.container(border=True, key="filters"):
    f1, f2, f3 = st.columns([1.3, 1, 1.9])
    store = f1.selectbox("Store", list(labels), format_func=labels.get, key="store")
    shelf = f2.segmented_control("Shelf life", ["All", "Perishable", "Long-life"], default="All") or "All"
    options = L.families_for(d, shelf)
    picked = f3.multiselect("Categories", options, format_func=nice, placeholder="All categories",
                            key=f"cats_{shelf}")
fams = picked or options
srow = stores.set_index("store_nbr").loc[store]
hero_box.html(f"""<div class='hero'><div>
  <div class='brand'>Dark store planner · quick commerce</div>
  <h1>Demand &amp; Reorder</h1>
  <p>Units each category will sell in the next 14 days, and what to order today.</p></div>
  <div class='chips'><span class='chip'>Store {store} · {srow.city}</span><span class='chip'>Data today · 15 Aug 2017</span>
  <span class='chip'>Forecast · 16 – 29 Aug 2017</span></div>
</div>""")
alert_box = st.empty()     # filled at the end (needs the India and reorder settings)
kpi_box = st.empty()

t_ov, t_cat, t_sea, t_ro, t_all, t_mod = st.tabs(
    [":material/space_dashboard: Overview", ":material/category: Category",
     ":material/celebration: Seasons & festivals", ":material/inventory_2: Reorder",
     ":material/storefront: All stores", ":material/verified: Model"])

# ---------------- Seasons & festivals (built first: alerts and the Reorder tab use its settings) ----------------
with t_sea:
    i1, i2, i3 = st.columns([1, 1.7, 1])
    planning_date = i1.date_input("India planning date", value=dt.date.today())
    season_filter = i2.segmented_control("Season", ["All", "Summer", "Monsoon", "Winter", "Festival"],
                                         default="All") or "All"
    city = i3.selectbox("City (weather)", list(L.INDIA_CITIES))
    rules = d["rules"].set_index("name")
    st.html("<span class='tag'>Rule-based for India · not model output</span>")

    fest = festivals(planning_date)
    show_fest = season_filter in ("All", "Festival")
    timeline = festivals(planning_date, 60)
    if show_fest and not timeline.empty:
        st.subheader("Festival timeline · next 60 days")
        start = pd.to_datetime(timeline.start_stocking_by)
        day = pd.to_datetime(timeline.date)
        fig = go.Figure(go.Bar(
            y=timeline.festival, base=start, x=(day - start).dt.total_seconds() * 1000, orientation="h",
            marker=dict(color=rgba(SERIES[3], 0.55), cornerradius=4), name="stocking window",
            customdata=timeline[["extra_stock_pct"]].assign(s=start.dt.strftime("%d %b"), d=day.dt.strftime("%a %d %b")),
            hovertemplate="%{y}<br>stock from %{customdata[1]}<br>festival %{customdata[2]} · "
                          "+%{customdata[0]}% stock<extra></extra>"))
        fig.add_trace(go.Scatter(y=timeline.festival, x=day, mode="markers+text", text=day.dt.strftime("%d %b"),
                                 textposition="middle right", marker=dict(color=SERIES[1], size=11, symbol="diamond"),
                                 name="festival day", hoverinfo="skip"))
        fig.add_vline(x=pd.Timestamp(planning_date), line=dict(color=MUTED, width=1, dash="dot"))
        style(fig, 34 * len(timeline) + 70).update_layout(hovermode="closest", showlegend=False, bargap=0.45)
        fig.update_xaxes(type="date", range=[pd.Timestamp(planning_date) - pd.Timedelta(days=2),
                                             pd.Timestamp(planning_date) + pd.Timedelta(days=72)], showgrid=True,
                         gridcolor=GRID)
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK2, size=12.5))
        st.plotly_chart(fig, width="stretch")
        note("Bar = when to start stocking up → ◆ festival day · dotted line = planning date.")

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.subheader("Upcoming festivals")
        if not show_fest:
            st.caption(f"Hidden by the season filter ({season_filter}).")
        elif fest.empty:
            st.caption("No festival in the next 45 days.")
        else:
            for r in fest.itertuples():
                pill = "<span class='pill'>stock now</span>" if r.stock_now else ""
                st.html(f"<div class='card' style='margin-bottom:10px'><span class='fest'>{r.festival}</span>{pill}"
                        f"<div class='s' style='margin:4px 0 6px'>{r.date:%a %d %b} · in {r.days_away} days · "
                        f"stock by {r.start_stocking_by:%d %b} · +{r.extra_stock_pct}% stock</div>"
                        f"<div class='s' style='color:#0f172a'>{r.india_products}</div></div>")
    with right:
        st.subheader("Season")
        season = L.current_season(planning_date)
        if season and season_filter in ("All", season):
            st.html(card(planning_date.strftime("%B"), season, rules.loc[season, "india_products"], small=True))
        elif season:
            st.caption(f"{season_filter} filter: {season} products are hidden.")
        else:
            st.caption(f"No season rule for {planning_date:%B}.")

    st.subheader(f"Weather in {city} · next 7 days")
    wdf, err = weather(city)
    alerts = {}
    icons = {"Rainy day": "🌧", "Cold day": "🥶", "Hot day": "🥵"}
    if err:
        st.caption(err)
    else:
        days = wdf.date.dt.strftime("%a %d")
        w1, w2 = st.columns(2, gap="large")
        fig = go.Figure(go.Bar(x=days, y=wdf.rain_mm, marker=dict(
            color=[SERIES[0] if v >= L.RAINY_MM else BLUES[1] for v in wdf.rain_mm], cornerradius=4),
            hovertemplate="%{x}<br>%{y:.1f} mm<extra></extra>"))
        style(fig, 210, "rain (mm)").update_layout(hovermode="closest")
        fig.add_hline(y=L.RAINY_MM, line=dict(color=MUTED, width=1, dash="dot"))
        w1.plotly_chart(fig, width="stretch")
        fig = go.Figure(go.Scatter(x=days, y=wdf.temp_max, mode="lines+markers", line=dict(color=SERIES[1], width=2),
                                   marker=dict(size=8), customdata=wdf.temp_min,
                                   hovertemplate="%{x}<br>max %{y:.1f} °C · min %{customdata:.1f} °C<extra></extra>"))
        style(fig, 210, "max temp (°C)").update_layout(hovermode="closest")
        w2.plotly_chart(fig, width="stretch")

        alerts = L.weather_alerts(wdf)
        shown = {n: v for n, v in alerts.items() if season_filter in ("All", rules.loc[n, "season"])}
        if not shown:
            st.caption("No rain, cold or heat alert this week.")
        else:
            st.html("<div class='list'>" + "".join(
                f"<div class='row'><span>{icons[n]} <b>{n}</b> · {', '.join(pd.to_datetime(v).strftime('%a %d'))}</span>"
                f"<span class='m'>stock up: {rules.loc[n, 'india_products']}</span></div>"
                for n, v in shown.items()) + "</div>")
        note(f"Rainy ≥ {L.RAINY_MM} mm · cold ≤ {L.COLD_MAX_C} °C · hot ≥ {L.HOT_MAX_C} °C (max temperature).")

    st.divider()
    st.subheader("What usually sells more in…")
    month = st.select_slider("Month", MONTHS, value=MONTHS[planning_date.month - 1], label_visibility="collapsed")
    m_no = MONTHS.index(month) + 1
    s1, s2 = st.columns([1.5, 1], gap="large")
    with s1:
        top = L.month_top(d, m_no, fams)
        if top.empty:
            st.caption("No pattern for the selected categories.")
        else:
            st.plotly_chart(hbar([nice(f) for f in top.family], list(top.index_pct), [SERIES[0]] * len(top),
                                 [f"{v:+.0f}%" for v in top.index_pct], 40 * len(top) + 20), width="stretch")
        note("Past pattern, not a forecast: each month vs the category's average month, Aug 2016 – Jul 2017, "
             "all stores (Ecuador).")
    with s2:
        m_season = L.current_season(dt.date(2000, m_no, 1))
        m_year = planning_date.year + (m_no < planning_date.month)     # next time this month comes round
        m_fest = festivals(planning_date, 366)
        m_dates = pd.to_datetime(m_fest.date)
        m_fest = m_fest[(m_dates.dt.month == m_no) & (m_dates.dt.year == m_year)]
        india = [f"<div class='row'><span>Season</span><span class='n'>{m_season or 'no season rule'}</span></div>"]
        if m_season:
            india.append(f"<div class='row'><span class='m' style='white-space:normal'>"
                         f"{rules.loc[m_season, 'india_products']}</span></div>")
        india += [f"<div class='row'><span>🪔 {r.festival}</span><span class='m'>{r.date:%d %b %Y}</span></div>"
                  for r in m_fest.itertuples()]
        if m_fest.empty:
            india.append("<div class='row'><span class='m'>No festival with a stocking rule</span></div>")
        st.html(f"<div class='label' style='margin:6px 0 8px'>India in {month} {m_year} · rule-based</div>"
                f"<div class='list'>{''.join(india)}</div>")

    st.subheader("Search a category's year")
    pat_fams = [f for f in fams if f in set(d["pattern"].family)]
    if pat_fams:
        first = top.family.iloc[0] if not top.empty else pat_fams[0]
        fam_s = st.selectbox("Category", pat_fams, index=pat_fams.index(first), format_func=nice,
                             key=f"pat_{shelf}_{len(fams)}_{m_no}",
                             label_visibility="collapsed", placeholder="Type a category…")
        pat = L.category_pattern(d, fam_s)
        fig = go.Figure(go.Bar(
            x=[m[:3] for m in MONTHS], y=pat.index_pct,
            marker=dict(color=[UP if v >= 0 else DOWN for v in pat.index_pct], cornerradius=4,
                        opacity=[1 if m == m_no else 0.45 for m in pat.month]),
            customdata=pat.avg_units_day,
            hovertemplate="%{x}<br>%{y:+.1f}% vs average month<br>%{customdata:,.0f} units/day (all stores)<extra></extra>"))
        style(fig, 260, "% vs average month").update_layout(hovermode="closest")
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        st.plotly_chart(fig, width="stretch")
        best = pat.nlargest(2, "index_pct")
        note(f"Best months for {nice(fam_s)}: " + ", ".join(f"{MONTHS[m - 1]} ({v:+.0f}%)"
                                                           for m, v in zip(best.month, best.index_pct)))
    else:
        st.caption("No pattern for the selected categories (Books did not sell in every month).")

# ---------------- Reorder ----------------
with t_ro:
    st.subheader(f"Reorder list · store {store}")
    base = d["plan"][(d["plan"].store_nbr == store) & d["plan"].family.isin(fams)]
    r1, r0, r2, r3 = st.columns([1.7, 1.25, 1, 1], vertical_alignment="bottom")
    boost_on = r1.toggle("India extra stock",
                         help="Adds festival / season / weather extra stock, using the Seasons & festivals tab settings. "
                              "Rule-based assumption, not model output.")
    safety = r0.segmented_control("Safety stock", list(L.SAFETY_LEVELS), default="Normal",
                                  help="Low = half the safety buffer, High = 1.5 ×. Normal is the agreed rule.") or "Normal"
    with r2.popover("✏️ Enter real stock", width="stretch"):
        st.caption("Stock in the data is simulated. Type your real stock: orders and status update.")
        stock_in = pd.DataFrame({"Category": base.family.map(nice), "Stock": base.stock_SIMULATED.astype(int)},
                                index=base.index)
        edited = st.data_editor(stock_in, hide_index=True, disabled=["Category"], height=320,
                                key=f"stock_{store}_{shelf}_{len(fams)}_{hash(tuple(fams))}",
                                column_config={"Stock": st.column_config.NumberColumn(min_value=0, step=1)})
    plan = L.replan(base, edited.Stock.fillna(0).astype(float), L.avg_daily_forecast(d, store),
                    L.SAFETY_LEVELS[safety])
    if boost_on:
        plan = L.apply_boosts(plan, L.active_rules(d, planning_date, season_filter, alerts))
    plan["rank"] = plan.status.map({"Urgent": 0, "Reorder": 1, "OK": 2})
    plan = plan.sort_values(["rank", "days_of_cover", "order_qty"], ascending=[True, True, False])
    whole = lambda x: x.round().astype(int)
    table = pd.DataFrame({
        "Category": plan.family.map(nice), "Status": plan.status.map(STATUS_LABEL),
        "Order": whole(plan.boosted_order if boost_on else plan.order_qty),
        "Stock": whole(plan.stock_SIMULATED), "Runs out": plan.days_of_cover.map(L.runs_out),
        "Days of stock": plan.days_of_cover, "Forecast": whole(plan.forecast_window),
        "Safety buffer": whole(plan.buffer), "Window (days)": plan.window_days})
    if boost_on:
        table["Extra because of"] = plan.boost_rule
    with r3.popover("⬇ Download", width="stretch"):
        st.download_button("Table (CSV)", table.to_csv(index=False).encode(), width="stretch",
                           file_name=f"reorder_plan_store_{store}.csv", mime="text/csv")
        to_order = table[table.Order > 0][["Category", "Order", "Window (days)", "Status"]]
        sheet = L.order_sheet(to_order, store, labels[store],
                              f"Data today 15 Aug 2017 · safety stock {safety}"
                              + (" · India extra stock on" if boost_on else "")
                              + " · stock simulated unless entered · open in a browser and print")
        st.download_button("Printable order sheet", sheet.encode(), width="stretch",
                           file_name=f"order_sheet_store_{store}.html", mime="text/html")
    urgent_rows = plan.status.eq("Urgent").to_numpy()
    # Urgent rows get a light red background
    styled = table.style.apply(lambda col: ["background-color: #fdf0f0" if u else "" for u in urgent_rows])
    units = st.column_config.NumberColumn(format="localized")
    st.dataframe(styled, hide_index=True, width="stretch", height=min(35 * len(table) + 38, 640),
                 column_config={"Order": units, "Stock": units, "Forecast": units, "Safety buffer": units,
                                "Days of stock": st.column_config.NumberColumn(format="%.1f")})
    n_changed = int(plan.stock_changed.sum())
    note("Order = forecast for the window + safety buffer − stock. Window: 3 days for fresh food, 14 for long-life. "
         + (f"Safety stock {safety}: buffer × {L.SAFETY_LEVELS[safety]:g}. " if safety != "Normal" else "")
         + (f"Stock: you entered {n_changed} value(s), the rest is simulated." if n_changed
            else "Stock is simulated (the dataset has no stock data)."))

# ---------------- Overview ----------------
k = per_store("kpis", store, tuple(fams))
with t_ov:
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
    fig.add_vrect(x0=L.TODAY_DATA, x1=L.TODAY_DATA + pd.Timedelta(days=14), fillcolor="#eef3fa", line_width=0,
                  layer="below")
    hol_names = shade_holidays(fig, L.store_holidays(d, store), series.date.min())
    st.plotly_chart(style(fig, 360, "units / day"), width="stretch")
    note("Solid = actual · dashed = forecast (blue shade) · "
         + ("band = likely range (80%)" if by_cat else "pick up to 4 categories to see each one with its range")
         + (f" · amber = Ecuador holiday: {hol_names}" if hol_names else ""))

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.subheader("Top sellers this week")
        tn = per_store("top_now", store, tuple(fams))
        st.html("<div class='list'>" + "".join(
            f"<div class='row'><span>{i + 1}. {nice(r.family)}</span>"
            f"<span><span class='n'>{r.units:,.0f}</span> <span class='m'>{r.share_pct:.0f}%</span></span></div>"
            for i, r in enumerate(tn.itertuples())) + "</div>")
        note("Units sold in the last 7 days (to 15 Aug 2017).")
    with c2:
        st.subheader("Moving this week")
        up, down = per_store("trending", store, tuple(fams))
        ups = [(r.family, r.change_pct, "up", "▲") for r in up.head(3).itertuples()]
        downs = [(r.family, r.change_pct, "down", "▼") for r in down.sort_values("change_pct").itertuples()]
        rows = ups + downs[:5 - len(ups)]
        if rows:
            st.html("<div class='list'>" + "".join(
                f"<div class='row'><span>{nice(f)}</span><span class='n {c}'>{a} {v:+.0f}%</span></div>"
                for f, v, c, a in rows) + "</div>")
        else:
            st.caption("Not enough sales history.")
        note("Last 7 days vs the 4 weeks before.")
    with c3:
        st.subheader("Order now")
        daily = L.avg_daily_forecast(d, store)
        now = plan[(plan.status != "OK") & (plan.family.map(daily) >= L.LOW_SELLER)].head(5)
        if now.empty:
            st.caption("Nothing to order for the selected categories.")
        else:
            st.html("<div class='list'>" + "".join(
                f"<div class='row'><span>{'🔴' if r.status == 'Urgent' else '🟡'} {nice(r.family)}</span>"
                f"<span><span class='m'>out {L.runs_out(r.days_of_cover)}</span> "
                f"<span class='n'>{(r.boosted_order if boost_on else r.order_qty):,.0f}</span></span></div>"
                for r in now.itertuples()) + "</div>")
        note("Runs out first · order in units · sellers under 1 unit/day are in the Reorder tab only.")

    st.subheader("Perishable vs long-life")
    sl = per_store("shelf_life_split", store, tuple(fams))
    st.html("<div class='kpis'>" + "".join(
        card(r.group, f"{r.share_pct:.0f}%", f"of next-14-day units · {r.forecast_units:,.0f} units · "
             f"{r.backtest_WAPE:.1f}% error in back-tests", small=True) for r in sl.itertuples()) + "</div>")

# ---------------- Category ----------------
with t_cat:
    rk = per_store("ranking", store, tuple(fams))
    left, right = st.columns([1, 1.45], gap="large")
    with right:
        fam = st.selectbox("Search a category", list(rk.family), format_func=nice, key=f"cat_{shelf}_{len(fams)}")
    with left:
        st.subheader("Who comes on top · next 14 days")
        top10 = rk.head(10)
        moves = []
        for r in top10.itertuples():
            step = r.rank_last - r.rank_next
            moves.append(f"{r.next_14d:,.0f} · " + (f"▲ {step}" if step > 0 else f"▼ {-step}" if step < 0 else "="))
        st.plotly_chart(hbar([nice(f) for f in top10.family], list(top10.next_14d),
                             [SERIES[0] if f == fam else BLUES[1] for f in top10.family], moves,
                             32 * len(top10) + 30), width="stretch")
        note("Forecast units, next 14 days · ▲▼ = places gained or lost vs the last 14 days.")
    with right:
        r = rk.set_index("family").loc[fam]
        tl, terr = trust[fam]
        st.html("<div class='kpis two'>" + "".join([
            card("Next 14 days", f"{r.next_14d:,.0f}", "units forecast", small=True),
            card("vs last 14 days", "—" if pd.isna(r.change_pct) else
                 f"<span class='{'delta-up' if r.change_pct >= 0 else 'delta-down'}'>{r.change_pct:+.1f}%</span>",
                 f"{r.last_14d:,.0f} units sold", small=True),
            card("Rank", f"#{int(r.rank_last)} → #{int(r.rank_next)}", "last → next 14 days", small=True),
            card("Trust", f"{TRUST_ICON[tl]} {tl}", f"{terr:.1f}% error on test days", small=True),
        ]) + "</div>")
        series, _ = per_store("demand_series", store, (fam,))
        a, f = series[series.kind == "actual"], series[series.kind == "forecast"]
        fig = go.Figure([
            go.Scatter(x=f.date, y=f.high, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=f.date, y=f.low, mode="lines", line=dict(width=0), fill="tonexty",
                       fillcolor=rgba(SERIES[0], 0.13), name="likely range (80%)", hoverinfo="skip"),
            go.Scatter(x=a.date, y=a.units, mode="lines", name="actual", line=dict(color=SERIES[0], width=2),
                       hovertemplate="%{y:,.0f} units"),
            go.Scatter(x=f.date, y=f.units, mode="lines", name="forecast", line=dict(color=SERIES[0], width=2, dash="dash"),
                       hovertemplate="%{y:,.0f} forecast")])
        promo = L.promo_days(d, store, fam).merge(f[["date", "units"]], on="date")
        promo = promo[promo.onpromotion > promo.onpromotion.median()]       # more promotions than usual
        if not promo.empty:
            fig.add_trace(go.Scatter(x=promo.date, y=promo.units, mode="markers", name="extra promotions",
                                     marker=dict(color=SERIES[1], size=8, line=dict(color="#fff", width=2)),
                                     customdata=promo.onpromotion,
                                     hovertemplate="%{customdata} items on promotion<extra></extra>"))
        fig.add_vline(x=L.TODAY_DATA, line=dict(color=MUTED, width=1, dash="dot"))
        cat_hol = shade_holidays(fig, L.store_holidays(d, store), a.date.min())
        st.plotly_chart(style(fig, 250, "units / day"), width="stretch")
        note("Orange dots = more items on promotion than usual (above the 14-day median of the store's plan)"
             + (f" · amber = Ecuador holiday: {cat_hol}" if cat_hol else ""))

    st.subheader(f"Why is {nice(fam)} moving?")
    why = L.category_why(d, store, fam)
    w1, w2 = st.columns([1.45, 1], gap="large")
    with w1:
        fig = go.Figure(go.Bar(y=why.reason, x=why.effect_pct, orientation="h",
                               marker=dict(color=[UP if v >= 0 else DOWN for v in why.effect_pct], cornerradius=4),
                               text=[f"{v:+.1f}%" for v in why.effect_pct], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>%{x:+.1f}%<extra></extra>"))
        style(fig, 280).update_layout(hovermode="closest", bargap=0.4)
        fig.add_vline(x=0, line=dict(color=MUTED, width=1))
        lo, hi = min(why.effect_pct.min(), 0), max(why.effect_pct.max(), 0)
        pad = (hi - lo) * 0.25 + 0.5
        fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=GRID, range=[lo - pad, hi + pad])
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK2, size=12.5))
        st.plotly_chart(fig, width="stretch")
    with w2:
        top_up, top_down = why.iloc[-1], why.iloc[0]
        st.html("<div class='list'>"
                f"<div class='row'><span>Biggest push up</span><span class='n up'>{top_up.reason} {top_up.effect_pct:+.1f}%</span></div>"
                f"<div class='row'><span>Biggest push down</span><span class='n down'>{top_down.reason} {top_down.effect_pct:+.1f}%</span></div>"
                "</div>")
        note("The model's own breakdown of each forecast (LightGBM contributions): how much each reason pushes the "
             "next 14 days up or down compared with the model's view of the last 14 days. Effects multiply, so they "
             "do not add up exactly.")
        pr = plan.set_index("family")
        if fam in pr.index:
            p = pr.loc[fam]
            order = p.boosted_order if boost_on else p.order_qty
            st.html("<div class='list' style='margin-top:10px'>"
                    f"<div class='row'><span>Status</span><span class='n'>{STATUS_LABEL[p.status]}</span></div>"
                    f"<div class='row'><span>Runs out</span><span class='n'>{L.runs_out(p.days_of_cover)}</span></div>"
                    f"<div class='row'><span>Order</span><span class='n'>{order:,.0f} units</span></div></div>")

# ---------------- All stores ----------------
def open_store():
    rows = st.session_state.all_pick.selection.rows
    if rows:
        st.session_state.store = int(st.session_state.all_table.iloc[rows[0]]["Store"])


with t_all:
    st.subheader("Which stores need attention?")
    al = L.all_stores(d)
    tbl = pd.DataFrame({"Store": al.store_nbr, "City": al.city, "Type": al.store_type, "Urgent": al.urgent,
                        "Reorder": al.reorder, "Next 14 days": al.next_14d.round().astype(int),
                        "vs last 14 days": al.change_pct})
    st.session_state.all_table = tbl
    st.dataframe(tbl, hide_index=True, width="stretch", height=520, key="all_pick", on_select=open_store,
                 selection_mode="single-row", column_config={
                     "Urgent": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=int(al.urgent.max()),
                                                               color="red"),
                     "Next 14 days": st.column_config.NumberColumn(format="localized"),
                     "vs last 14 days": st.column_config.NumberColumn(format="%+.1f%%")})
    note("All categories, simulated stock. Sorted by urgent categories · click a row to open that store.")

# ---------------- Model ----------------
with t_mod:
    st.html("<div class='kpis'>" + "".join([
        card("Back-tests (6 × 14 days)", f"{bt['chosen_v2']['mean']:.1f}%",
             f"error · simple average {bt['chosen_v2']['baseline_mean']:.1f}%"),
        card("Unseen test days", f"{test['v1']['WAPE']:.1f}%",
             f"first model · simple average {test['baseline']['WAPE']:.1f}%"),
        card("Christmas 2016", f"{xmas['v2']['WAPE']:.1f}%", f"error · simple average {xmas['WAPE_baseline']:.1f}%"),
        card("80% range", f"{bt['range_coverage_pct']['mean']:.1f}%", "of actual values inside it (back-tests)"),
    ]) + "</div>")
    note("Error = WAPE: units predicted wrong ÷ units sold. Lower is better.")

    m1, m2 = st.columns([1.4, 1], gap="large")
    with m1:
        st.subheader("Test days: actual vs forecast")
        tc = L.test_check(d, fams)
        fig = go.Figure([
            go.Scatter(x=tc.date, y=tc.sales, mode="lines+markers", name="actual", line=dict(color=INK, width=2),
                       marker=dict(size=6), hovertemplate="%{y:,.0f} actual"),
            go.Scatter(x=tc.date, y=tc.pred, mode="lines", name="model (v2)", line=dict(color=SERIES[0], width=2),
                       hovertemplate="%{y:,.0f} model"),
            go.Scatter(x=tc.date, y=tc.baseline, mode="lines", name="simple average",
                       line=dict(color=MUTED, width=2, dash="dot"), hovertemplate="%{y:,.0f} simple average")])
        st.plotly_chart(style(fig, 300, "units / day, all stores"), width="stretch")
        note(f"2–15 Aug 2017, selected categories. The current model (v2) scored {test['v2']['WAPE']:.1f}% here, "
             f"but only after these days had been seen; the first model's unseen score is {test['v1']['WAPE']:.1f}%.")
    with m2:
        st.subheader("Error by category")
        ce = d["cat_err"][d["cat_err"].family.isin(fams)].sort_values("WAPE_model")
        shown_ce = ce.head(12)
        st.plotly_chart(hbar([f"{TRUST_ICON[trust[f][0]]} {nice(f)}" for f in shown_ce.family],
                             list(shown_ce.WAPE_model.clip(upper=100)), [SERIES[0]] * len(shown_ce),
                             [f"{v:.1f}%" for v in shown_ce.WAPE_model], 26 * len(shown_ce) + 30), width="stretch")
        note(f"Easiest {len(shown_ce)} of {len(ce)} · first model on unseen test days · 🟢 under {L.RELIABLE_WAPE}% · "
             f"🟡 under {L.ROUGH_WAPE}% · ⚪ higher, or under 1 unit/day per store.")

    st.subheader("How it works")
    st.markdown(f"""
- **Model:** LightGBM learns from 2015–2017 sales. Every past-sales clue is at least 14 days old, so all 14 days
  are forecast at once without peeking at the future.
- **Fine-tuning** was judged on the 6 back-tests only: {bt['chosen_v2']['mean']:.1f}% (current) vs
  {bt['experiment_mean']['v1']['WAPE']:.1f}% (first model) vs {bt['chosen_v2']['baseline_mean']:.1f}% (simple average).
- **More tries, none kept:** a bigger model, another training start year, extra holiday and promotion clues,
  and an average of 3 models. None passed the rule set before the run (at least 0.3 points better on the back-tests,
  at most 1 window worse). Closest: average of 3 models, {res['experiments_6i']['steps']['5 seeds']['backtest_mean_new']:.2f}%
  vs {res['experiments_6i']['v2_backtest_mean']:.2f}%.
- **Limits:** Ecuador supermarket data stands in for India · stock is simulated · category level, not single items ·
  India festival, season and weather rules are assumptions.
""")

# ---------------- Alert strip + numbers (top of the page) ----------------
chips = []
n_urgent = int((plan.status == "Urgent").sum())
if n_urgent:
    chips.append(f"<div class='alert red'>🔴 <b>{n_urgent}</b> categor{'y runs' if n_urgent == 1 else 'ies run'} "
                 f"out within 2 days in store {store}</div>")
else:
    chips.append("<div class='alert green'>🟢 Nothing runs out in the next 2 days</div>")
if show_fest and not fest.empty:
    f0 = fest.iloc[0]
    chips.append(f"<div class='alert amber'>🪔 <b>{f0.festival}</b> in {f0.days_away} days · "
                 f"{'stock up now' if f0.stock_now else f'stock by {f0.start_stocking_by:%d %b}'}</div>")
if shown_w := [n for n in alerts if season_filter in ("All", rules.loc[n, "season"])]:
    first = shown_w[0]
    chips.append(f"<div class='alert blue'>{icons[first]} "
                 f"<b>{first}</b> in {city} on {pd.to_datetime(alerts[first][0]):%a %d %b}</div>")
alert_box.html(f"<div class='alerts'>{''.join(chips)}</div>")

kpi_box.html("<div class='kpis'>" + "".join([
    card("Next 14 days", f"{k['forecast_14d']:,.0f}", "units · " + delta(k["change_pct"])),
    card("Urgent", n_urgent, "run out within 2 days", red=n_urgent > 0),
    card("To reorder", int((plan.status == "Reorder").sum()), "categories that need an order"),
    card("Model error", f"{bt['chosen_v2']['mean']:.1f}%", f"vs {bt['chosen_v2']['baseline_mean']:.1f}% for a simple average"),
]) + "</div>")

st.html("<div class='foot'>Data: <a href='https://www.kaggle.com/competitions/store-sales-time-series-forecasting'>"
        "Favorita store sales (Kaggle)</a> · festival dates: Python holidays library · "
        "<a href='https://open-meteo.com/'>Weather data by Open-Meteo.com</a> (CC BY 4.0) · all numbers in units, "
        "not money</div>")
