# Dark Store Demand Forecasting — project context

## Who / how
- I'm Samarth, a university student. This is my DSML project. Explain things in simple, short English.
- Accuracy matters less than CORRECTNESS: every number must be honest and reproducible.
- After each stage: short report, then STOP and wait for me.

## End goal
Quick-commerce companies (Blinkit, Zepto, Instamart) run small "dark stores". Running out of an item = bad customer experience; overstocking = waste.
Build an ML model that predicts units sold per store × product category for the next 14 days, turn it into a reorder list, and show it in a Streamlit dashboard with:
- store dropdown
- next-14-day forecast chart per category
- current-demand graph (last 30 days actual joined to the forecast)
- "Trending now" (categories most above their usual level this week)
- reorder table (red when stock runs out soon)
- festival alerts for India (e.g. "Diwali in N days — stock up sweets, gifts, toys")
- season filter (Summer / Monsoon / Winter / Festival) that hides out-of-season items
- perishable vs long-life view
The project has 3 "extras": (1) festival effect, (2) perishable vs long-life predictability, (3) reorder dashboard.

## Data
Main training data: Kaggle "Store Sales – Time Series Forecasting" (Favorita, Ecuador): 54 stores, 33 product categories, daily, 2013-01-01 to 2017-08-15. Ecuador stands in for Indian dark stores (honest limitation).
Folder: ~/DSML-Project
- raw/store_sales/       train, stores, holidays_events, oil, transactions CSVs. test.csv is used ONLY for planned onpromotion on the forecast dates (it has no sales), nothing else.
- raw/online_retail/     UCI Online Retail II (UK gifts) — optional, winter/gift analysis only
- raw/weather_ecuador/   weather.csv (date, city, rain_mm, temp_max), 37,136 rows, Open-Meteo, 22 cities — the ONE code-generated file allowed in raw/ (written by 03_model when missing; external API download = raw data)
- raw/national_holiday/, raw/extra/ (m5, favorita full, Diwali datasets, archive) — NOT used for training
- clean/store_sales_clean.parquet — cleaned table, 2,785,464 rows × 21 columns
- notebooks/ 01_cleaning, 02_eda, 03_model, 04_evaluation (+ *_run = executed copies)
- models/ lgbm_v1 (Stage 4), lgbm_final (v1, Stage 5), lgbm_final_v2 (v2, current), lgbm_q10 / lgbm_q90 (80% range); *_info.json
- reports/ charts q1–q6, q8–q12 (no q7), results.json (SINGLE SOURCE OF TRUTH for every number), test_metrics.json, backtest*.csv, christmas_backtest.json, forecast_next14.parquet, reorder_plan.parquet (+ .csv copies for download), test_predictions.csv (v1) / _v2.csv, category_errors.csv, festival_effect.csv
- src/ features.py (v1 build_features unchanged + v2 add-on), model.py (settings, CHOSEN = "v2"), train.py, backtest.py, forecast.py, reorder.py, results.py, build_app_data.py — each has a module docstring
- app/ app.py + logic.py + data/ (0.7 MB); data_rules/ India rules; tests/test_pipeline.py (pytest, 15 checks; 6 marked needs_data → skipped without clean/ and raw/) + tests/test_models_and_app.py (7 checks, no data needed: models load/predict/reject bad input, results.json = saved outputs, dashboard runs); src/tune.py (6i experiments)
- .github/workflows/ci.yml — GitHub Actions: CI (Python 3.12, structure, CHECKSUMS, syntax, pytest → 16 passed / 6 skipped without data) + CD (bundle models/ + reports as a 30-day artifact; no deploy, no release). Remote: origin = https://github.com/samarthmaru24/dark-store-demand-forecasting.git
- run_all.py: [--retrain → train] → forecast → reorder → backtest → results.json → app data
- make_03_model.py (generator for 03_model.ipynb), make_stage5.py (generator for Stage 5; ran once)
- CHECKSUMS.txt — SHA-256 of generator scripts, generated files and pipeline code. src/features.py was extended after generation (v2 add-on); tests/test_pipeline.py checks that the generated v1 build_features code is still in it unchanged.
- .gitignore excludes raw/ and clean/ (data never goes to GitHub). Local git repo initialised here; GitHub remote is connected in Stage 7.
Clean table columns: date, store_nbr, family, sales, onpromotion, is_closed, city, state, store_type, cluster, is_event, is_national_holiday, is_regional_holiday, is_local_holiday, is_holiday, dcoilwtico, transactions, is_payday, is_quake, is_perishable, never_sold.
"sales" = UNITS sold (not money). Never write $, ₹ or "revenue".

## HARD RULES (always)
1. Never modify, move or delete anything in raw/. Only exception: raw/weather_ecuador/weather.csv may be (re)downloaded by code. No other code may write to raw/.
2. The 14-day rule: any past sales / oil / transactions feature must be at least 14 days old. No same-day oil or transactions. Split by time, never shuffle.
3. Test period 2017-08-02 to 2017-08-15 is locked — used first in Stage 5 (v1). Model changes are accepted or rejected ONLY on back-test windows; after a change, report test WAPE for v1 and v2 side by side and say the test was already seen.
4. Report only numbers actually printed by code.
5. Never silently change agreed logic or rewrite a pre-written notebook. If a change is needed: explain what and why, smallest fix, and wait for my OK.
6. Pre-written files come from generator scripts with SHA-256 checksums; always verify the checksum before running.

## Roadmap (7 stages) and status
1. Plan & setup — DONE (folders, GitHub, Kaggle, Python libs)
2. Get & clean data — DONE
   Rules: missing days (25 Dec) added with is_closed=1; store-days with 0 total sales across all categories also is_closed=1 (e.g. 1 Jan); rows before each store's first sale dropped (8 stores opened late, store 52 only on 2017-04-20); holidays: drop transferred=True and "Work Day"; National = all stores, Regional = match state, Local = match city, "Event" → is_event; oil ffill then bfill; is_payday = 15th or month-end; is_quake = 2016-04-16..2016-05-15; is_perishable = BREAD/BAKERY, DAIRY, DELI, EGGS, MEATS, POULTRY, PREPARED FOODS, PRODUCE, SEAFOOD; never_sold = store×category pairs with zero total sales (53 pairs).
3. Explore data (EDA) — DONE
   Findings: top 3 = GROCERY I 32%, BEVERAGES 20%, PRODUCE 11% of units; Sunday +29%, Thursday −21% vs average; avg units/day grew from about 385k (2013) to 856k (2017); holiday-day lift (holiday vs normal days): FROZEN FOODS +223%, LIQUOR,WINE,BEER +124%; build-up (days just before holidays vs normal days): FROZEN FOODS +92.9%, LIQUOR,WINE,BEER +73.8%; BOOKS lift (+76%) is from a tiny base (0.07 → 0.13 units/day), not meaningful; jumpiness (CV) perishable 0.38 vs long-life 1.07 — mainly because many long-life categories sell very few units; promotion lifts are a link, not proof; payday +1%.
4. Build model — DONE (notebooks/03_model.ipynb, verified by checksum)
   LightGBM on log1p(sales), 34 features: store/category info, calendar, holiday flags, days_to_holiday, onpromotion, lag_14/21/28/35/42, mean_7/28/56 and std_28 of sales shifted 14 days, trans_lag14, oil_lag14, rain_mm, temp_max, rainy_day. Closed days excluded from training; never_sold pairs dropped.
   Split: train 2015-01-01..2017-07-18 (1,538,954 rows), valid 2017-07-19..2017-08-01 (24,206 rows), test locked 2017-08-02..2017-08-15.
   Validation WAPE: baseline (average of last 14 days) 21.81%, LightGBM 12.82%, LightGBM without weather 12.74% → about 41% better than baseline. Weather did NOT help (honest finding). Top features: lag_14 51%, mean_7 26%, mean_28 16%.
   Note: the city "Libertad" geocoded to a place near Guayaquil, not the coastal town — no impact since weather isn't used.
5. Test & understand results — DONE (notebooks/04_evaluation_run.ipynb, checksums in CHECKSUMS.txt)
   Results (as printed): Use weather: False | 31 features | 140 trees. Train 2015-01-01..2017-08-01, Test 2017-08-02..2017-08-15.
   Test WAPE: baseline 21.98% (MAE 103.91), LightGBM 15.51% (MAE 73.34) → 29.4% lower than baseline.
   Test (15.5%) is worse than validation (12.7%): mostly 12–13 Aug (weekend after the 11 Aug transferred national holiday), over-forecast by ~21%; the model has no "days since holiday" feature. Diagnostic only: 13.63% without those 2 days — the official number stays 15.51%.
   Easiest: PRODUCE 10.1, BREAD/BAKERY 12.6, DAIRY 12.7, GROCERY I 12.7, MEATS 13.5. Hardest: SCHOOL AND OFFICE SUPPLIES 70.2 (back-to-school season), HARDWARE 73.9, BABY CARE 134.6, HOME APPLIANCES 192.8, BOOKS 420.0 (tiny bases ≤1.4 units/day → WAPE not meaningful; BOOKS model worse than baseline).
   Perishable 12.0% vs long-life 17.2%. Low sellers 52.2%, medium 24.5%, top 15.1%.
   Christmas 2016 back-test (closed 25 Dec excluded): LightGBM 20.1% vs baseline 27.9%; total units −11.0% (baseline −13.2%); FROZEN FOODS predicted 445,398 vs actual 795,112 (catches part of the rush, not all).
   Saved: models/lgbm_final.txt + lgbm_final_info.json, reports/category_errors.csv, test_predictions.csv, q9–q11.
   For Stage 6: safety buffers from category WAPE must be capped/handled for tiny categories (e.g. BOOKS 420%) — agree the rule with me first.
   make_stage5.py creates src/features.py (shared feature code) and notebooks/04_evaluation.ipynb: final test on locked 14 days, error per category (saves reports/category_errors.csv), perishable vs long-life, low vs top sellers, Christmas 2016 back-test (12–25 Dec 2016, Extra 1), and trains the final model on all data → models/lgbm_final.txt + lgbm_final_info.json. Uses the no-weather feature set automatically if it had lower validation error.
6. Reorder engine + Streamlit dashboard — DONE
   6a Forecast engine — DONE: src/forecast.py (python3 src/forecast.py) → reports/forecast_next14.csv (now .parquet + .csv copy, v2 model — see 6d), 2017-08-16..2017-08-29, 24,206 rows (54 stores × 33 categories minus never_sold). Reuses src/features.py unchanged; holiday flags rebuilt with the 01_cleaning rules (verified identical on history); adds 60 calendar-only days so days_to_holiday sees the 9 Oct holiday (54 → 41).
      With planned promotions (decided in 6b): total forecast 11,130,075 units vs 11,444,795 in the last 14 days (−2.7%). (Earlier, with promotions = 0: 10,765,373, −5.9%.)
      Edge effect found: in the original features, days_to_holiday on 2017-08-12..15 was 60 (data ended) instead of 55–58; only those 4 days, tiny effect on lgbm_final.
   6b Bias check + reorder plan + India rules — DONE
      Bias check (reports/test_predictions.csv): overall +4.88% (OVER-forecast), long-life +4.71%, perishable +5.24%.
      DECISION: NO bias correction. Test was +4.9% (over) but the Christmas back-test was −11% (under), so a single factor isn't reliable; slight over-forecast is the safer error for stock-outs.
      DECISION: no new model features (e.g. days_since_holiday) — the test set has been seen, so changing the model now would leak test information.
      src/reorder.py → reports/reorder_plan.csv (1,729 rows; Stage 6b version — buffer and Urgent rule replaced in 6d). Window perishable 3 / long-life 14 days; buffer = forecast_window × test WAPE capped 50% (capped: HARDWARE, SCHOOL AND OFFICE SUPPLIES); fixed 1-unit buffer for < 1 unit/day per store (BABY CARE, BOOKS, HOME APPLIANCES); order_qty rounded up.
      stock_SIMULATED = recent 14-day avg daily sales × U(0.3, 1.5) × window_days, seed 42. days_of_cover = stock / avg daily forecast (rounded to 0.1 before the status, so shown number and status agree; empty if forecast is 0).
      Status: "Urgent" if days_of_cover < 2, else "Reorder" if order_qty > 0, else "OK".
      Result: Urgent 257 / Reorder 995 / OK 477 (long-life 111 / 843 / 289; perishable 146 / 152 / 188).
      Quirk: tiny sellers whose simulated stock rounds to 0 (e.g. BOOKS, HOME APPLIANCES in store 44) show as Urgent with 0.0 days of cover.
      data_rules/conditions_rules.csv (my draft, rule-based assumptions) + data_rules/festival_name_map.csv (rule name → `holidays` library name, all 8 found; Karwa Chauth = "Karaka Chaturthi (Karwa Chouth)", Diwali = "Diwali (Deepavali)", Ganesh Chaturthi = "Ganesh Chaturthi / Vinayak Chaturthi"; Id-ul-Fitr 2027 is "(estimated)"). All related_categories match the 33 families. Library: holidays 0.107.
   6c Streamlit dashboard — DONE: app/app.py (UI) + app/logic.py (all calculations, testable without Streamlit).
      Run from the project root: streamlit run app/app.py. Theme in .streamlit/config.toml. requirements.txt pinned (Python 3.14.6 locally).
      Reads ONLY app/data/ (0.7 MB, tracked in git; rebuilt by python3 src/build_app_data.py): history_last60.parquet (2017-06-17..2017-08-15), stores.csv, forecast_next14.parquet, reorder_plan.parquet, category_errors.csv, conditions_rules.csv, festival_name_map.csv, results.json. Never runs the model.
      Pipeline order: python3 src/forecast.py → python3 src/reorder.py → python3 src/build_app_data.py.
      Boosts: boosted_order = order + forecast_window × extra_stock_pct for active rules (festivals whose "start stocking by" date has arrived, the current season, rainy-day if the rain check found ≥ 5 mm). If several rules match a family, the largest % is used (no stacking). Season filter keeps only that season's rules.
      Seasons from planning-date month: Summer Mar–May, Monsoon Jun–Sep (wins June), Winter Nov–Feb; October has no season rule.
      Weather alerts: Open-Meteo forecast API (precipitation_sum, temperature_2m_max/min, 7 days, real today, 8 Indian cities); friendly message if it fails.
      Thresholds (assumptions): Rainy day rain ≥ 5 mm (Monsoon), Cold day max ≤ 15 °C (Winter), Hot day max ≥ 40 °C (Summer). "Cold day" and "Hot day" rows added to conditions_rules.csv (+10%, lead 1 day), same pattern as "Rainy day".
      Dummy/synthetic data: used ONLY to test that alerts trigger (never for training or accuracy numbers).
      Trending: last 7 days vs previous 28 days average units/day, open days only, categories under 1 unit/day skipped. Heatmap uses a log colour scale (hover = exact units).
      Smoke test (2026-10-24): store 44 next 14 days 578,586 units (+1.5% vs last 14 days), 4 Urgent, 19 Reorder; 0 exceptions in AppTest runs and server log.
   DASHBOARD DATE DESIGN: the forecast/reorder side uses the data's "today" = 2017-08-15 (Ecuador demo). The India festival/season panel uses an "India planning date" picker (default = real today), clearly labelled as a separate, rule-based panel.
   - conditions_rules.csv: India condition → product groups, with a season column. Rule-based assumptions, not model output — label them so.
   - Indian festival dates: Python `holidays` library, holidays.country_holidays("IN", years=..., categories=("public", "optional")).
   6d Fine-tuning — DONE (CURRENT STATE; supersedes the older 6a/6b model, buffer, status and file details above)
      A1 Back-tests (src/backtest.py): 6 windows of 14 days starting 2017-01-16, 02-27, 04-10, 05-22, 06-19, 07-19; train 2015-01-01 → window start; mean = simple average of window WAPEs.
      A3+A4 experiment (python3 src/backtest.py --experiment, same 140 trees for all): mean WAPE baseline 25.31 | v1 15.32 | v1+A4 14.70 | v1+A4+Tweedie 13.72; perishable 23.67 | 14.72 | 13.52 | 12.05; bias +0.96 | −4.17 | −3.97 | −1.96%.
      DECISION: keep v1+A4+Tweedie = v2 (1.60 points better ≥ 0.3, perishable better). Better in 5 of 6 windows; worse on 2017-04-10 (17.59 vs 15.63).
      v2 = v1 + days_since_holiday (cap 60) + sales_lag_364, objective tweedie (variance_power 1.2) on raw sales. v1 still reproducible (model.variant("v1")).
      Test days (ALREADY SEEN, v2 scored afterwards): baseline 21.98 | v1 15.51 | v2 13.66 (perishable 10.6, long-life 15.1; low sellers 57.3 (v1 52.2) — worse, top sellers 13.1 (v1 15.1)).
      Christmas back-test: v1 20.1% (−11.0% units, reproduces Stage 5 exactly) | v2 17.8% (−2.2% units; FROZEN FOODS 750,276 vs actual 795,112).
      A7 Ranges (first version, replaced in 6g): quantile LightGBM (alpha 0.1 / 0.9, log1p target, v2 features). Back-test coverage mean 82.8% (81.6–84.4; perishable 81.2, long-life 83.5), test 83.6%. Forecast range clipped to include the point forecast in 155 of 24,206 rows (0.6%). Row ranges can't be summed into a total range → the app shows bands only per category.
      Reorder (current): buffer = Σ over window max(forecast_high − forecast, 0), capped at 50% of forecast_window (624 rows capped), ≥ 1 unit for tiny categories (BABY CARE, BOOKS, HOME APPLIANCES). B3: Urgent only if avg daily forecast ≥ 1. Result with the first ranges: Urgent 206 / Reorder 1,237 / OK 286 (current: see 6g).
      Forecast (v2, 2017-08-16..29): 11,236,282 units vs 11,444,795 last 14 days (−1.8%).
      B5 tests: 10 passed (lag_14 / sales_lag_364 / trans / oil age, no same-day inputs, v1 code unchanged, splits, outputs, hand-made reorder example, festival dates, rule categories). A mutated lag test fails as expected.
      B6 Python: Streamlit Cloud supports released Python versions still receiving security updates; pick it in "Advanced settings" (runtime.txt is ignored). Tested on 3.14 (local) and a fresh 3.12 venv from requirements.txt: 10 tests passed, app 0 exceptions.
      Reproducibility: python3 run_all.py --retrain reproduces results.json and the reorder plan identically.
      App: band = 80% range (≤ 4 categories), KPI = back-test error 13.7% vs 25.3% (test v1/v2 in the help text), per-store cached calculations, footer credits (Kaggle/Favorita, holidays, Open-Meteo CC BY 4.0).
   6e Minimal redesign (2026-10-24, my request: simple, clean, less text) — app/app.py only, app/logic.py and all numbers unchanged.
      Layout: filters in one top row (store, shelf life, categories; empty = all) → 4 number cards → tabs Forecast | Reorder | India planning | About; no sidebar. Long captions cut to one short note each (kept: "stock is simulated", "rule-based, not model output", test v1 vs v2 in About).
      Fonts (.streamlit/config.toml, Google Fonts): Inter (text), Fraunces (headings, big numbers), JetBrains Mono (small labels). Category names shown as "Grocery II" (the CSV download is the same table as shown).
      Trending is now a ▲/▼ list; reorder status shown as 🔴 Urgent / 🟡 Reorder / 🟢 OK. Checked: AppTest 6 scenarios 0 exceptions, screenshots at 1440 px and 390 px (phone).
   6f New dashboard (2026-10-27, my request: professional, clean, presentable; plan in DASHBOARD_PLAN.md) — CURRENT UI
      Theme: navy header band, light cool-grey page (#f5f7fa), white cards, Inter + Plus Jakarta Sans; chart colours unchanged (validated palette); toolbar "viewer".
      Layout: header → filters (store, shelf life, categories) → alert strip (urgent count, next India festival, weather) → 4 cards → 6 tabs:
      Overview (30 → 14 day chart, top sellers this week, moving this week, order now, perishable vs long-life) | Category (who comes on top + rank change, category chart, trust label, "why") | Seasons & festivals (planning date, season filter, city, festivals, weather, month pattern, category-year search) | Reorder (table, urgent rows red, boost toggle, "Enter real stock", CSV) | All stores (click a row → opens store) | Model (back-test / unseen test / Christmas / range cards, test-days chart, error by category).
      New pipeline step src/explain.py (after forecast) → reports/forecast_why.parquet: LightGBM pred_contrib of v2, grouped into 7 reasons; effect = next 14 days vs model's view of last 14 days. Check printed: contributions add up to the saved forecast (largest gap 0.000000 units).
      build_app_data.py adds monthly_pattern.csv (Aug 2016–Jul 2017 only: earlier years have recording gaps, e.g. PRODUCE ~200 vs ~110,000 units/day; BOOKS left out) and test_daily.parquet (test days, all stores). App data still 0.7 MB.
      Trust label (category_errors.csv, v1 unseen test): Reliable < 20%, Rough guide < 50%, else Low confidence (also < 1 unit/day per store). Tiny sellers skipped in "Order now" preview and month ranking.
      "Enter real stock" re-runs the src/reorder.py rule for typed rows only (others keep saved values). Runs-out date = 15 Aug + floor(days_of_cover) + 1.
      Tests: 12 passed (+ typed stock = reorder rule; new app files complete, month pattern ends before test). AppTest 9 scenarios 0 exceptions; screenshots 1440 px and 390 px.
      FOUND: the 10% range model was far too low for big sellers (e.g. BEVERAGES store 44: low 790 vs forecast 8,903–15,800/day) → fixed in 6g.
   6g Range fix + remaining dashboard items (2026-10-27, my OK: "make the changes") — CURRENT STATE
      Diagnosis (back-tests, by point-forecast size): big sellers (≥ 1,000 units/day) had 0.3% below the low end and 15.4% above the high end (ideal 10 / 10); overall coverage 82.8% hid it.
      Fix: range models learn log1p(sales) − log1p(mean_28) (relative to the recent level; mean_28 is 14+ days old) = RANGE_TARGET "relative" in src/model.py ("level" = old, still reproducible); saved in lgbm_final_v2_info.json (range_target).
      Rule stated before the run: accept if mean coverage within 80 ± 3 and big sellers' below/above both closer to 10. python3 src/backtest.py --ranges-experiment (reports/backtest_ranges_experiment.csv), means over 6 windows:
        all: level 82.8 cov (7.3 below / 9.9 above) | relative 80.0 (10.4 / 9.7); 1000+: level 84.4 (0.3 / 15.4) | relative 77.7 (11.8 / 10.5); 100–1000: level 80.1 (9.1 / 10.8) | relative 77.8 (11.8 / 10.3). → accepted. Weak spot: window 2017-04-10, 1000+ below low 23.5%.
      After python3 run_all.py --retrain: v2 point model file, forecast, back-test WAPEs and Christmas results identical to before. Back-test range coverage 80.0% (77.8–81.3; perishable 78.5, long-life 80.6); test 78.2% (was 83.6%, test already seen). Range clipped to include the point forecast: 1,071 of 24,206 rows (4.4%, was 0.6%). Range total per row summed 9,599,139 to 13,563,677.
      Reorder now: Urgent 206 / Reorder 1,253 / OK 270 (50 status changes: 17 OK→Reorder, 33 Reorder→OK); total buffer 1,288,595 → 1,808,454 units, total order 2,305,658 → 2,660,234 (stock simulated); 590 rows capped at 50%.
      python3 run_all.py (no retrain) run twice → results.json, reorder plan, forecast, forecast_why identical.
      Dashboard additions: Ecuador holiday shading (app/data/holidays_window.parquet, 01_cleaning rules; store 44 = 11 Aug), extra-promotion dots in the category chart (days above the 14-day median of planned onpromotion), festival timeline (stock-by → festival day, next 60 days), safety stock Low / Normal / High (buffer × 0.5 / 1 / 1.5; Normal = agreed rule), printable HTML order sheet. Heatmap removed (replaced by the ranking).
      Tests: 13 passed (+ safety level, holiday rules).
   6h UI polish + click tests (2026-10-27) — app/app.py only, numbers unchanged
      Header shows the selected store; white filter bar and chart panels; tab icons; coloured ▲▼ deltas; urgent card tinted red; Overview lists 5 rows each; Reorder tab title; chart right margin.
      Real-browser click tests (headless Chrome): typed stock 0 for Automotive → Urgent 2 → 3, To reorder 21 → 20, kept after closing the popover and switching tabs; Download menu has CSV + printable sheet; All stores row click → store 3 (487,195 units, 7 urgent). 0 exceptions. AppTest 15 scenarios 0 exceptions; 13 tests passed.
      DASHBOARD_GUIDE.md = presenter notes: every feature, why it exists, demo path, likely questions.
   6i Accuracy experiments (2026-10-27, my OK: tuning + start year, then holiday features, "even more") — src/tune.py
      RULES WRITTEN BEFORE THE RUN. Locked test days never used. #5 (days 1–3 models) NOT done: needs relaxing the 14-day rule (my decision).
      Tuning windows (new, never used): 4 × 14 days in 2016 starting 2016-02-01 (Carnaval), 06-06, 08-01 (moved independence holiday), 10-31 (Difuntos + Cuenca + bridge); always trained from 2015-01-01.
      Steps, each starting from the model kept by the step before (start = v2):
        1 settings: step-by-step search on the tuning windows only — num_leaves {63,127,255} → Tweedie power {1.1,1.2,1.3,1.5} → min_child_samples {20,50,200} → colsample_bytree {0.5,0.8}; trees checked at 140/300/500/800/1200/1600/2000. Lowest mean tuning WAPE wins, then it must pass RULE on the back-tests.
        2 start year: 2014-01-01 or 2016-01-01 instead of 2015-01-01 (back-tests only); must pass RULE; if both pass, lower mean wins.
        3 holiday features (national holiday kind: holiday / moved / bridge / extra shopping day, long weekend, days to / since a real day off, name of the nearest holiday): RULE + tuning mean not worse.
        4 promotion feature (planned promotions minus the recent 28-day level, 14+ days old): RULE + tuning mean not worse.
        5 average of 3 seeds (geometric mean, so the "why" parts still add up): RULE + tuning mean not worse.
      RULE: back-test mean WAPE improves by ≥ 0.3 points AND at most 1 of the 6 windows gets worse.
      After the decisions: test WAPE reported for v1 / v2 / new side by side (test already seen, never used to decide); 80% range coverage on back-tests must stay within 80 ± 3, else report and ask.
      RESULT (python3 src/tune.py, about 6 h; reports/tune_settings.csv, tune_steps.csv, tune_decisions.json): NOTHING KEPT → v2 stays (back-test 13.72, tuning windows 14.73). No new model, so no new test score.
        1 settings: search picked 255 leaves, power 1.3, min_child 50, colsample 0.8, 2000 trees (edge of the grid), tuning 13.82 → back-tests 13.98 (−0.25, 2 worse; better in 4 windows but January 2017 20.15 vs 15.29) → rejected.
        2 start year: 2014 → 13.48 (+0.24, 3 worse), 2016 → 13.63 (+0.09, 3 worse) → rejected.
        3 holiday: 13.81 (−0.08, 2 worse), tuning 15.03 → rejected. Helped Easter 2017-04-10 (17.59 → 16.25), Carnaval, Aug 2016 (13.00 → 12.35); hurt Jan 2017 (17.18) and Nov 2016 (18.14).
        4 promo: 14.12 (−0.39, 3 worse), tuning 14.52 → rejected.
        5 seeds: 13.55 (+0.17, 1 worse), tuning 14.68 → rejected (closest).
      Pipeline left on v2: python3 run_all.py --retrain → model file, forecast, why, plan, back-tests, Christmas identical (SHA-256). Kept: src/tune.py, features.add_v3_features (unused by v2), model.load_holidays + window(train_start); results.json "experiments_6i"; Model tab line "More tries, none kept". Tests: 15 passed (+ holiday features on known days, promo level 14+ days old). CHECKSUMS 21 OK.
      Lesson: the 6 back-test windows differ a lot (Jan 2017 is unstable); small gains don't hold everywhere. Remaining big lever = #5 short-horizon models (needs my decision on the 14-day rule).
7. Finish & present — IN PROGRESS (2026-10-30: GitHub repo + CI/CD files ready, not pushed yet). TODO: deploy on Streamlit Community Cloud, README, report, slides, 2-minute demo.

## For the report
(all numbers: reports/results.json)
- Headline unseen score: v1 test WAPE 15.51% vs baseline 21.98% (29.4% better) on the locked 14 days.
- Fine-tuning judged on 6 back-test windows only: v2 13.72% vs v1 15.32% vs baseline 25.31%. v2 test 13.66% was measured after the test had been seen (37.9% better than baseline) — say so.
- v2 is better for top sellers (13.1 vs 15.1) but worse for low sellers (57.3 vs 52.2).
- Christmas back-test v2 17.8% (units −2.2%) vs v1 20.1% (−11%): same-day-last-year helps festival rushes.
- 6i: 5 more improvement tries (bigger model, start year, holiday features, promotion feature, 3 seeds) all failed the pre-set rule → v2 stays. Holiday features helped holiday windows but hurt others.
- 80% prediction ranges cover 80.0% of actual values in back-tests. The first version covered 82.8% overall but was badly off for big sellers (0.3% below low / 15.4% above high, ideal 10/10); learning the range relative to the recent level fixed it (11.8 / 10.5), judged on back-tests only.
- Valid→test gap (12.7% → 15.5%) mainly 12–13 Aug, the weekend after the 11 Aug (transferred) national holiday.
- Perishable 12.0% vs long-life 17.2%; low sellers 52.2%, top sellers 15.1% → volume matters more than shelf life.
- Christmas 2016 back-test: WAPE 20.1% vs 27.9% baseline; total units −11%; FROZEN FOODS rush only about half captured.
- Christmas settings (140 trees, no weather) were chosen with July 2017 validation = small look-ahead (limitation).
- No bias correction (test +4.9% over vs Christmas −11% under; over-forecast is the safer error for stock-outs).

## Future work
- (done in 6d: days_since_holiday, judged on back-tests only.)
- A2 short-horizon models (days 1–3 using recent sales): skipped — breaks the 14-day rule and too complex for this project.
- A5 promotion features (promotion vs normal level per category): skipped.
- A6 hyper-parameter tuning / averaging several models: skipped.
- A8 Kaggle submission for 2017-08-16..31 as a fresh unseen score: optional after Stage 7; needs a 16-day horizon (our model is 14 days) and Kaggle scores RMSLE, not WAPE.

## Honest limits (keep in report)
Ecuador data, not Indian; no stock data (stock simulated); category level, not item level; sales ≠ demand when out of stock; weather did not help at category level; India season/festival mapping is rule-based; future promotions = planned promotions from test.csv (0 where no plan); v2's test score was measured after the test had been seen.
