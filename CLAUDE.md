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
- notebooks/ 01_cleaning, 02_eda, 03_model (+ 03_model_run = executed copy)
- models/ lgbm_v1.txt, lgbm_v1_info.json
- reports/ charts q1–q6 and q8 (no q7), festival_effect.csv
- src/, app/ — empty so far (except .gitkeep)
- make_03_model.py (generator for 03_model.ipynb), make_stage5.py (generator for Stage 5, not yet run)
- CHECKSUMS.txt — SHA-256 of every generator script and the files it creates
- .gitignore excludes raw/ and clean/ (data never goes to GitHub). Local git repo initialised here; GitHub remote is connected in Stage 7.
Clean table columns: date, store_nbr, family, sales, onpromotion, is_closed, city, state, store_type, cluster, is_event, is_national_holiday, is_regional_holiday, is_local_holiday, is_holiday, dcoilwtico, transactions, is_payday, is_quake, is_perishable, never_sold.
"sales" = UNITS sold (not money). Never write $, ₹ or "revenue".

## HARD RULES (always)
1. Never modify, move or delete anything in raw/. Only exception: raw/weather_ecuador/weather.csv may be (re)downloaded by code. No other code may write to raw/.
2. The 14-day rule: any past sales / oil / transactions feature must be at least 14 days old. No same-day oil or transactions. Split by time, never shuffle.
3. Test period 2017-08-02 to 2017-08-15 is locked — used only in Stage 5, once.
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
   6a Forecast engine — DONE: src/forecast.py (python3 src/forecast.py) → reports/forecast_next14.csv, 2017-08-16..2017-08-29, 24,206 rows (54 stores × 33 categories minus never_sold). Reuses src/features.py unchanged; holiday flags rebuilt with the 01_cleaning rules (verified identical on history); adds 60 calendar-only days so days_to_holiday sees the 9 Oct holiday (54 → 41).
      With planned promotions (decided in 6b): total forecast 11,130,075 units vs 11,444,795 in the last 14 days (−2.7%). (Earlier, with promotions = 0: 10,765,373, −5.9%.)
      Edge effect found: in the original features, days_to_holiday on 2017-08-12..15 was 60 (data ended) instead of 55–58; only those 4 days, tiny effect on lgbm_final.
   6b Bias check + reorder plan + India rules — DONE
      Bias check (reports/test_predictions.csv): overall +4.88% (OVER-forecast), long-life +4.71%, perishable +5.24%.
      DECISION: NO bias correction. Test was +4.9% (over) but the Christmas back-test was −11% (under), so a single factor isn't reliable; slight over-forecast is the safer error for stock-outs.
      DECISION: no new model features (e.g. days_since_holiday) — the test set has been seen, so changing the model now would leak test information.
      src/reorder.py → reports/reorder_plan.csv (1,729 rows). Window perishable 3 / long-life 14 days; buffer = forecast_window × test WAPE capped 50% (capped: HARDWARE, SCHOOL AND OFFICE SUPPLIES); fixed 1-unit buffer for < 1 unit/day per store (BABY CARE, BOOKS, HOME APPLIANCES); order_qty rounded up.
      stock_SIMULATED = recent 14-day avg daily sales × U(0.3, 1.5) × window_days, seed 42. days_of_cover = stock / avg daily forecast (rounded to 0.1 before the status, so shown number and status agree; empty if forecast is 0).
      Status: "Urgent" if days_of_cover < 2, else "Reorder" if order_qty > 0, else "OK".
      Result: Urgent 257 / Reorder 995 / OK 477 (long-life 111 / 843 / 289; perishable 146 / 152 / 188).
      Quirk: tiny sellers whose simulated stock rounds to 0 (e.g. BOOKS, HOME APPLIANCES in store 44) show as Urgent with 0.0 days of cover.
      data_rules/conditions_rules.csv (my draft, rule-based assumptions) + data_rules/festival_name_map.csv (rule name → `holidays` library name, all 8 found; Karwa Chauth = "Karaka Chaturthi (Karwa Chouth)", Diwali = "Diwali (Deepavali)", Ganesh Chaturthi = "Ganesh Chaturthi / Vinayak Chaturthi"; Id-ul-Fitr 2027 is "(estimated)"). All related_categories match the 33 families. Library: holidays 0.107.
   6c Streamlit dashboard — DONE: app/app.py (UI) + app/logic.py (all calculations, testable without Streamlit).
      Run from the project root: streamlit run app/app.py. Theme in .streamlit/config.toml. requirements.txt pinned (Python 3.14.6 locally).
      Reads ONLY app/data/ (1.2 MB, tracked in git; rebuilt by python3 src/build_app_data.py): history_last60.parquet (2017-06-17..2017-08-15), stores.csv, forecast_next14.csv, reorder_plan.csv, category_errors.csv, conditions_rules.csv, festival_name_map.csv, eval_summary.json. Never runs the model.
      Pipeline order: python3 src/forecast.py → python3 src/reorder.py → python3 src/build_app_data.py.
      Boosts: boosted_order = order + forecast_window × extra_stock_pct for active rules (festivals whose "start stocking by" date has arrived, the current season, rainy-day if the rain check found ≥ 5 mm). If several rules match a family, the largest % is used (no stacking). Season filter keeps only that season's rules.
      Seasons from planning-date month: Summer Mar–May, Monsoon Jun–Sep (wins June), Winter Nov–Feb; October has no season rule.
      Rain: Open-Meteo forecast API (precipitation_sum, 7 days, real today); friendly message if it fails.
      Trending: last 7 days vs previous 28 days average units/day, open days only, categories under 1 unit/day skipped. Heatmap uses a log colour scale (hover = exact units).
      Smoke test (2026-10-24): store 44 next 14 days 578,586 units (+1.5% vs last 14 days), 4 Urgent, 19 Reorder; 0 exceptions in AppTest runs and server log.
   DASHBOARD DATE DESIGN: the forecast/reorder side uses the data's "today" = 2017-08-15 (Ecuador demo). The India festival/season panel uses an "India planning date" picker (default = real today), clearly labelled as a separate, rule-based panel.
   - Forecast the 14 days after 2017-08-15 with models/lgbm_final.txt + src/features.py (build future rows; onpromotion = planned promotions from test.csv, 0 where no plan — say so).
   - order = forecast over window + safety buffer − current stock, minimum 0. Window: perishables 2–3 days, long-life 14 days. Buffer from each category's test WAPE (reports/category_errors.csv). Current stock is SIMULATED (dataset has none) and labelled so.
   - conditions_rules.csv (written together with me): India condition → product groups, with a season column. Examples: Diwali → sweets, chocolates, gifts, toys, decor; Holi → colours, sweets; Raksha Bandhan → rakhis, sweets, gifts; Monsoon (Jun–Sep) / rainy → umbrellas, raincoats, quick-dry clothes, mosquito repellent, tea, snacks; Winter (Nov–Feb) → hoodies, sweatshirts, moisturiser; Summer (Mar–Jun) → cold drinks, ice cream, curd, ORS, sunscreen. These are rule-based assumptions, not model output — label them so.
   - Indian festival dates: Python `holidays` library, holidays.country_holidays("IN", years=..., categories=("public", "optional")).
   - Build all dashboard features listed under "End goal".
7. Finish & present — TODO: deploy on Streamlit Community Cloud, README, report, slides, 2-minute demo.

## For the report
- Test WAPE 15.51% vs baseline 21.98% (29.4% better) on the locked 14 days.
- Valid→test gap (12.7% → 15.5%) mainly 12–13 Aug, the weekend after the 11 Aug (transferred) national holiday.
- Perishable 12.0% vs long-life 17.2%; low sellers 52.2%, top sellers 15.1% → volume matters more than shelf life.
- Christmas 2016 back-test: WAPE 20.1% vs 27.9% baseline; total units −11%; FROZEN FOODS rush only about half captured.
- Christmas settings (140 trees, no weather) were chosen with July 2017 validation = small look-ahead (limitation).
- No bias correction (test +4.9% over vs Christmas −11% under; over-forecast is the safer error for stock-outs).

## Future work
- days_since_holiday feature (would help the after-holiday weekend dip) — not added now because the test set has been seen.

## Honest limits (keep in report)
Ecuador data, not Indian; no stock data (stock simulated); category level, not item level; sales ≠ demand when out of stock; weather did not help at category level; India season/festival mapping is rule-based; future promotions assumed 0.
