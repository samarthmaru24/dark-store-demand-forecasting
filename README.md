# Dark Store Demand Forecasting

Quick-commerce companies (Blinkit, Zepto, Instamart) run small "dark stores". Running out of an item is a bad
customer experience; overstocking is waste. This project predicts **units sold per store × product category for
the next 14 days**, turns the forecast into a **reorder list**, and shows it in a **Streamlit dashboard** with
India festival, season and weather alerts.

> All quantities are **units sold**, not money. Every number below comes from `reports/results.json`,
> which is written by code (`python3 run_all.py`).

## Results

The model is LightGBM. Every past-sales input is at least 14 days old, so the whole 14-day forecast can be made
on day 0 with no leakage. Error is measured as **WAPE** (total units wrong ÷ total units sold; lower is better).

| | Error (WAPE) | Simple baseline (14-day average) |
|---|---|---|
| **Back-tests**: 6 windows of 14 days in 2017, before the test period (v2) | **13.72%** (min 10.63, max 17.59) | 25.31% |
| **Locked test days** 2017-08-02 to 2017-08-15, **v1** (first, unseen score) | **15.51%** (29.4% better than baseline) | 21.98% |
| Locked test days, **v2** (scored after the test had been seen) | 13.66% | 21.98% |
| Christmas back-test 2016-12-12 to 2016-12-24 (v1 / v2) | 20.1% / 17.8% | 27.9% |

- **How v2 was chosen (honestly):** v2 adds *days since the last holiday* and *same weekday last year*, and uses a
  Tweedie objective. It was compared with v1 **only on the 6 back-test windows** (mean 13.72% vs 15.32%;
  perishables 12.05% vs 14.72%) and kept because it passed the agreed rule (≥ 0.3 points better and perishables
  not worse). The test days had already been seen in Stage 5, so v1's **15.51%** stays the headline unseen score.
- **Where it works best:** high-volume, perishable categories (test, v2: perishable 10.6%, long-life 15.1%,
  top sellers 13.1%). **Where it struggles:** low sellers (57.3%), the weekend after a public holiday, festival rushes.
- **Prediction ranges:** 10% and 90% quantile models give an 80% range. It covered 82.8% of actual values in
  the back-tests (target 80%).
- **Weather did not help** at category level (validation 12.82% with weather vs 12.74% without), so the model
  does not use it. Weather is used in the dashboard as a rule-based alert instead.

## Dashboard

```bash
streamlit run app/app.py
```

- Store and category filters, KPI row, **last 30 days actual + 14-day forecast** (with 80% range per category)
- **Trending now** (last 7 days vs previous 28), forecast heatmap (category × day)
- **Reorder table**: order = forecast over window + safety buffer − stock (perishables 3 days, long-life 14 days;
  buffer from the 90% range), colour-coded Urgent / Reorder / OK, CSV download
- **India planning panel (rule-based, not model output):** upcoming festivals with "start stocking by" dates,
  current season, and live 7-day **rain / cold / heat alerts** for 8 Indian cities
- Optional "India rule boosts" to the order quantities (clearly labelled as an assumption)

The app reads only the small files in `app/data/` (0.7 MB). It never runs the model.

## How to run

```bash
pip install -r requirements.txt          # tested on Python 3.14 and 3.12
python3 run_all.py --retrain             # train v2 + range models, forecast, reorder, back-tests, results, app data
python3 run_all.py                       # same, using the saved models
python3 -m pytest tests -q               # 10 checks: 14-day rule, splits, outputs, reorder maths, rules
streamlit run app/app.py
```

`raw/` and `clean/` are not in the repo (see `.gitignore`). Download the data into `raw/` and run
`notebooks/01_cleaning.ipynb` first. Generated notebooks and scripts are verified with `CHECKSUMS.txt`
(`shasum -a 256 -c CHECKSUMS.txt`).

## Project structure

```
notebooks/   01_cleaning, 02_eda, 03_model, 04_evaluation (+ *_run = executed copies)
src/         features.py (shared features), model.py (settings), train.py, backtest.py, forecast.py,
             reorder.py, results.py, build_app_data.py
app/         app.py (Streamlit UI), logic.py (calculations), data/ (precomputed files)
data_rules/  conditions_rules.csv (India festival / season / weather rules), festival_name_map.csv
models/      lgbm_v1 (Stage 4), lgbm_final (v1), lgbm_final_v2, lgbm_q10 / lgbm_q90 (ranges)
reports/     charts q1–q12, results.json, back-test and test outputs, forecast and reorder plan
tests/       test_pipeline.py
run_all.py   whole pipeline in one command
```

## Honest limits

- **Ecuador data stands in for India** (Favorita supermarkets, 2013–2017); India rules are assumptions.
- **Stock is simulated** (the dataset has no stock levels) and labelled SIMULATED everywhere.
- **Category level, not item level.** Sales ≠ demand when a shelf was empty.
- **Future promotions** come from the dataset's planned promotions (`test.csv`), 0 where no plan.
- v2's test score was measured after the test period had been seen (see above).

## Data and credits

- **Store Sales – Time Series Forecasting** (Corporación Favorita, via Kaggle):
  https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
- **Online Retail II** (UCI Machine Learning Repository, CC BY 4.0): https://archive.ics.uci.edu/dataset/502/online+retail+ii
  (downloaded for optional gift/winter analysis; not used by the model)
- **[Weather data by Open-Meteo.com](https://open-meteo.com/)** (CC BY 4.0): historical weather for the model
  experiment and the live forecast in the dashboard
- **Indian festival dates:** Python [`holidays`](https://pypi.org/project/holidays/) library
