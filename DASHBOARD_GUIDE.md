# Dashboard guide — what each feature does and why

Presenter notes for the Dark Store Planner dashboard (`streamlit run app/app.py`).
Every number here comes from the app or `reports/results.json`. Examples use **store 44 (Quito)**, the default store.
Festival and weather examples depend on the real date; the ones below are from **27 Oct 2026**.

---

## 1. The idea in one line

> A dark store manager opens this page and in **10 seconds** knows: what will sell in the next 14 days,
> what runs out first, and what to order today.

**Who it is for** (this decided the design):

| User | Their question | Where they look |
|---|---|---|
| Store manager | "What do I order today?" | Alert strip, Overview → *Order now*, Reorder tab |
| City / ops manager | "Which stores are in trouble?" | All stores tab |
| Planner | "Diwali is coming — what extra stock?" | Seasons & festivals tab |
| Examiner / recruiter | "Is the model good and honest?" | Model tab |

---

## 2. Top of the page (always visible)

### Header (navy band)
- **What:** title, one-line purpose, and three chips: the selected store, *Data today · 15 Aug 2017*,
  *Forecast · 16 – 29 Aug 2017*.
- **Why:** the dataset ends on 15 Aug 2017. Showing the data's "today" up front stops anyone thinking the
  forecast is for this week. The store chip shows which store you are looking at.

### Filters (one white bar)
- **What:** store dropdown, shelf life (All / Perishable / Long-life), categories (empty = all 33).
- **Why:** one row, above everything, so every chart and number below follows the same filters.
  The shelf-life filter is project Extra 2 (perishable vs long-life).

### Alert strip
- **What:** short coloured messages, only when they matter:
  🔴 *"2 categories run out within 2 days in store 44"* · 🪔 *"Karwa Chauth in 2 days · stock up now"* ·
  🌧 a rain / cold / heat alert if the live weather triggers one.
- **Why:** the most urgent facts first, before any chart. A manager should not have to search for a problem.

### 4 number cards
| Card | Store 44 example | Why it is there |
|---|---|---|
| Next 14 days | 603,921 units, ▲ +5.9% vs last 14 days | The size of the job, and whether demand is going up or down |
| Urgent | 2 (red card) | Categories that run out within 2 days, the main action |
| To reorder | 21 | How many categories need an order |
| Model error | 13.7% vs 25.3% simple average | Trust: how often the model is wrong, next to a simple method |

---

## 3. The 6 tabs

### Tab 1 · Overview: "what is happening right now?"
| Feature | What it shows | Why |
|---|---|---|
| **Last 30 days → next 14 days** chart | Solid = actual sales, dashed = forecast (blue shaded area) | Joins the past to the future in one picture, so the forecast is believable |
| Likely range (pick ≤ 4 categories) | 80% band around each forecast | A forecast is never exact. The band shows how sure we are |
| **Amber columns** | Ecuador public holidays (store 44: 11 Aug) | Explains spikes and dips; the weekend after 11 Aug is where the model was weakest |
| **Top sellers this week** | Top 5 categories, units and share | "What sells most" at a glance |
| **Moving this week** | Biggest risers ▲ and fallers ▼ (last 7 days vs the 4 weeks before) | The "Trending now" goal: what is unusually hot or cold |
| **Order now** | The 5 categories that run out first, with run-out date and order quantity | The store manager's to-do list |
| **Perishable vs long-life** | Share of next-14-day units + back-test error for each | Extra 2: fresh food is 41% of units in store 44 and is *easier* to predict (12.1% vs 14.5% error) |

### Tab 2 · Category: "which category comes on top, and why?"
| Feature | What it shows | Why |
|---|---|---|
| **Who comes on top · next 14 days** | Top 10 categories by forecast, with ▲▼ places gained or lost | Your idea: Beverages moves from #3 to #1 in store 44 |
| **Search a category** | Type any category name | Quick jump to one category |
| 4 mini cards | Next 14 days, change vs last 14 days, rank #3 → #1, **trust label** | The trust label (🟢 Reliable / 🟡 Rough guide / ⚪ Low confidence) comes from that category's error on unseen test days, so people don't trust Books (420% error) like Produce (10.1%) |
| Category chart | Actual, forecast, 80% range, **orange dots** = days with more promotions planned than usual | Promotions push sales up, so the dots explain forecast peaks |
| **Why is it moving?** | Bars: how much each reason pushes the next 14 days up (blue) or down (red) | The model's own breakdown of every forecast (LightGBM *contributions*). Reasons: recent sales, same time last year, calendar, holidays, promotions, shoppers & oil, store & category. Store 44 Beverages: promotions +5.5%, calendar −3.3% |
| Status · runs out · order | Same numbers as the Reorder tab | Connects "why" to "what to do" |

### Tab 3 · Seasons & festivals: "what is coming up?" (India, rule-based)
Grey tag on the tab: **Rule-based for India · not model output.** This is on purpose: these rules are
assumptions, not machine learning.

| Feature | What it shows | Why |
|---|---|---|
| Planning date, season filter, city | Controls for this tab | The India side uses the **real date**, separate from the Ecuador forecast date |
| **Festival timeline · next 60 days** | Bar = when to start stocking up → ◆ festival day | Shows lead time at a glance: Diwali needs stock from 1 Nov |
| Upcoming festivals | Cards with date, "stock by" date, extra stock %, products (e.g. Diwali: sweets, gifts, toys, +30%) | Project Extra 1 (festival effect), for India. Dates come from the Python `holidays` library |
| Season card | Summer (Mar–May), Monsoon (Jun–Sep), Winter (Nov–Feb) products | The season filter hides out-of-season items |
| **Weather · next 7 days** | Live rain and temperature for 8 Indian cities (Open-Meteo) + alerts | Rain ≥ 5 mm, max ≤ 15 °C or ≥ 40 °C trigger "stock up" alerts |
| **What usually sells more in…** (month slider) | Categories that usually sell most above their average month | Your ideas 1 + 4. A **past pattern** (Aug 2016 – Jul 2017), *not* a forecast: the model can only see 14 days ahead |
| **Search a category's year** | 12-month chart for one category (blue = above average, red = below) | e.g. Frozen Foods +297% in December, School supplies +214% in April |

### Tab 4 · Reorder: "what do I order?"
| Feature | What it shows | Why |
|---|---|---|
| **Reorder table** | Status, order, stock, **runs out** date, days of stock, forecast, safety buffer, window | The main output of the project (Extra 3). Urgent rows are **red** |
| Formula (note under the table) | Order = forecast for the window + safety buffer − stock | Simple and explainable. Window: 3 days for fresh food, 14 days for long-life |
| **Safety stock Low / Normal / High** | Safety buffer × 0.5 / 1 / 1.5 | A what-if: a careful manager can keep more. *Normal* is the agreed rule |
| **India extra stock** toggle | Adds festival / season / weather extra % | Connects the India rules to the order. Off by default because it is an assumption |
| **✏️ Enter real stock** | Type the store's real stock; order and status update | The dataset has **no stock data**, so stock is simulated. A real user can replace it with real numbers. Same rule as the pipeline (tested) |
| **⬇ Download** | Table as CSV, or a **printable order sheet** with a "Done ✓" column | CSV for spreadsheets, the sheet for the warehouse floor |

### Tab 5 · All stores: "where do I look first?"
- **What:** all 54 stores, sorted by urgent categories (red bar), with reorder count and next-14-day units.
  **Click a row** and the whole dashboard switches to that store.
- **Why:** an ops manager handles many stores. Stores 3, 4 (Quito) and 23 (Ambato) share the most urgent items (7 each); store 3 is first because it also has the most to reorder (25).

### Tab 6 · Model: "can I trust it?"
| Card / chart | Value | What to say |
|---|---|---|
| Back-tests (6 × 14 days) | 13.7% vs 25.3% simple average | How the final model was chosen: judged only on past windows |
| Unseen test days | 15.5% (first model) vs 22.0% | The honest headline: these days were locked until the end |
| Christmas 2016 | 17.8% vs 27.9% | Festival rush: the hardest case |
| 80% range | 80.0% of actual values inside | The range means what it says |
| Test days: actual vs forecast | Black = actual, blue = model, grey = simple average | Shows the model follows the weekly ups and downs; the simple average is flat |
| Error by category | Easiest 12 categories, with trust icons | Volume matters: big categories are easier |

The note under the chart says the current model (v2) scored 13.7% on the test days **only after they had been
seen**. Saying this openly shows the results are honest.

---

## 4. Design choices and why

| Choice | Why |
|---|---|
| **Light page, navy header** | Light backgrounds stay readable on projectors; the navy band gives a professional, branded look |
| **White cards and panels** on a light grey page | Groups each idea into one box. The eye reads it panel by panel |
| **Few colours, each with one meaning** | Blue = forecast / up, red = urgent / down, amber = holiday / festival, green = OK. Status always has an icon + word too (🔴 Urgent), so it works for colour-blind viewers |
| Chart colours from a **colour-blind-safe palette** | Checked with a validator, not by eye |
| **Inter + Plus Jakarta Sans** fonts, tabular numbers | Clean and modern; numbers line up in columns |
| **Units only, never money** | The data is units sold. No ₹ or $ anywhere |
| **Two dates, never mixed** | Forecast side = 15 Aug 2017 (Ecuador data); India side = real today. Labelled everywhere |
| **Three kinds of numbers, always labelled** | *Model forecast*, *rule-based (India)*, *simulated (stock)*. Nobody mistakes an assumption for a prediction |
| Short notes under charts, details on hover | Less text on screen; the detail is there when needed |
| Works on a phone | Columns stack, cards go 2 × 2 (checked at 390 px width) |

---

## 5. A 2-minute demo path

1. **Header + alert strip** (15 s): "Store 44, data ends 15 Aug 2017. Two categories run out within 2 days, and
   Karwa Chauth is in 2 days."
2. **Cards** (15 s): "604 thousand units next 14 days, up 5.9%. Our model's error is 13.7%, the simple method's 25.3%."
3. **Overview chart** (20 s): "Solid is real, dashed is forecast. The amber column is a public holiday."
4. **Category → Beverages** (25 s): "Beverages moves to #1. *Why?* Promotions push it up 5.5%."
5. **Reorder** (20 s): "Red rows first. Stock is simulated, but I can type the real number" (type 0 → urgent goes up).
6. **Seasons & festivals** (15 s): "Diwali timeline: start stocking 1 Nov. These are rules, not the model."
7. **Model** (10 s): "Honest scores: 15.5% on locked test days, 13.7% on back-tests."

---

## 6. Questions you may get

| Question | Short answer |
|---|---|
| Why Ecuador data for India? | No public Indian dark-store sales data. Favorita (Ecuador) has the same shape: many stores × categories × days. The India side is rule-based and labelled. |
| Is the stock real? | No, the dataset has no stock, so it is simulated. That is why "Enter real stock" exists. |
| Why only 14 days ahead? | Every input is at least 14 days old, so the 14 days can be forecast at once without peeking at the future. Further ahead we show past patterns, labelled "not a forecast". |
| Why does Books show 420% error? | It sells under 1 unit/day per store, so tiny mistakes are huge percentages. It gets a ⚪ "Low confidence" label. |
| How do you know the "why" bars are right? | They are the model's own breakdown. The parts add up exactly to the forecast (checked in code: largest gap 0.000000 units). |
| Did you try to make it more accurate? | Yes, 5 more ideas: a bigger model, another training start year, holiday features, a promotion feature, an average of 3 models. Rules were written before the run; none passed, so v2 stays (closest: 13.55% vs 13.72%). The Model tab says this under "How it works". |
| Did weather help the model? | No: validation 12.82% with weather vs 12.74% without. So weather is used as a rule-based alert, not a model input. |
| What does the range mean? | 80% of actual sales should fall inside it. In back-tests: 80.0%. |

---

## 7. Limits (say them before someone else does)

Ecuador data, not Indian · stock is simulated · category level, not single items · sales ≠ demand when an item
is out of stock · India festival, season and weather rules are assumptions · the current model's test score was
measured after the test days had been seen (the unseen score is 15.5%).
