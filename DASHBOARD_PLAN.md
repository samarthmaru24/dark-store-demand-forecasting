# Dashboard plan — thinking as the user

> Status 2026-10-27: built as the new dashboard (CLAUDE.md, stage 6f). Built: sections 2, 3 (tabs A–E), 7 (ideas 1–4), editable stock, trust labels, runs-out date. Not built: promotion markers, holiday shading, festival timeline, safety slider, printable sheet.

✅ = already in the app · 🆕 = new idea · 🔧 = exists but should change

---

## 1. Who will open this dashboard?

| User | What they want in 10 seconds | Main question |
|---|---|---|
| **Store manager** (one dark store) | "What do I order today, and what runs out first?" | *What do I do now?* |
| **City / ops manager** (many stores) | "Which stores are in trouble today?" | *Where do I look first?* |
| **Planner** (festivals, seasons) | "Diwali is coming — what extra stock, and when?" | *What is coming up?* |
| **Teacher / examiner / recruiter** | "Is the model any good? Is it honest?" | *Can I trust these numbers?* |

**Rule for the design:** the first screen answers the store manager. Everything else is one click away.

---

## 2. What the first screen must show (top to bottom)

1. **Header** ✅ — title, store dropdown, "today = 15 Aug 2017 (demo data)".
2. **Alert strip** 🆕 — one line, only when something matters:
   - 🔴 "4 categories run out in under 2 days" → click goes to the reorder list
   - 🪔 "Diwali in 12 days — start stocking sweets, gifts, toys" (rule-based)
   - 🌧 "Rain on Thu, Fri in Mumbai — stock up tea, snacks" (rule-based)
3. **4 number cards** ✅ — next 14 days units · urgent · to reorder · model error.
4. **"Order now" list** 🆕 — top 5 urgent rows right here (category, runs out on, order qty), not hidden in a tab.
5. **Forecast chart** ✅ — last 30 days (solid) → next 14 days (dashed) + 80% range band.

---

## 3. Tabs and what goes inside each

### Tab A — Forecast ✅
- ✅ Last 30 days → next 14 days chart, with range band (up to 4 categories)
- ✅ Trending this week (▲ / ▼ list)
- ✅ Perishable vs long-life cards
- ✅ Heatmap: category × day
- 🆕 **Promotion markers** on the chart (days with planned promotions — the forecast data already has `onpromotion`)
- 🆕 **Weekend / holiday shading** on the chart, so a spike has a visible reason
- 🆕 **"How sure are we?" label per category** from `category_errors.csv`:
  - 🟢 Reliable (error < 20%) · 🟡 Rough (20–50%) · ⚪ Too small to predict (tiny sellers like Books)

### Tab B — Reorder ✅
- ✅ Table: status, order, stock*, days of stock, forecast, buffer, window
- ✅ Festival/season/weather extra-stock toggle
- ✅ Download CSV
- 🔧 **"Runs out on" date** (e.g. *Thu 17 Aug*) next to "days of stock" — easier to read than 1.4
- 🆕 **Type your real stock** — the stock column is editable; order and status update at once.
  This fixes the biggest honest limit (stock is simulated) for a real user.
- 🆕 **Safety level slider** — "Low / Normal / High" buffer (changes how much extra to keep)
- 🆕 **Printable order sheet** — just category + order qty, for the warehouse

### Tab C — All stores 🆕 (for the ops manager)
- Table of all 54 stores: city, urgent count, reorder count, next-14-day units, change vs last 14 days
- Sort by "most urgent" by default; click a store → switches the store dropdown
- Small bar chart: top 10 stores by urgent count
- (Uses `reorder_plan.parquet` and `forecast_next14.parquet` that are already in `app/data/`.)

### Tab D — India planning ✅
- ✅ Planning date, season filter, city
- ✅ Upcoming festivals (date, days away, "stock by" date, products, +%)
- ✅ Season card
- ✅ 7-day weather (rain, temperature) + alerts
- 🆕 **Festival calendar strip** — next 60 days as a timeline with festival dots
- 🔧 Season filter should also **hide out-of-season categories** in the Reorder tab (goal in CLAUDE.md:
  "hides out-of-season items") — today it only hides the rules
- Keep the label: *"Rule-based suggestions for India, not model output."*

### Tab E — How good is the model? 🔧 (today: "About", text only)
- 🆕 **Past check chart**: the locked test days (2–15 Aug 2017) — actual vs predicted.
  Needs `test_predictions_v2.csv` copied into `app/data/` by `src/build_app_data.py`.
- 🆕 **Error by category** bar chart (from `category_errors.csv`, already in `app/data/`)
- 🆕 **Christmas back-test** one-line result + the existing chart image `q11`
- ✅ Honest notes: v1 15.51% is the unseen score; v2 13.66% was measured after the test was seen
- ✅ Limits: Ecuador data, simulated stock, category level, rule-based India panel

---

## 4. What it should look like (layout sketch)

```
┌──────────────────────────────────────────────────────────────────────┐
│ DARK STORE PLANNER · demo data · today = 15 Aug 2017                 │
│ Demand & reorder                                                     │
│ [Store 44 · Quito ▾]  [All | Perishable | Long-life]  [Categories ▾] │
├──────────────────────────────────────────────────────────────────────┤
│ 🔴 4 categories run out in < 2 days   🪔 <festival> in N days   🌧 Rain │  ← alert strip
├────────────┬────────────┬────────────┬───────────────────────────────┤
│ <units>    │ <n>        │ <n>        │ 13.7%                         │
│ units next │ urgent     │ to reorder │ model error (vs 25.3%)        │
│ 14 days    │            │            │                               │
├────────────┴────────────┴────────────┴───────────────────────────────┤
│ ORDER NOW                                    runs out    order       │
│ 🔴 <category>                                <date>      <qty>       │
│ 🔴 <category>                                <date>      <qty>       │
│ ...                                          [see all →]             │
├──────────────────────────────────────────────────────────────────────┤
│ [Forecast] [Reorder] [All stores] [India planning] [Model check]     │
│                                                                      │
│   units/day                     ┊ today                              │
│     ╱╲    ╱╲   ╱╲    ╱╲         ┊ ╌╌╱╲╌╌╱╲╌╌   ← dashed = forecast   │
│   ╱    ╲╱    ╲╱  ╲╱    ╲        ┊░░░░░░░░░░░   ← band = 80% range    │
│   ─── last 30 days ────────────┊─ next 14 ─                         │
│                                                                      │
│   Trending ▲▼        │  Perishable vs long-life                      │
└──────────────────────────────────────────────────────────────────────┘
```
`<…>` = filled in by the app.
On a phone: filters stack, 4 cards become 2 × 2, "Order now" stays on top, charts go full width.

---

## 5. Design rules (keep it clean)

- **One question per section.** If a section needs a paragraph to explain, cut it.
- **Colours mean one thing each:** red = urgent, yellow = reorder, green = OK, blue = forecast.
- **Three kinds of numbers, always labelled:**
  `Model forecast` · `Rule-based (India)` · `Simulated (stock)` — a small grey tag next to each.
- **Units only.** Never ₹, $ or "revenue".
- **Two dates, never mixed:** forecast side = 15 Aug 2017 (data), India side = planning date (real today).
- **Hover for detail**, not extra text on the page.
- **Same look as now:** Inter text, Fraunces headings, light background, rounded cards.

---

## 6. Build order (suggested)

| # | Item | Why first | Effort |
|---|---|---|---|
| 1 | Alert strip + "Order now" on the first screen | Answers the main user in 10 seconds | Small |
| 2 | "Runs out on" date + editable stock | Makes the reorder table real-world useful | Small |
| 3 | All stores tab | Second user type; data already there | Medium |
| 4 | Trust label per category | Honest; stops people trusting Books 420% | Small |
| 5 | Model check tab (charts) | For the report / examiner | Medium (adds 1 file to `app/data/`) |
| 6 | Season filter hides out-of-season categories | Finishes an original goal | Small |
| 7 | Promotion markers, holiday shading, festival timeline | Nice extras | Small each |
| 8 | Safety slider, printable order sheet | Nice extras | Small each |

---

## 7. Samarth's ideas (added after review)

Note: our data has **33 categories**, not single products. "Product" below = category.

### Idea 1 — Search: "what is on top in which season?" 🆕
- Search box: type a category (e.g. "Frozen foods") → 12-month bar chart of its usual pattern + its best months.
- Or pick a month / season → ranked list of categories that rise most in that month.
- Source: **historical pattern** 2013–2016 (Ecuador), each month vs the same year's average (removes yearly growth),
  only years where the category was selling. New small file `app/data/monthly_pattern.csv` made by
  `src/build_app_data.py`.
- Shown next to it: the **India rule** for that season (from `conditions_rules.csv`), labelled rule-based.
- Honest label: Ecuador is on the equator — no summer/monsoon/winter. Its peaks come from Christmas and
  school season, not weather.

### Idea 2 — "What is live now" 🆕 (🔧 grows the Trending list)
- "Now" panel: top 5 categories by units in the last 7 days + biggest risers (▲) and fallers (▼).
- Data "now" = 15 Aug 2017 (end of the dataset). There is **no live sales feed** — say so.
- Truly live parts (real today): India weather alerts and upcoming festivals.

### Idea 3 — One page per category: "will it be on top, and why?" 🆕
- Pick a category → its own chart: last 30 days + next 14 days + 80% range.
- **Rank change:** "#5 in the last 14 days → #3 in the next 14 days".
- **Why** (from the model itself): LightGBM can split every forecast into the effect of each input
  (`pred_contrib`). Grouped into plain reasons, e.g.:
  `recent sales level` · `day of week` · `same time last year` · `promotions planned` · `holiday nearby`.
  Shown as "pushes up / pushes down" bars. Computed once in the pipeline and saved to a small file
  in `app/data/` — the app still never runs the model.
- Also on the page: its trust label (🟢/🟡/⚪), stock, order, runs-out date.

### Idea 4 — "Which will be on top in a future season?" 🆕
- The model can only forecast **14 days ahead** (14-day rule). It cannot honestly predict next Diwali or next winter.
- So for future seasons the app shows: **"Usually on top in <month>"** from the historical pattern (Idea 1),
  plus the India rule for that season — clearly labelled *"past pattern, not a forecast"*.
- For the next 14 days it shows the real model ranking (Idea 3).

### Suggested new tab layout
`Now` (ideas 2 + order-now list) · `Category` (idea 3) · `Seasons` (ideas 1 + 4) · `Reorder` · `All stores` ·
`India planning` · `Model check`

---

**Not changing:** the model, `results.json` numbers, the reorder formula. All new items only read
existing files or let the user type in their own numbers.
