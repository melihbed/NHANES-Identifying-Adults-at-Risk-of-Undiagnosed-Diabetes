# Dashboard Build Spec — Tableau / Power BI

Target: a 3-page workbook that lets a non-technical stakeholder (a) see who carries
undiagnosed-diabetes risk, (b) judge whether the model is trustworthy, and (c) choose
an operating point.

**Data source:** `dashboard/data/*.csv` (rebuild with `python -m src.dashboard`).
**Model:** relate `dim_*` to `fact_participants` on the code columns (or just use the
pre-labelled text columns already in the fact table). `agg_*` tables are standalone.

Palette: one neutral grey for "below range / not flagged", one saturated accent (amber →
red) for risk. Diverging only on the calibration page. Keep it colour-blind safe.

---

## Page 1 — "Who is at risk?"

**KPI row (from `fact_participants`)**
| Tile | Field / calc |
|---|---|
| Adults screened | `COUNTD(SEQN)` → 4,698 |
| Undiagnosed rate (unweighted) | `AVG(is_diabetes_range)` → 2.5 % |
| Undiagnosed rate (population-weighted) | `SUM(survey_weight*is_diabetes_range)/SUM(survey_weight)` → 2.0 % |
| Cases in cohort | `SUM(is_diabetes_range)` → 116 |

**Chart A — prevalence gradient (bar).** Source `agg_prevalence_by_group.csv`.
Rows = `category`, columns = `prevalence`, small-multiple / filter by `dimension`.
Sort descending. Reference line at the 2.5 % cohort mean. Tooltip: `participants`,
`positives`, `weighted_prevalence`.
*Story it tells:* waist ≥ 110 cm 5.8 % · BMI ≥ 35 5.4 % · < high-school education 6.2 % ·
Non-Hispanic Black 5.0 % · age 65+ 3.6 %.

**Chart B — age × waist heatmap.** Source `fact_participants`. Rows `age_band`,
cols `waist_band`, colour = `AVG(is_diabetes_range)`, label = n. Shows risk rising
jointly with both.

**Filters:** `sex`, `race_ethnicity`, `split` (default: all).

---

## Page 2 — "Is the model any good?"

**KPI row** (hard-code from `reports/model_metrics.json` or a small CSV): ROC-AUC 0.84 ·
PR-AUC 0.074 vs 0.024 no-skill · Brier 0.024 · Test recall 0.96.

**Chart C — calibration.** Source `agg_calibration_bins.csv`. `mean_predicted_risk` (x)
vs `observed_prevalence` (y), line + points; add a 45° `y = x` reference. On-target =
points hug the diagonal.

**Chart D — score distribution by outcome.** Source `fact_participants`, filter
`split = "test"`. Histogram / box of `risk_score` split by `outcome_label`. Vertical line
at 0.0235. Shows the classes are separated but overlapping.

**Chart E — confusion matrix @ 0.0235.** Source `fact_participants`, `split = "test"`.
2×2 of `flagged` × `is_diabetes_range`, label counts (22 / 284 / 1 / 633).

---

## Page 3 — "Choosing the operating point"

**Parameter:** `Threshold` (0.010–0.060, step 0.0025).

**Chart F — recall vs precision vs flag-rate.** Source `agg_threshold_sweep.csv`.
x = `threshold`; three lines: `recall_sensitivity`, `precision_ppv`, `flag_rate`.
Vertical rule at the `Threshold` parameter. Callout box:
`"At {Threshold}: catch {recall}% of cases, flag {flag_rate}% of adults, {false_positives} false alarms."`

**Chart G — workload trade-off (dual axis).** x = `threshold`, bar = `true_positives`,
line = `false_positives`. Makes the "cost of extra recall" visible.

**Text panel:** the three named operating points (70 / 80 / 90 % recall) from the model
card, and the reminder that calibrated scores mean the dial moves without retraining.

---

## Notes for whoever builds it

- `risk_score` for `train` rows is cross-validated (out-of-fold), so mixing train+test in
  aggregates is legitimate — every participant has an honest score.
- Use `survey_weight` whenever the headline is "U.S. adults"; state "in this sample"
  otherwise.
- `bmi*` columns are included for context only — the model does **not** use BMI.
- Don't present precision without context: low PPV is expected and acceptable for a
  first-stage screen backed by a confirmatory lab test.
