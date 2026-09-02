# Dashboard Data Dictionary

Extracts written by `python -m src.dashboard` to `dashboard/data/`. Star schema:
`fact_participants` is the grain (one row per cohort adult); `dim_*` are lookups;
`agg_*` are pre-aggregated for specific dashboard pages.

Source of truth for codes: NHANES 2021–2023 variable documentation.

---

## `fact_participants.csv` — grain: one NHANES participant

| Column | Type | Description |
|---|---|---|
| `SEQN` | id | NHANES respondent sequence number |
| `split` | text | `train` or `test` (80/20 stratified, seed 42) |
| `age_years` | int | Age at screening (`RIDAGEYR`); 80 = top-coded 80+ |
| `age_band` | text | `20-34`, `35-44`, `45-54`, `55-64`, `65+` |
| `sex` | text | `Male` / `Female` (`RIAGENDR`) |
| `race_ethnicity` | text | labelled `RIDRETH3` (see `dim_race`) |
| `education` | text | labelled `DMDEDUC2` (see `dim_education`); blank if refused/unknown |
| `income_poverty_ratio` | float | Family income ÷ poverty threshold (`INDFMPIR`); 5 = top-coded 5+ |
| `bmi` | float | Body mass index kg/m² (`BMXBMI`) — **not a model feature** (collinear w/ waist) |
| `bmi_category` | text | `<25`, `25-29.9`, `30-34.9`, `35+` |
| `waist_cm` | float | Waist circumference, cm (`BMXWAIST`) — model feature |
| `waist_band` | text | `<90`, `90-99`, `100-109`, `110+` |
| `hba1c_pct` | float | Glycohemoglobin % (`LBXGH`) — outcome source, **not** a model input |
| `is_diabetes_range` | 0/1 | Target: `hba1c_pct >= 6.5` |
| `outcome_label` | text | `Diabetes-range HbA1c` / `Below range` |
| `risk_score` | float | Calibrated model probability. Leakage-free: out-of-fold for `train`, held-out prediction for `test` |
| `risk_band` | text | `Low` (<0.015) · `Moderate` (0.015–0.0235) · `Elevated (flagged)` (0.0235–0.05) · `High (flagged)` (0.05+) |
| `flagged` | 0/1 | `risk_score >= 0.0235` (the 80 %-recall operating threshold) |
| `survey_weight` | float | MEC exam weight `WTMEC2YR` — use for population-representative aggregates |

### Weighted vs unweighted
Unweighted counts describe *this sample*. For U.S.-population estimates, weight by
`survey_weight` (e.g. `SUM([survey_weight] * [is_diabetes_range]) / SUM([survey_weight])`).

---

## `dim_sex.csv` / `dim_race.csv` / `dim_education.csv`

| File | Key | Label column |
|---|---|---|
| `dim_sex` | `sex_code` (1,2) | `sex` |
| `dim_race` | `race_code` (1,2,3,4,6,7) | `race_ethnicity` |
| `dim_education` | `education_code` (1–5) | `education` |

---

## `agg_prevalence_by_group.csv` — for the "Who is at risk" page

Long / tidy: one row per (dimension, category).

| Column | Description |
|---|---|
| `dimension` | `age_band` \| `sex` \| `race_ethnicity` \| `education` \| `bmi_category` \| `waist_band` |
| `category` | value within that dimension |
| `participants` | n in cohort |
| `positives` | n with diabetes-range HbA1c |
| `prevalence` | `positives / participants` (unweighted) |
| `weighted_prevalence` | survey-weighted rate |
| `mean_risk_score` | mean calibrated model score in the group |

## `agg_calibration_bins.csv` — for the "Model quality" page

| Column | Description |
|---|---|
| `decile` | 1–10, risk-score deciles |
| `participants` | n in decile |
| `mean_predicted_risk` | mean model score |
| `observed_prevalence` | actual positive rate — plot against `mean_predicted_risk` for the calibration line |

## `agg_threshold_sweep.csv` — for the "Operating point" page

| Column | Description |
|---|---|
| `threshold` | decision cut-point, 0.010 → 0.060 |
| `flag_rate` | share of cohort flagged |
| `recall_sensitivity` | TP / (TP + FN) |
| `precision_ppv` | TP / (TP + FP) |
| `specificity` | TN / (TN + FP) |
| `true_positives` / `false_positives` | counts at that threshold |
