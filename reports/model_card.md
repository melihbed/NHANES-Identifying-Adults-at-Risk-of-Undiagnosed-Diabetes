# Model Card — Undiagnosed Diabetes HbA1c Screener

*Generated to accompany `models/undiagnosed_diabetes_screener.joblib`. Numeric values are
reproduced by `python -m src.pipeline` and stored in `reports/model_metrics.json`.*

## Overview

| | |
|---|---|
| **Task** | Binary probability estimate: diabetes-range HbA1c (≥ 6.5 %) among adults with no prior diabetes diagnosis |
| **Model** | `CalibratedClassifierCV` (sigmoid / Platt, 5-fold) wrapping a class-weighted logistic regression pipeline |
| **Inputs** | age, income-to-poverty ratio, waist circumference, sex, race/ethnicity, education level |
| **Output** | calibrated probability in [0, 1]; a binary flag at the operating threshold **0.0235** |
| **Training data** | NHANES Aug 2021 – Aug 2023, 3,758 adults (93 positive), 80 % stratified split |
| **Intended use** | first-stage screening / triage and epidemiological research |
| **Out of scope** | clinical diagnosis, individual treatment decisions, populations unlike U.S. NHANES adults |

## Pipeline

1. **Preprocess** — numeric: median impute → standardise; categorical: mode impute → one-hot (drop first). Fitted inside CV folds only.
2. **Estimator** — `LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear")`.
   Chosen by GridSearchCV over `C` and class weight scored on CV PR-AUC; the surface was flat
   (top configs within one CV standard error), so a parsimony rule selected the most
   regularised balanced model.
3. **Calibrate** — Platt scaling on 5-fold internal splits.
4. **Threshold** — highest score cut-point reaching ≥ 80 % recall on training out-of-fold predictions.

## Performance — held-out test (940 adults, 23 positive)

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC | **0.844** | ranking quality |
| PR-AUC | **0.074** | vs 0.024 no-skill (≈ 3×) |
| Brier score | **0.024** | calibrated |
| Recall / sensitivity | **0.957** (22/23) | at operating threshold |
| Specificity | 0.690 | |
| Precision / PPV | 0.072 | 1 confirmed case per ~14 flags |
| Flagged rate | 0.326 | share of adults sent for confirmatory HbA1c |
| Confusion @ 0.0235 | TP 22 · FP 284 · FN 1 · TN 633 | |

### Alternative operating points (training out-of-fold)

| Target recall | Threshold | Flagged rate | Precision | Specificity |
|---|---|---|---|---|
| 0.70 | 0.0293 | 26 % | 0.067 | 0.75 |
| **0.80** (selected) | **0.0235** | **35 %** | 0.057 | 0.66 |
| 0.90 | 0.0207 | 40 % | 0.056 | 0.61 |

Because scores are calibrated, the threshold can be re-set for a different
recall/workload trade-off **without retraining**.

## What the model learned (odds ratios, standardised inputs)

| Feature | Odds ratio | Direction |
|---|---|---|
| Waist circumference | 2.87 | ↑ risk |
| Age | 2.13 | ↑ risk |
| Education = 9–11th grade (vs < 9th) | 1.99 | ↑ risk |
| Race/ethnicity = Non-Hispanic Black | 1.94 | ↑ risk |
| Income-to-poverty ratio | 0.79 | ↓ risk |
| Higher education categories | 0.6–0.7 | ↓ risk |

Consistent with the descriptive gradients in `notebooks/02` and with diabetes
epidemiology. These are **associations in observational data**, not causal effects.

## Ethical considerations & limitations

- **Rare positive class** (116 total): all minority-class metrics have wide intervals; the
  recall point estimate especially so.
- **Outcome under-counts** true undiagnosed diabetes (HbA1c-only vs CDC's HbA1c-or-fasting-glucose): weighted 2.0 % vs published 4.5 %.
- **Demographic features carry risk.** Race/ethnicity and education act partly as proxies for
  unmeasured social determinants and access to care. Deploying a flag that uses them requires
  an equity review; a body-measurement-only variant should be evaluated as an alternative.
- **Not survey-weighted for inference.** Design effects would widen the reported intervals.
- **No external validation** yet (planned: NHANES 2017–2020).

## Files

- `models/undiagnosed_diabetes_screener.joblib` — `{"model", "threshold", "features"}`
- `reports/model_metrics.json` — full metric dump
- `reports/figures/` — ROC, PR, calibration, confusion-matrix plots
