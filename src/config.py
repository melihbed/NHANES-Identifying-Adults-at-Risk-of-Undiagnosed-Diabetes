"""Project-wide configuration: paths, feature definitions and modelling constants.

Keeping these in one place means the notebooks, the packaged pipeline and the unit
tests all agree on the cohort definition and the model specification.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DASHBOARD_DIR = ROOT / "dashboard" / "data"

ANALYSIS_COHORT_CSV = ARTIFACTS_DIR / "analysis_cohort.csv"
MODEL_PATH = MODELS_DIR / "undiagnosed_diabetes_screener.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"

# ------------------------------------------------------------------ NHANES sources
# 2021 - August 2023 cycle ("_L" suffix). SEQN is the participant identifier.
NHANES_FILES = {
    "demographics": "DEMO_L.xpt",       # age, sex, race/ethnicity, education, income, weights
    "diabetes": "DIQ_L.xpt",            # prior diagnosis question (DIQ010)
    "glycohemoglobin": "GHB_L.xpt",     # HbA1c (LBXGH) - the outcome source
    "body_measures": "BMX_L.xpt",       # BMI (BMXBMI), waist circumference (BMXWAIST)
}

# ---------------------------------------------------------------- cohort definition
MIN_AGE = 20                 # 20+ so the education variable (DMDEDUC2) is defined for everyone
A1C_DIABETES_THRESHOLD = 6.5  # % - ADA diabetes-range HbA1c cut-point
DIQ010_NO = 2                # "No" to "ever told you have diabetes"
RIDEXPRG_PREGNANT = 1        # pregnant at exam -> excluded

# ------------------------------------------------------------------------ features
# Candidate predictors are all "routinely available" at a primary-care visit.
NUMERIC_CANDIDATES = ["RIDAGEYR", "INDFMPIR", "BMXBMI", "BMXWAIST"]
CATEGORICAL_FEATURES = ["RIAGENDR", "RIDRETH3", "DMDEDUC2"]

# BMXBMI is dropped from the final model: it is Spearman rho = 0.89 collinear with
# BMXWAIST, and a waist-only model matched BMI+waist on cross-validated PR-AUC while
# keeping the logistic coefficients interpretable (see notebook 03).
NUMERIC_FEATURES = ["RIDAGEYR", "INDFMPIR", "BMXWAIST"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "a1c_diabetes_range"

# survey-design columns carried through for weighted sanity checks
SURVEY_COLS = ["WTMEC2YR", "WTPH2YR", "SDMVSTRA", "SDMVPSU"]
ID_COL = "SEQN"
OUTCOME_SOURCE = "LBXGH"

# --------------------------------------------------------------- modelling choices
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Tuned logistic-regression hyper-parameters. The GridSearchCV surface was flat
# (all top configs within one CV standard error), so a parsimony rule was applied:
# the most-regularised model with the standard balanced class weight.
LOGREG_PARAMS = dict(
    C=0.1,
    class_weight="balanced",
    solver="liblinear",
    max_iter=2000,
    random_state=RANDOM_STATE,
)

# Screening operating point: smallest score threshold reaching this out-of-fold recall.
RECALL_TARGET = 0.80
ALT_RECALL_TARGETS = (0.70, 0.90)

# Decision threshold produced by the pipeline for RECALL_TARGET on the training
# out-of-fold scores. Written back here so the dashboard extract and notebooks agree;
# `python -m src.pipeline` recomputes and stores the exact value in model_metrics.json.
OPERATING_THRESHOLD = 0.0235

# Human-readable labels for the categorical NHANES codes (used in EDA / dashboard).
SEX_LABELS = {1: "Male", 2: "Female"}
RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other / Multi-racial",
}
EDUCATION_LABELS = {
    1: "< 9th grade",
    2: "9-11th grade",
    3: "High school / GED",
    4: "Some college / AA",
    5: "College graduate+",
}
