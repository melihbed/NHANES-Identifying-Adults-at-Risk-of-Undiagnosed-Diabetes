"""Load NHANES source files and build the analytical cohort.

Mirrors ``notebooks/01_data_preparation.ipynb`` in a testable form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw(dataset_dir=config.DATASET_DIR) -> dict[str, pd.DataFrame]:
    """Read the four NHANES XPT files used by the model."""
    frames = {}
    for name, filename in config.NHANES_FILES.items():
        frames[name] = pd.read_sas(dataset_dir / filename, format="xport")
    return frames


def merge_sources(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-merge every source onto the demographics frame on SEQN.

    Left joins keep every demographics participant; each source is validated as
    one row per participant so a silent fan-out cannot inflate the cohort.
    """
    merged = frames["demographics"]
    for key in ("diabetes", "glycohemoglobin", "body_measures"):
        merged = merged.merge(frames[key], on=config.ID_COL, how="left", validate="one_to_one")
    return merged


def build_cohort(merged: pd.DataFrame) -> pd.DataFrame:
    """Apply the eligibility criteria and attach the binary outcome.

    Eligible = adults >= MIN_AGE, self-reported no prior diabetes diagnosis, a
    valid HbA1c measurement, and not pregnant at the exam.
    """
    is_adult = merged["RIDAGEYR"] >= config.MIN_AGE
    reports_no_diabetes = merged["DIQ010"] == config.DIQ010_NO
    has_hba1c = merged[config.OUTCOME_SOURCE].notna()
    not_pregnant = merged["RIDEXPRG"].isna() | (merged["RIDEXPRG"] != config.RIDEXPRG_PREGNANT)

    cohort = merged[is_adult & reports_no_diabetes & has_hba1c & not_pregnant].copy()
    cohort[config.TARGET] = (cohort[config.OUTCOME_SOURCE] >= config.A1C_DIABETES_THRESHOLD).astype(int)
    return cohort


def build_analysis_frame(cohort: pd.DataFrame) -> pd.DataFrame:
    """Select the columns kept for modelling / EDA and clean invalid codes."""
    keep = (
        [config.ID_COL]
        + config.NUMERIC_CANDIDATES
        + config.CATEGORICAL_FEATURES
        + [config.OUTCOME_SOURCE]
        + config.SURVEY_COLS
        + [config.TARGET]
    )
    analysis = cohort[keep].copy()
    # DMDEDUC2: 7 = "Refused", 9 = "Don't know" are not education levels -> missing.
    analysis["DMDEDUC2"] = analysis["DMDEDUC2"].replace({7: np.nan, 9: np.nan})
    return analysis


def weighted_prevalence(df: pd.DataFrame, outcome=config.TARGET, weight="WTMEC2YR") -> float:
    """Survey-weighted mean of a 0/1 outcome."""
    w = df[weight].to_numpy()
    y = df[outcome].to_numpy()
    return float(np.sum(w * y) / np.sum(w))


def prepare(dataset_dir=config.DATASET_DIR, save_to=config.ANALYSIS_COHORT_CSV) -> pd.DataFrame:
    """Full data-preparation path: raw files -> saved analysis cohort."""
    analysis = build_analysis_frame(build_cohort(merge_sources(load_raw(dataset_dir))))
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        analysis.to_csv(save_to, index=False)
    return analysis


def load_analysis_frame(path=config.ANALYSIS_COHORT_CSV) -> pd.DataFrame:
    """Load the prepared cohort, regenerating it from raw files if absent."""
    if not path.exists():
        return prepare(save_to=path)
    return pd.read_csv(path)


def check_cohort_invariants(analysis: pd.DataFrame) -> None:
    """Assertions that must hold for any valid prepared cohort."""
    assert analysis[config.ID_COL].is_unique, "SEQN must be unique"
    assert analysis[config.OUTCOME_SOURCE].notna().all(), "every row needs an HbA1c value"
    assert analysis["RIDAGEYR"].min() >= config.MIN_AGE, "cohort must be adults only"
    assert analysis["WTMEC2YR"].gt(0).all(), "survey weights must be positive"
    assert set(analysis[config.TARGET].unique()) <= {0, 1}, "target must be binary"
