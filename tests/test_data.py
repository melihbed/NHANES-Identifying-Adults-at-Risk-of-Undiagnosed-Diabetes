"""Cohort-construction invariants (the guards from notebook 01, as real tests)."""

import numpy as np

from src import config
from src.data import (
    build_analysis_frame,
    build_cohort,
    check_cohort_invariants,
    merge_sources,
    load_raw,
    weighted_prevalence,
)


def test_analysis_frame_invariants(analysis_frame):
    check_cohort_invariants(analysis_frame)


def test_expected_shape_and_prevalence(analysis_frame):
    # Cohort size is fixed by the NHANES 2021-2023 files committed to the repo.
    assert len(analysis_frame) == 4698
    assert analysis_frame[config.TARGET].mean() == 0.024691358024691357
    assert analysis_frame["RIDAGEYR"].min() >= config.MIN_AGE


def test_target_matches_hba1c_rule(analysis_frame):
    expected = (analysis_frame[config.OUTCOME_SOURCE] >= config.A1C_DIABETES_THRESHOLD).astype(int)
    assert analysis_frame[config.TARGET].equals(expected)


def test_no_diagnosed_or_pregnant_leak_into_cohort():
    merged = merge_sources(load_raw())
    cohort = build_cohort(merged)
    # everyone reported "no" to prior diabetes diagnosis
    assert (cohort["DIQ010"] == config.DIQ010_NO).all()
    # nobody flagged pregnant at exam
    assert (cohort["RIDEXPRG"] != config.RIDEXPRG_PREGNANT).all()


def test_education_invalid_codes_recoded():
    cohort = build_cohort(merge_sources(load_raw()))
    analysis = build_analysis_frame(cohort)
    assert not analysis["DMDEDUC2"].isin([7, 9]).any()


def test_weighted_prevalence_below_unweighted(analysis_frame):
    # HbA1c-only definition under-counts vs CDC's combined criterion; weighting
    # pulls the estimate down further. Just assert it stays a small positive rate.
    w = weighted_prevalence(analysis_frame)
    assert 0 < w < analysis_frame[config.TARGET].mean()
    assert np.isclose(w, 0.0203, atol=5e-3)
