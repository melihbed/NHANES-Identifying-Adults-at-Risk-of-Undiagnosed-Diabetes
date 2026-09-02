"""Pipeline behaviour: no leakage, sane thresholds, reproducible test metrics."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src import config
from src.features import make_preprocessor
from src.modeling import (
    make_calibrated_model,
    make_logistic_pipeline,
    out_of_fold_scores,
    select_threshold,
    sweep_thresholds,
)
from src.pipeline import run, split


@pytest.fixture(scope="module")
def xy(analysis_frame):
    X = analysis_frame[config.MODEL_FEATURES]
    y = analysis_frame[config.TARGET].astype(int)
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)


def test_preprocessor_is_fit_only_on_training_rows(xy):
    X_train, X_test, y_train, _ = xy
    pre = make_preprocessor()
    pre.fit(X_train, y_train)
    # median imputation value must come from train, not the full frame
    train_median = X_train["BMXWAIST"].median()
    imputer = pre.named_transformers_["numerical"].named_steps["imputer"]
    idx = config.NUMERIC_FEATURES.index("BMXWAIST")
    assert np.isclose(imputer.statistics_[idx], train_median)


def test_sweep_thresholds_monotone_in_recall(xy):
    X_train, _, y_train, _ = xy
    model = make_logistic_pipeline()
    scores = out_of_fold_scores(model, X_train, y_train)
    sweep = sweep_thresholds(y_train, scores, [0.7, 0.8, 0.9]).sort_values("recall_target")
    # higher target recall => lower threshold, more flags
    assert sweep["threshold"].is_monotonic_decreasing
    assert sweep["flagged_rate"].is_monotonic_increasing
    assert (sweep["recall"] >= sweep["recall_target"] - 1e-9).all()


def test_selected_threshold_hits_recall_target(xy):
    X_train, _, y_train, _ = xy
    scores = out_of_fold_scores(make_calibrated_model(), X_train, y_train)
    t = select_threshold(y_train, scores, config.RECALL_TARGET)
    achieved_recall = ((scores >= t).astype(int) & y_train.to_numpy()).sum() / y_train.sum()
    assert achieved_recall >= config.RECALL_TARGET
    assert 0 < t < 0.5


def test_split_is_stratified(analysis_frame):
    X_train, X_test, y_train, y_test = split(analysis_frame)
    assert abs(y_train.mean() - y_test.mean()) < 0.005


def test_end_to_end_metrics_are_reproducible():
    summary = run(save=False)
    m = summary["test_metrics"]
    assert m["recall"] == pytest.approx(0.95652, abs=1e-4)
    assert m["roc_auc"] == pytest.approx(0.84391, abs=1e-4)
    assert m["pr_auc"] > 2 * m["no_skill_pr_auc"]          # beats chance by >2x
    assert m["brier_score"] < 0.03                          # calibrated
