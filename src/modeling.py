"""Model construction, threshold selection and evaluation helpers.

Mirrors ``notebooks/03_modeling.ipynb`` in a testable form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from . import config
from .features import make_preprocessor


def make_logistic_pipeline(numeric_features=config.NUMERIC_FEATURES, **overrides) -> Pipeline:
    """Preprocessor + tuned, class-weighted logistic regression."""
    params = {**config.LOGREG_PARAMS, **overrides}
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features)),
            ("classifier", LogisticRegression(**params)),
        ]
    )


def make_calibrated_model(base=None, cv=config.CV_FOLDS) -> CalibratedClassifierCV:
    """Platt-scale the logistic pipeline so the scores read as probabilities."""
    base = base if base is not None else make_logistic_pipeline()
    return CalibratedClassifierCV(estimator=clone(base), method="sigmoid", cv=cv)


def cv_splitter(folds=config.CV_FOLDS) -> StratifiedKFold:
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)


def out_of_fold_scores(model, X, y, folds=config.CV_FOLDS) -> np.ndarray:
    """Leakage-free predicted probabilities for every training row."""
    return cross_val_predict(
        clone(model), X, y, cv=cv_splitter(folds), method="predict_proba", n_jobs=-1
    )[:, 1]


def sweep_thresholds(y_true, scores, recall_targets) -> pd.DataFrame:
    """For each target recall, the highest threshold that still reaches it, with metrics."""
    y_true = np.asarray(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve aligns precision[:-1]/recall[:-1] to `thresholds`
    curve = pd.DataFrame(
        {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]}
    )
    rows = []
    for target in recall_targets:
        meeting = curve[curve["recall"] >= target]
        threshold = float(meeting["threshold"].max()) if len(meeting) else 0.0
        rows.append({"recall_target": target, **point_metrics(y_true, scores, threshold)})
    return pd.DataFrame(rows)


def point_metrics(y_true, scores, threshold) -> dict:
    """Confusion-matrix-derived metrics at a fixed decision threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "flagged_rate": float(y_pred.mean()),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def select_threshold(y_true, scores, recall_target=config.RECALL_TARGET) -> float:
    """Operating threshold: highest score cut-point that still reaches `recall_target`."""
    sweep = sweep_thresholds(y_true, scores, [recall_target])
    return float(sweep.iloc[0]["threshold"])


@dataclass
class Evaluation:
    threshold: float
    metrics: dict
    scores: np.ndarray
    predictions: np.ndarray

    @property
    def confusion(self) -> np.ndarray:
        m = self.metrics
        return np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])


def evaluate(model, X_test, y_test, threshold) -> Evaluation:
    """Score a fitted model on held-out data at the chosen threshold."""
    scores = model.predict_proba(X_test)[:, 1]
    pred = (scores >= threshold).astype(int)
    base = point_metrics(y_test, scores, threshold)
    base.update(
        {
            "accuracy": float((pred == np.asarray(y_test)).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "pr_auc": float(average_precision_score(y_test, scores)),
            "brier_score": float(brier_score_loss(y_test, scores)),
            "no_skill_pr_auc": float(np.asarray(y_test).mean()),
        }
    )
    return Evaluation(threshold=float(threshold), metrics=base, scores=scores, predictions=pred)


def coefficient_table(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Standardised-scale logistic coefficients and their odds ratios."""
    pre = fitted_pipeline.named_steps["preprocessor"]
    clf = fitted_pipeline.named_steps["classifier"]
    names = list(pre.get_feature_names_out())
    coefs = clf.coef_.ravel()
    return (
        pd.DataFrame({"feature": names, "coefficient": coefs, "odds_ratio": np.exp(coefs)})
        .sort_values("odds_ratio", ascending=False)
        .reset_index(drop=True)
    )
