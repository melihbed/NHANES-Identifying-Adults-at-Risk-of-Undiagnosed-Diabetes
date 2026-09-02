"""End-to-end reproduction: prepare data -> train -> calibrate -> evaluate -> persist.

Run ``python -m src.pipeline`` from the project root. Outputs:
  * models/undiagnosed_diabetes_screener.joblib  - fitted, calibrated model + threshold
  * reports/model_metrics.json                   - held-out test metrics
  * reports/figures/*.png                        - ROC / PR / calibration / confusion
"""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split

from . import config
from .data import check_cohort_invariants, load_analysis_frame, weighted_prevalence
from .modeling import (
    coefficient_table,
    evaluate,
    make_calibrated_model,
    make_logistic_pipeline,
    out_of_fold_scores,
    select_threshold,
    sweep_thresholds,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def split(df):
    X = df[config.MODEL_FEATURES].copy()
    y = df[config.TARGET].astype(int).copy()
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )


def run(save: bool = True) -> dict:
    df = load_analysis_frame()
    check_cohort_invariants(df)

    X_train, X_test, y_train, y_test = split(df)

    calibrated = make_calibrated_model()
    oof = out_of_fold_scores(calibrated, X_train, y_train)
    operating = sweep_thresholds(
        y_train, oof, [config.RECALL_TARGET, *config.ALT_RECALL_TARGETS]
    ).sort_values("recall_target")
    threshold = select_threshold(y_train, oof, config.RECALL_TARGET)

    calibrated.fit(X_train, y_train)
    result = evaluate(calibrated, X_test, y_test, threshold)

    # interpretable coefficients come from the uncalibrated logistic fit
    logit = make_logistic_pipeline().fit(X_train, y_train)
    coefs = coefficient_table(logit)

    summary = {
        "cohort_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_positives": int(y_train.sum()),
        "test_positives": int(y_test.sum()),
        "unweighted_prevalence": float(df[config.TARGET].mean()),
        "weighted_prevalence": weighted_prevalence(df),
        "features": config.MODEL_FEATURES,
        "logreg_params": {k: str(v) for k, v in config.LOGREG_PARAMS.items()},
        "operating_points": operating.round(5).to_dict(orient="records"),
        "selected_threshold": float(threshold),
        "test_metrics": {k: round(v, 5) if isinstance(v, float) else v
                         for k, v in result.metrics.items()},
        "top_odds_ratios": coefs.head(6).round(4).to_dict(orient="records"),
    }

    if save:
        _persist(calibrated, threshold, summary, result, y_test)
    return summary


def _persist(model, threshold, summary, result, y_test):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": model, "threshold": threshold, "features": config.MODEL_FEATURES},
        config.MODEL_PATH,
    )
    config.METRICS_PATH.write_text(json.dumps(summary, indent=2))
    _figures(result, y_test)
    print(f"model  -> {config.MODEL_PATH.relative_to(config.ROOT)}")
    print(f"metrics-> {config.METRICS_PATH.relative_to(config.ROOT)}")


def _figures(result, y_test):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        PrecisionRecallDisplay,
        RocCurveDisplay,
        ConfusionMatrixDisplay,
    )
    from sklearn.calibration import CalibrationDisplay

    labels = ["Outside range", "Diabetes range"]

    RocCurveDisplay.from_predictions(y_test, result.scores)
    plt.title("ROC - held-out test")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "roc_curve.png", dpi=120)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, result.scores)
    plt.title("Precision-Recall - held-out test")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "pr_curve.png", dpi=120)
    plt.close()

    CalibrationDisplay.from_predictions(y_test, result.scores, n_bins=8, strategy="quantile")
    plt.title("Calibration - held-out test")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "calibration_curve.png", dpi=120)
    plt.close()

    ConfusionMatrixDisplay(result.confusion, display_labels=labels).plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix @ threshold {result.threshold:.3f}")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    out = run(save=True)
    print(json.dumps(out["test_metrics"], indent=2))
