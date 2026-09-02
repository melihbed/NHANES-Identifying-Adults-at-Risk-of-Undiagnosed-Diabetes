"""Preprocessing: build the leakage-safe ColumnTransformer used by every model."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def make_preprocessor(
    numeric_features=config.NUMERIC_FEATURES,
    categorical_features=config.CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Median-impute + standardise numerics; mode-impute + one-hot encode categoricals.

    Every step is a fitted transformer, so wrapping this in a Pipeline with the
    estimator guarantees the imputation/scaling statistics are learned on training
    folds only.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numeric, list(numeric_features)),
            ("categorical", categorical, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
