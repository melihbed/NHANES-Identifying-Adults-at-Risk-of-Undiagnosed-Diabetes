"""Build the star-schema CSV extracts consumed by the Tableau / Power BI dashboard.

Run ``python -m src.dashboard`` from the project root. Writes to ``dashboard/data/``:

  fact_participants.csv        one row per cohort member (features + model risk score)
  dim_sex.csv / dim_race.csv / dim_education.csv   code -> label lookups
  agg_prevalence_by_group.csv  tidy prevalence table for the "who is at risk" page
  agg_calibration_bins.csv     predicted vs observed risk deciles
  agg_threshold_sweep.csv      recall / precision / flag-rate vs decision threshold
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import load_analysis_frame
from .modeling import make_calibrated_model, out_of_fold_scores
from .pipeline import split

AGE_BINS = [20, 35, 45, 55, 65, 120]
AGE_LABELS = ["20-34", "35-44", "45-54", "55-64", "65+"]
WAIST_BINS = [0, 90, 100, 110, 300]
WAIST_LABELS = ["<90", "90-99", "100-109", "110+"]
BMI_BINS = [0, 25, 30, 35, 300]
BMI_LABELS = ["<25 (normal)", "25-29.9 (overweight)", "30-34.9 (obese I)", "35+ (obese II+)"]
RISK_BANDS = [0, 0.015, 0.0235, 0.05, 1.0]
RISK_LABELS = ["Low", "Moderate", "Elevated (flagged)", "High (flagged)"]


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sex"] = out["RIAGENDR"].map(config.SEX_LABELS)
    out["race_ethnicity"] = out["RIDRETH3"].map(config.RACE_LABELS)
    out["education"] = out["DMDEDUC2"].map(config.EDUCATION_LABELS)
    out["age_band"] = pd.cut(out["RIDAGEYR"], AGE_BINS, labels=AGE_LABELS, right=False)
    out["waist_band"] = pd.cut(out["BMXWAIST"], WAIST_BINS, labels=WAIST_LABELS)
    out["bmi_category"] = pd.cut(out["BMXBMI"], BMI_BINS, labels=BMI_LABELS)
    return out


def build_fact(df: pd.DataFrame) -> pd.DataFrame:
    """Participant-level fact table with an out-of-fold model risk score.

    Scores are cross-validated (leakage-free) for training rows and from the
    fitted model for held-out rows, so every participant has an honest score.
    """
    X_train, X_test, y_train, y_test = split(df)
    model = make_calibrated_model()
    oof = out_of_fold_scores(model, X_train, y_train)
    model.fit(X_train, y_train)
    test_scores = model.predict_proba(X_test)[:, 1]

    scores = pd.Series(index=df.index, dtype=float)
    scores.loc[X_train.index] = oof
    scores.loc[X_test.index] = test_scores

    fact = _labelled(df)
    fact["split"] = np.where(df.index.isin(X_test.index), "test", "train")
    fact["risk_score"] = scores.values
    fact["risk_band"] = pd.cut(fact["risk_score"], RISK_BANDS, labels=RISK_LABELS)
    fact["flagged"] = (fact["risk_score"] >= config.OPERATING_THRESHOLD).astype(int)
    fact["outcome_label"] = np.where(fact[config.TARGET] == 1, "Diabetes-range HbA1c", "Below range")

    cols = [
        config.ID_COL, "split",
        "RIDAGEYR", "age_band", "sex", "race_ethnicity", "education",
        "INDFMPIR", "BMXBMI", "bmi_category", "BMXWAIST", "waist_band",
        "LBXGH", config.TARGET, "outcome_label",
        "risk_score", "risk_band", "flagged", "WTMEC2YR",
    ]
    return fact[cols].rename(columns={
        "RIDAGEYR": "age_years", "INDFMPIR": "income_poverty_ratio",
        "BMXBMI": "bmi", "BMXWAIST": "waist_cm", "LBXGH": "hba1c_pct",
        config.TARGET: "is_diabetes_range", "WTMEC2YR": "survey_weight",
    })


def build_prevalence(fact: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in ["age_band", "sex", "race_ethnicity", "education", "bmi_category", "waist_band"]:
        g = fact.groupby(dim, observed=True)
        for category, sub in g:
            rows.append({
                "dimension": dim,
                "category": str(category),
                "participants": int(len(sub)),
                "positives": int(sub["is_diabetes_range"].sum()),
                "prevalence": float(sub["is_diabetes_range"].mean()),
                "weighted_prevalence": float(
                    np.sum(sub["survey_weight"] * sub["is_diabetes_range"])
                    / np.sum(sub["survey_weight"])
                ),
                "mean_risk_score": float(sub["risk_score"].mean()),
            })
    return pd.DataFrame(rows)


def build_calibration_bins(fact: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    f = fact.dropna(subset=["risk_score"]).copy()
    f["decile"] = pd.qcut(f["risk_score"], n_bins, labels=False, duplicates="drop")
    g = f.groupby("decile")
    return pd.DataFrame({
        "decile": g.size().index.astype(int) + 1,
        "participants": g.size().values,
        "mean_predicted_risk": g["risk_score"].mean().values,
        "observed_prevalence": g["is_diabetes_range"].mean().values,
    })


def build_threshold_sweep(fact: pd.DataFrame) -> pd.DataFrame:
    f = fact.dropna(subset=["risk_score"])
    y = f["is_diabetes_range"].to_numpy()
    rows = []
    for t in np.round(np.arange(0.010, 0.061, 0.0025), 4):
        pred = (f["risk_score"].to_numpy() >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        rows.append({
            "threshold": float(t),
            "flag_rate": float(pred.mean()),
            "recall_sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
            "precision_ppv": tp / (tp + fp) if (tp + fp) else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "true_positives": tp,
            "false_positives": fp,
        })
    return pd.DataFrame(rows)


def main() -> None:
    config.DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    df = load_analysis_frame()
    fact = build_fact(df)

    fact.to_csv(config.DASHBOARD_DIR / "fact_participants.csv", index=False)
    build_prevalence(fact).to_csv(config.DASHBOARD_DIR / "agg_prevalence_by_group.csv", index=False)
    build_calibration_bins(fact).to_csv(config.DASHBOARD_DIR / "agg_calibration_bins.csv", index=False)
    build_threshold_sweep(fact).to_csv(config.DASHBOARD_DIR / "agg_threshold_sweep.csv", index=False)

    pd.DataFrame({"sex_code": list(config.SEX_LABELS), "sex": list(config.SEX_LABELS.values())}) \
        .to_csv(config.DASHBOARD_DIR / "dim_sex.csv", index=False)
    pd.DataFrame({"race_code": list(config.RACE_LABELS), "race_ethnicity": list(config.RACE_LABELS.values())}) \
        .to_csv(config.DASHBOARD_DIR / "dim_race.csv", index=False)
    pd.DataFrame({"education_code": list(config.EDUCATION_LABELS), "education": list(config.EDUCATION_LABELS.values())}) \
        .to_csv(config.DASHBOARD_DIR / "dim_education.csv", index=False)

    print(f"wrote 7 extracts to {config.DASHBOARD_DIR.relative_to(config.ROOT)}/")
    print(fact[["risk_score", "is_diabetes_range", "flagged"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
