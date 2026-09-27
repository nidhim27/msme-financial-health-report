"""
Batch-scores every MSME in the dataset (predictions only, no per-record SHAP —
that stays on-demand/real-time, see src/explain.py) and caches the result so
the API and dashboard can list/filter/search 50k MSMEs instantly instead of
re-running the model pipeline on every page load.

Run: python -m src.batch_score
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly import _prepped, load_anomaly_bundle
from src.data_pipeline import ID_COL, SUBSCORE_TARGETS, build_feature_matrix, load_feature_schema, load_raw
from src.health_card import RISK_CLASSES, _load_models

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "health_cards_summary.csv"


def main():
    print("Loading data + models ...")
    df = load_raw()
    models = _load_models()
    X, _ = build_feature_matrix(df, dummy_columns=models["dummy_columns"])

    out = df[[ID_COL, "Customer_Segment", "Business_Type", "Industry", "Location_Category"]].copy()

    print("Scoring sub-scores ...")
    for target, model in models["subscores"].items():
        out[f"pred_{target}"] = np.clip(model.predict(X), 0, 100).round(1)

    print("Scoring overall score, PD, risk, eligibility, credit limit ...")
    out["pred_Financial_Health_Score"] = np.clip(models["overall"].predict(X), 0, 100).round(1)
    out["pred_Probability_of_Default"] = np.clip(models["pd"].predict(X), 0, 1).round(4)

    risk_proba = models["risk"].predict_proba(X)
    risk_idx = risk_proba.argmax(axis=1)
    out["pred_Credit_Risk_Category"] = [RISK_CLASSES[i] for i in risk_idx]

    eligible_proba = models["eligible"].predict_proba(X)[:, 1]
    out["pred_Credit_Eligible"] = np.where(eligible_proba >= 0.5, "Yes", "No")

    raw_limit = np.expm1(models["credit_limit"].predict(X))
    turnover = df["Annual_Turnover_INR"].values
    capped_limit = np.minimum(raw_limit, turnover)
    out["pred_Recommended_Credit_Limit_INR"] = np.where(
        out["pred_Credit_Eligible"] == "Yes", np.round(capped_limit), 0
    )

    print("Scoring anomaly detector ...")
    bundle = load_anomaly_bundle()
    Xa = _prepped(df)[bundle["features"]]
    Xa_scaled = bundle["scaler"].transform(Xa)
    anomaly_flags = bundle["model"].predict(Xa_scaled)
    anomaly_raw = bundle["model"].decision_function(Xa_scaled)
    out["is_anomalous"] = anomaly_flags == -1
    out["anomaly_score"] = np.clip((0.25 - anomaly_raw) / 0.5 * 100, 0, 100).round(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUTPUT_PATH}")

    print()
    print("Portfolio summary:")
    print(out["pred_Credit_Risk_Category"].value_counts())
    print(out["pred_Credit_Eligible"].value_counts())
    print(f"Anomalous: {out['is_anomalous'].sum()} ({out['is_anomalous'].mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
