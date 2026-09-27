"""
End-to-end scoring engine: raw alternative-data record -> Financial Health Card.

This is the single place that stitches together every model (sub-scores,
overall score, PD, risk category, eligibility, credit limit, anomaly
detection, SHAP explainability, rule-based insights) into the deliverable
described in the assignment: a Financial Health Card with an overall score,
six dimension scores, risk indicators, and key insights/recommendations.

Used directly by both the FastAPI service (api/main.py) and the Streamlit
dashboard (dashboard/app.py) so there is exactly one scoring code path.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.anomaly import load_anomaly_bundle, score_anomaly
from src.data_pipeline import (
    CREDIT_LIMIT_TARGET, ELIGIBLE_TARGET, ID_COL, NUMERIC_COLS, OVERALL_SCORE_TARGET,
    RAW_CSV_PATH, SUBSCORE_TARGETS, build_feature_matrix, load_feature_schema,
)
from src.explain import explain_classifier, explain_regressor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SUBSCORE_META = {
    "Business_Stability_Score": (
        "Business Stability",
        "Grow operating history and reduce employee attrition to strengthen this dimension.",
    ),
    "Cashflow_Score": (
        "Cash Flow",
        "Build higher average bank balances and reduce overdraft dependence.",
    ),
    "Revenue_Consistency_Score": (
        "Revenue Consistency",
        "Reduce transaction volatility and smooth out seasonal revenue swings.",
    ),
    "Payment_Behaviour_Score": (
        "Payment Behaviour",
        "Keep EMI payments on time and settle vendor dues faster.",
    ),
    "Business_Growth_Score": (
        "Business Growth",
        "Sustain revenue and GST sales growth momentum over consecutive periods.",
    ),
    "Compliance_Score": (
        "Compliance",
        "File GST returns on time and keep a consistent return frequency.",
    ),
}


@lru_cache(maxsize=1)
def _load_models():
    models = {
        "subscores": {t: joblib.load(MODELS_DIR / f"subscore_{t}.joblib") for t in SUBSCORE_TARGETS},
        "overall": joblib.load(MODELS_DIR / "overall_score.joblib"),
        "pd": joblib.load(MODELS_DIR / "pd_model.joblib"),
        "risk": joblib.load(MODELS_DIR / "risk_classifier.joblib"),
        "eligible": joblib.load(MODELS_DIR / "eligible_classifier.joblib"),
        "credit_limit": joblib.load(MODELS_DIR / "credit_limit_model.joblib"),
        "dummy_columns": load_feature_schema(),
    }
    return models


@lru_cache(maxsize=1)
def _anomaly_bundle():
    return load_anomaly_bundle()


@lru_cache(maxsize=1)
def _raw_lookup() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV_PATH)
    return df.set_index(ID_COL)


RISK_CLASSES = ["Low", "Medium", "High"]


def get_raw_record(msme_id: str) -> dict:
    row = _raw_lookup().loc[msme_id]
    return row.to_dict()


def list_msme_ids(limit: int = 200) -> list[str]:
    return list(_raw_lookup().index[:limit])


def _risk_indicators(raw: dict, anomaly: dict) -> list[str]:
    flags = []
    if raw.get("Overdraft_Usage_Ratio") is not None and raw["Overdraft_Usage_Ratio"] > 0.5:
        flags.append("High reliance on overdraft facility")
    if raw.get("Customer_Concentration_Ratio") is not None and raw["Customer_Concentration_Ratio"] > 0.5:
        flags.append("Revenue concentrated in a small number of customers")
    if raw.get("Avg_Invoice_Payment_Delay_Days") is not None and raw["Avg_Invoice_Payment_Delay_Days"] > 30:
        flags.append("Slow receivables collection (invoices paid late)")
    if raw.get("Vendor_Payment_Timeliness_Pct") is not None and raw["Vendor_Payment_Timeliness_Pct"] < 70:
        flags.append("Inconsistent vendor payment discipline")
    if raw.get("GST_Filing_Timeliness_Pct") is not None and raw["GST_Filing_Timeliness_Pct"] < 70:
        flags.append("Irregular GST filing compliance")
    if raw.get("Has_Existing_Loan") == "Yes" and raw.get("EMI_On_Time_Rate_Pct") is not None and raw["EMI_On_Time_Rate_Pct"] < 80:
        flags.append("History of delayed EMI payments on existing credit")
    if raw.get("Revenue_Growth_Rate_Pct") is not None and raw["Revenue_Growth_Rate_Pct"] < 0:
        flags.append("Declining revenue trend")
    if raw.get("Employee_Attrition_Rate_Pct") is not None and raw["Employee_Attrition_Rate_Pct"] > 30:
        flags.append("High employee attrition")
    if raw.get("Cashflow_Stability_Index") is not None and raw["Cashflow_Stability_Index"] < 0.4:
        flags.append("Volatile monthly cash flow")
    if raw.get("Transaction_Volatility_Index") is not None and raw["Transaction_Volatility_Index"] > 0.5:
        flags.append("High day-to-day transaction volatility")
    if anomaly["is_anomalous"]:
        flags.append("AI model flagged an unusual financial-behaviour pattern — recommend manual review")
    return flags


def _insights_and_recommendations(subscores: dict) -> tuple[list[str], list[str]]:
    ordered = sorted(subscores.items(), key=lambda kv: kv[1])
    weakest = ordered[:2]
    strongest = ordered[-1]

    insights = [
        f"Strongest dimension: {SUBSCORE_META[strongest[0]][0]} ({strongest[1]:.1f}/100).",
    ]
    for key, val in weakest:
        insights.append(f"Weakest dimension: {SUBSCORE_META[key][0]} ({val:.1f}/100).")

    recommendations = [SUBSCORE_META[key][1] for key, _ in weakest]
    return insights, recommendations


def generate_health_card(raw_record: dict, msme_id: str | None = None, explain: bool = True) -> dict:
    models = _load_models()
    df_row = pd.DataFrame([raw_record])
    for col in NUMERIC_COLS:
        if col in df_row.columns:
            df_row[col] = pd.to_numeric(df_row[col], errors="coerce")
    X, _ = build_feature_matrix(df_row, dummy_columns=models["dummy_columns"])

    subscores = {}
    for target, model in models["subscores"].items():
        subscores[target] = round(float(np.clip(model.predict(X)[0], 0, 100)), 1)

    overall_score = round(float(np.clip(models["overall"].predict(X)[0], 0, 100)), 1)
    pd_pred = round(float(np.clip(models["pd"].predict(X)[0], 0, 1)), 4)

    risk_proba = models["risk"].predict_proba(X)[0]
    risk_idx = int(np.argmax(risk_proba))
    risk_category = RISK_CLASSES[risk_idx]

    eligible_proba = models["eligible"].predict_proba(X)[0]
    eligible = "Yes" if eligible_proba[1] >= 0.5 else "No"

    if eligible == "Yes":
        turnover = float(raw_record.get("Annual_Turnover_INR", 0) or 0)
        raw_limit = float(np.expm1(models["credit_limit"].predict(X)[0]))
        credit_limit = round(min(raw_limit, turnover), 0) if turnover else round(raw_limit, 0)
    else:
        credit_limit = 0.0

    anomaly = score_anomaly(_anomaly_bundle(), df_row)
    risk_indicators = _risk_indicators(raw_record, anomaly)
    insights, recommendations = _insights_and_recommendations(subscores)

    card = {
        "msme_id": msme_id,
        "overall_financial_health_score": overall_score,
        "sub_scores": {k: v for k, v in subscores.items()},
        "probability_of_default": pd_pred,
        "credit_risk_category": risk_category,
        "credit_risk_probabilities": {c: round(float(p), 4) for c, p in zip(RISK_CLASSES, risk_proba)},
        "credit_eligible": eligible,
        "recommended_credit_limit_inr": credit_limit,
        "anomaly_detection": anomaly,
        "risk_indicators": risk_indicators,
        "key_insights": insights,
        "recommendations": recommendations,
    }

    if explain:
        card["explainability"] = {
            "overall_score": explain_regressor("overall_score.joblib", X),
            "credit_risk": explain_classifier("risk_classifier.joblib", X, RISK_CLASSES),
            "credit_eligibility": explain_classifier("eligible_classifier.joblib", X, ["No", "Yes"]),
        }

    return card


def generate_health_card_for_id(msme_id: str, explain: bool = True) -> dict:
    raw = get_raw_record(msme_id)
    return generate_health_card(raw, msme_id=msme_id, explain=explain)
