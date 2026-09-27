"""
Data ingestion + feature engineering for the Cars24 MSME Financial Health Report.

Raw alternative-data columns (GST, UPI, banking, payroll, invoicing, vendor,
credit-bureau-lite) are turned into a model-ready feature matrix. Target /
outcome columns (the six sub-scores, overall score, PD, risk category,
eligibility, credit limit) are kept separate — they are what the models learn
to predict, never inputs to each other's models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "msme_synthetic_50k.csv"
FEATURE_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "models" / "feature_schema.json"

ID_COL = "MSME_ID"

CATEGORICAL_COLS = [
    "Customer_Segment",
    "Business_Type",
    "Industry",
    "Location_Category",
    "GST_Return_Frequency",
    "Has_Existing_Loan",
]

NUMERIC_COLS = [
    "Years_in_Operation",
    "Employee_Count",
    "Annual_Turnover_INR",
    "Monthly_GST_Sales_INR",
    "Monthly_GST_Purchases_INR",
    "GST_Filing_Timeliness_Pct",
    "Monthly_UPI_Inflow_INR",
    "Monthly_UPI_Outflow_INR",
    "UPI_Avg_Ticket_Size_INR",
    "UPI_Daily_Txn_Count",
    "Transaction_Volatility_Index",
    "Monthly_Bank_Credits_INR",
    "Monthly_Bank_Debits_INR",
    "Average_Bank_Balance_INR",
    "Cashflow_Stability_Index",
    "Monthly_Loan_EMI_INR",
    "EMI_On_Time_Rate_Pct",
    "Overdraft_Usage_Ratio",
    "Monthly_Payroll_INR",
    "Salary_Consistency_Pct",
    "Employee_Attrition_Rate_Pct",
    "Avg_Invoice_Payment_Delay_Days",
    "Customer_Concentration_Ratio",
    "Vendor_Payment_Timeliness_Pct",
    "Seasonality_Index",
    "Revenue_Growth_Rate_Pct",
    "Credit_History_Months",
]

SUBSCORE_TARGETS = [
    "Business_Stability_Score",
    "Cashflow_Score",
    "Revenue_Consistency_Score",
    "Payment_Behaviour_Score",
    "Business_Growth_Score",
    "Compliance_Score",
]

OVERALL_SCORE_TARGET = "Financial_Health_Score"
PD_TARGET = "Probability_of_Default"
RISK_TARGET = "Credit_Risk_Category"
ELIGIBLE_TARGET = "Credit_Eligible"
CREDIT_LIMIT_TARGET = "Recommended_Credit_Limit_INR"

ALL_TARGETS = SUBSCORE_TARGETS + [
    OVERALL_SCORE_TARGET, PD_TARGET, RISK_TARGET, ELIGIBLE_TARGET, CREDIT_LIMIT_TARGET,
]


def load_raw(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive underwriting-style ratio features from the raw alternative-data columns.

    Kept separate from raw columns so both feed the model — ratios capture
    the "shape" of the business (leverage, liquidity, margin) that raw levels
    alone don't.
    """
    out = df.copy()

    out["GST_Sales_Purchase_Ratio"] = out["Monthly_GST_Sales_INR"] / (out["Monthly_GST_Purchases_INR"] + 1)
    out["GST_Sales_To_Turnover_Ratio"] = (out["Monthly_GST_Sales_INR"] * 12) / (out["Annual_Turnover_INR"] + 1)
    out["UPI_Net_Flow_Ratio"] = (out["Monthly_UPI_Inflow_INR"] - out["Monthly_UPI_Outflow_INR"]) / (
        out["Monthly_UPI_Inflow_INR"] + 1
    )
    out["Bank_Net_Flow_Ratio"] = (out["Monthly_Bank_Credits_INR"] - out["Monthly_Bank_Debits_INR"]) / (
        out["Monthly_Bank_Credits_INR"] + 1
    )
    out["Avg_Balance_To_Debits_Ratio"] = out["Average_Bank_Balance_INR"] / (out["Monthly_Bank_Debits_INR"] + 1)
    out["EMI_To_Bank_Credit_Ratio"] = out["Monthly_Loan_EMI_INR"] / (out["Monthly_Bank_Credits_INR"] + 1)
    out["Payroll_To_Turnover_Ratio"] = (out["Monthly_Payroll_INR"] * 12) / (out["Annual_Turnover_INR"] + 1)
    out["Turnover_Per_Employee"] = out["Annual_Turnover_INR"] / (out["Employee_Count"] + 1)
    out["UPI_To_Bank_Credit_Ratio"] = out["Monthly_UPI_Inflow_INR"] / (out["Monthly_Bank_Credits_INR"] + 1)

    return out


ENGINEERED_COLS = [
    "GST_Sales_Purchase_Ratio",
    "GST_Sales_To_Turnover_Ratio",
    "UPI_Net_Flow_Ratio",
    "Bank_Net_Flow_Ratio",
    "Avg_Balance_To_Debits_Ratio",
    "EMI_To_Bank_Credit_Ratio",
    "Payroll_To_Turnover_Ratio",
    "Turnover_Per_Employee",
    "UPI_To_Bank_Credit_Ratio",
]

MODEL_NUMERIC_COLS = NUMERIC_COLS + ENGINEERED_COLS


def build_feature_matrix(df: pd.DataFrame, dummy_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categoricals and align to a fixed column set.

    Pass `dummy_columns` (saved from training) at inference time so a single
    new record encodes to the exact same columns the models were trained on.
    """
    feat = engineer_features(df)
    numeric = feat[MODEL_NUMERIC_COLS]
    categorical = pd.get_dummies(feat[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS).astype(int)
    X = pd.concat([numeric, categorical], axis=1)

    if dummy_columns is None:
        dummy_columns = list(X.columns)
    else:
        X = X.reindex(columns=dummy_columns, fill_value=0)

    return X, dummy_columns


def save_feature_schema(dummy_columns: list[str]) -> None:
    FEATURE_SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_SCHEMA_PATH, "w") as f:
        json.dump({"dummy_columns": dummy_columns}, f, indent=2)


def load_feature_schema() -> list[str]:
    with open(FEATURE_SCHEMA_PATH) as f:
        return json.load(f)["dummy_columns"]


def raw_input_template() -> dict:
    """A dict template of every raw field the API/dashboard need to collect for a new MSME."""
    template = {c: None for c in CATEGORICAL_COLS + NUMERIC_COLS}
    return template
