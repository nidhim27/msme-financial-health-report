"""
Unsupervised anomaly detection over transactional behaviour.

There is no labelled "fraud"/"anomaly" column in the source data (nor would
there be in production — anomalous behaviour is precisely what you don't have
a clean label for), so this is IsolationForest over the raw transactional
signals: it flags MSMEs whose GST/UPI/banking/payroll pattern sits far from
the bulk of the population, which is what an underwriter actually wants
surfaced ("this business's numbers don't look like other businesses like it").
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import load_raw

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ANOMALY_FEATURES = [
    "Transaction_Volatility_Index",
    "Cashflow_Stability_Index",
    "Overdraft_Usage_Ratio",
    "GST_Filing_Timeliness_Pct",
    "EMI_On_Time_Rate_Pct",
    "Salary_Consistency_Pct",
    "Employee_Attrition_Rate_Pct",
    "Avg_Invoice_Payment_Delay_Days",
    "Customer_Concentration_Ratio",
    "Vendor_Payment_Timeliness_Pct",
    "Revenue_Growth_Rate_Pct",
    "UPI_Avg_Ticket_Size_INR",
    "UPI_Daily_Txn_Count",
]


def _prepped(df: pd.DataFrame) -> pd.DataFrame:
    X = df[ANOMALY_FEATURES].copy()
    # EMI_On_Time_Rate_Pct is null for no-loan businesses — neutral-fill so it
    # doesn't itself look anomalous just for not having a loan.
    X["EMI_On_Time_Rate_Pct"] = X["EMI_On_Time_Rate_Pct"].fillna(X["EMI_On_Time_Rate_Pct"].median())
    return X


def train_anomaly_model(contamination: float = 0.03, random_state: int = 42):
    df = load_raw()
    X = _prepped(df)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    model = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=random_state, n_jobs=-1
    )
    model.fit(Xs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": ANOMALY_FEATURES}, MODELS_DIR / "anomaly_model.joblib")
    return model, scaler


def load_anomaly_bundle():
    return joblib.load(MODELS_DIR / "anomaly_model.joblib")


def score_anomaly(bundle: dict, raw_row: pd.DataFrame) -> dict:
    """raw_row: single-row DataFrame with the raw MSME columns."""
    X = _prepped(raw_row)[bundle["features"]]
    Xs = bundle["scaler"].transform(X)
    model = bundle["model"]
    raw_score = model.decision_function(Xs)[0]  # higher = more normal
    is_anomaly = bool(model.predict(Xs)[0] == -1)
    # Convert to a 0-100 "anomaly risk" scale (higher = more anomalous) for display.
    anomaly_pct = float(np.clip((0.25 - raw_score) / 0.5 * 100, 0, 100))
    return {"is_anomalous": is_anomaly, "anomaly_score": round(anomaly_pct, 1)}


if __name__ == "__main__":
    print("Training anomaly detector on transactional features ...")
    model, scaler = train_anomaly_model()
    df = load_raw()
    X = _prepped(df)
    Xs = scaler.transform(X)
    flags = model.predict(Xs)
    print(f"Flagged {(flags == -1).sum()} / {len(df)} MSMEs as anomalous ({(flags==-1).mean()*100:.2f}%)")
