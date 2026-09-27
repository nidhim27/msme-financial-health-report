"""
Reconstructs a plausible last-N-month trajectory from a single cross-sectional
snapshot (growth rate + seasonality index), so both the API and the dashboard
can render trend charts without needing raw monthly time series in the
dataset. A production system sources this directly from monthly GST/UPI/bank
statements instead of simulating it — see docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import hashlib

import numpy as np


def simulate_monthly_series(
    seed_key: str, base_value: float, seasonality_index: float, growth_rate_pct: float, months: int = 12
) -> list[float]:
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    monthly_growth = (growth_rate_pct / 100) / 12
    trend = np.array([(1 + monthly_growth) ** (m - months) for m in range(months)])
    season = 1 + seasonality_index * 0.3 * np.sin(np.linspace(0, 2 * np.pi, months))
    noise = rng.normal(1.0, 0.04, months)
    series = base_value * trend * season * noise
    return [round(float(v), 2) for v in series]


def build_trend_bundle(msme_id: str, record: dict, months: int = 12) -> dict:
    growth = record.get("Revenue_Growth_Rate_Pct", 0) or 0
    season = record.get("Seasonality_Index", 0) or 0
    return {
        "months": [f"M-{months - 1 - i}" for i in range(months)],
        "gst_sales": simulate_monthly_series(msme_id + "gst", record["Monthly_GST_Sales_INR"], season, growth, months),
        "gst_purchases": simulate_monthly_series(msme_id + "gst_p", record["Monthly_GST_Purchases_INR"], season, growth * 0.8, months),
        "upi_inflow": simulate_monthly_series(msme_id + "upi_in", record["Monthly_UPI_Inflow_INR"], season, growth, months),
        "upi_outflow": simulate_monthly_series(msme_id + "upi_out", record["Monthly_UPI_Outflow_INR"], season, growth * 0.6, months),
        "bank_credits": simulate_monthly_series(msme_id + "bank_cr", record["Monthly_Bank_Credits_INR"], season, growth, months),
        "bank_debits": simulate_monthly_series(msme_id + "bank_db", record["Monthly_Bank_Debits_INR"], season, growth * 0.7, months),
        "payroll": simulate_monthly_series(msme_id + "payroll", record["Monthly_Payroll_INR"], 0.1, max(growth * 0.3, 0), months),
    }
