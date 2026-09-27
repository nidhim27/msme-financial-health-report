"""Pydantic request/response models for the MSME Financial Health Report API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MSMERawInput(BaseModel):
    """Raw alternative-data record for one MSME, as it would arrive after
    consent-based aggregation from GSTN, UPI/NPCI, Account Aggregator, EPFO,
    and banking rails."""

    msme_id: Optional[str] = Field(None, description="Existing MSME ID; omit to ingest a new business")

    # Business profile
    Customer_Segment: Literal["NTC", "NTB", "Existing-to-Credit"]
    Business_Type: Literal["LLP", "Partnership", "Proprietorship", "Private Limited"]
    Industry: Literal[
        "Logistics", "Services", "Textiles", "Food & Beverage", "Retail",
        "IT & Digital", "Construction", "Healthcare", "Manufacturing", "Trading",
    ]
    Location_Category: Literal["Metro", "Tier 1", "Tier 2", "Tier 3", "Rural"]
    Years_in_Operation: float
    Employee_Count: int
    Annual_Turnover_INR: float

    # GSTN
    Monthly_GST_Sales_INR: float
    Monthly_GST_Purchases_INR: float
    GST_Filing_Timeliness_Pct: float
    GST_Return_Frequency: Literal["Monthly", "Quarterly"]

    # UPI / NPCI
    Monthly_UPI_Inflow_INR: float
    Monthly_UPI_Outflow_INR: float
    UPI_Avg_Ticket_Size_INR: float
    UPI_Daily_Txn_Count: float
    Transaction_Volatility_Index: float

    # Banking / Account Aggregator
    Monthly_Bank_Credits_INR: float
    Monthly_Bank_Debits_INR: float
    Average_Bank_Balance_INR: float
    Cashflow_Stability_Index: float
    Has_Existing_Loan: Literal["Yes", "No"]
    Monthly_Loan_EMI_INR: float = 0
    EMI_On_Time_Rate_Pct: Optional[float] = None
    Overdraft_Usage_Ratio: float

    # EPFO / payroll
    Monthly_Payroll_INR: float
    Salary_Consistency_Pct: float
    Employee_Attrition_Rate_Pct: float

    # Invoices / receivables / vendors
    Avg_Invoice_Payment_Delay_Days: float
    Customer_Concentration_Ratio: float
    Vendor_Payment_Timeliness_Pct: float

    # Business trend
    Seasonality_Index: float
    Revenue_Growth_Rate_Pct: float
    Credit_History_Months: int = 0


class IngestResponse(BaseModel):
    msme_id: str
    status: str
    message: str


class HealthCardResponse(BaseModel):
    msme_id: Optional[str]
    overall_financial_health_score: float
    sub_scores: dict
    probability_of_default: float
    credit_risk_category: str
    credit_risk_probabilities: dict
    credit_eligible: str
    recommended_credit_limit_inr: float
    anomaly_detection: dict
    risk_indicators: list[str]
    key_insights: list[str]
    recommendations: list[str]
    explainability: Optional[dict] = None


class CreditRecommendationResponse(BaseModel):
    msme_id: Optional[str]
    credit_eligible: str
    credit_risk_category: str
    probability_of_default: float
    recommended_credit_limit_inr: float
