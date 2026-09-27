"""
Streamlit dashboard for the Cars24 MSME Financial Health Report.

Run: streamlit run dashboard/app.py

Talks directly to the scoring engine in src/health_card.py (no need to have
the FastAPI service running) so it can be demoed standalone, while the API in
api/main.py exposes the exact same logic over REST for other consumers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import RAW_CSV_PATH
from src.health_card import generate_health_card, list_msme_ids
from src.trends import build_trend_bundle

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "health_cards_summary.csv"

st.set_page_config(page_title="Cars24 MSME Financial Health Report", layout="wide", page_icon="\U0001F4CA")

RISK_COLORS = {"Low": "#1a9850", "Medium": "#f5a623", "High": "#d73027"}


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_PATH)


@st.cache_data
def load_raw_df() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV_PATH)


def render_score_header(card: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall Financial Health Score", f"{card['overall_financial_health_score']:.1f} / 100")
    risk = card["credit_risk_category"]
    c2.markdown(
        f"**Credit Risk Category**<br><span style='font-size:1.6em;color:{RISK_COLORS[risk]}'>{risk}</span>",
        unsafe_allow_html=True,
    )
    c3.metric("Probability of Default", f"{card['probability_of_default']*100:.2f}%")
    c4.metric("Credit Eligible", card["credit_eligible"])
    c5.metric("Recommended Limit", f"₹{card['recommended_credit_limit_inr']:,.0f}")


def render_subscores(card: dict):
    sub = card["sub_scores"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(sub.values()) + [list(sub.values())[0]],
                                   theta=list(sub.keys()) + [list(sub.keys())[0]],
                                   fill="toself", name="Score"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False,
                       height=420, margin=dict(t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)


def render_bar_subscores(card: dict):
    sub = card["sub_scores"]
    df = pd.DataFrame({"Dimension": list(sub.keys()), "Score": list(sub.values())})
    df["Dimension"] = df["Dimension"].str.replace("_Score", "").str.replace("_", " ")
    fig = px.bar(df, x="Score", y="Dimension", orientation="h", range_x=[0, 100],
                 color="Score", color_continuous_scale="RdYlGn")
    fig.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def render_explainability(card: dict):
    drivers = card["explainability"]["overall_score"]["top_drivers"]
    df = pd.DataFrame(drivers)
    df["color"] = df["impact"].apply(lambda v: "Increases score" if v > 0 else "Decreases score")
    fig = px.bar(
        df.sort_values("impact"), x="impact", y="feature", orientation="h", color="color",
        color_discrete_map={"Increases score": "#1a9850", "Decreases score": "#d73027"},
        labels={"impact": "SHAP impact on Overall Financial Health Score", "feature": ""},
    )
    fig.update_layout(height=340, margin=dict(t=10, b=10), legend_title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Base (population average) score: {card['explainability']['overall_score']['base_value']:.1f} → "
        f"model prediction: {card['explainability']['overall_score']['prediction']:.1f}"
    )


def render_trends(trends: dict):
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**GST Performance**")
        df = pd.DataFrame({"Month": trends["months"], "GST Sales": trends["gst_sales"], "GST Purchases": trends["gst_purchases"]})
        st.plotly_chart(px.line(df, x="Month", y=["GST Sales", "GST Purchases"], markers=True), use_container_width=True)

        st.markdown("**Cash Flow Analysis (Bank)**")
        df = pd.DataFrame({"Month": trends["months"], "Credits": trends["bank_credits"], "Debits": trends["bank_debits"]})
        st.plotly_chart(px.line(df, x="Month", y=["Credits", "Debits"], markers=True), use_container_width=True)
    with t2:
        st.markdown("**UPI Transaction Trend**")
        df = pd.DataFrame({"Month": trends["months"], "UPI Inflow": trends["upi_inflow"], "UPI Outflow": trends["upi_outflow"]})
        st.plotly_chart(px.line(df, x="Month", y=["UPI Inflow", "UPI Outflow"], markers=True), use_container_width=True)

        st.markdown("**Payroll Consistency**")
        df = pd.DataFrame({"Month": trends["months"], "Payroll": trends["payroll"]})
        st.plotly_chart(px.bar(df, x="Month", y="Payroll"), use_container_width=True)

    st.caption(
        "Trend charts are reconstructed from the latest snapshot's growth rate and seasonality "
        "index (the dataset is cross-sectional, one row per MSME). A production deployment would "
        "plot actual monthly GST/UPI/bank statement history via the Account Aggregator feed."
    )


def render_insights(card: dict):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk Indicators**")
        if card["anomaly_detection"]["is_anomalous"]:
            st.error(f"⚠ Anomalous behaviour flagged (anomaly score {card['anomaly_detection']['anomaly_score']}/100)")
        if card["risk_indicators"]:
            for flag in card["risk_indicators"]:
                st.warning(flag)
        else:
            st.success("No material risk indicators detected.")
    with c2:
        st.markdown("**Key Insights & Recommendations**")
        for i in card["key_insights"]:
            st.info(i)
        for r in card["recommendations"]:
            st.write(f"→ {r}")


def render_health_card(record: dict, msme_id: str):
    card = generate_health_card(record, msme_id=msme_id, explain=True)
    st.subheader(f"Financial Health Card — {msme_id}")
    render_score_header(card)
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Multidimensional Score Radar**")
        render_subscores(card)
    with col2:
        st.markdown("**Dimension Breakdown**")
        render_bar_subscores(card)

    st.divider()
    st.markdown("### Explainability — why this score")
    render_explainability(card)

    st.divider()
    st.markdown("### Revenue, Cash Flow & Operational Trends")
    trends = build_trend_bundle(msme_id, record)
    render_trends(trends)

    st.divider()
    render_insights(card)


def raw_input_form(defaults: dict | None = None) -> dict | None:
    defaults = defaults or {}
    with st.form("new_msme_form"):
        c1, c2, c3, c4 = st.columns(4)
        segment = c1.selectbox("Customer Segment", ["NTC", "NTB", "Existing-to-Credit"])
        btype = c2.selectbox("Business Type", ["Proprietorship", "Partnership", "LLP", "Private Limited"])
        industry = c3.selectbox("Industry", ["Retail", "Services", "Trading", "Manufacturing", "Logistics",
                                              "Textiles", "Food & Beverage", "IT & Digital", "Construction", "Healthcare"])
        location = c4.selectbox("Location Category", ["Metro", "Tier 1", "Tier 2", "Tier 3", "Rural"])

        c1, c2, c3 = st.columns(3)
        years = c1.number_input("Years in Operation", 0.0, 60.0, 3.0)
        employees = c2.number_input("Employee Count", 0, 5000, 5)
        turnover = c3.number_input("Annual Turnover (INR)", 0.0, 1e9, 3_000_000.0, step=100000.0)

        st.markdown("**GST**")
        c1, c2, c3, c4 = st.columns(4)
        gst_sales = c1.number_input("Monthly GST Sales (INR)", 0.0, 1e8, 250000.0, step=10000.0)
        gst_purchases = c2.number_input("Monthly GST Purchases (INR)", 0.0, 1e8, 180000.0, step=10000.0)
        gst_timeliness = c3.slider("GST Filing Timeliness %", 0.0, 100.0, 90.0)
        gst_freq = c4.selectbox("GST Return Frequency", ["Monthly", "Quarterly"])

        st.markdown("**UPI**")
        c1, c2, c3, c4 = st.columns(4)
        upi_in = c1.number_input("Monthly UPI Inflow (INR)", 0.0, 1e8, 150000.0, step=10000.0)
        upi_out = c2.number_input("Monthly UPI Outflow (INR)", 0.0, 1e8, 110000.0, step=10000.0)
        upi_ticket = c3.number_input("UPI Avg Ticket Size (INR)", 0.0, 1e6, 500.0)
        upi_txn = c4.number_input("UPI Daily Txn Count", 0.0, 5000.0, 15.0)

        st.markdown("**Banking**")
        c1, c2, c3, c4 = st.columns(4)
        bank_credits = c1.number_input("Monthly Bank Credits (INR)", 0.0, 1e8, 300000.0, step=10000.0)
        bank_debits = c2.number_input("Monthly Bank Debits (INR)", 0.0, 1e8, 260000.0, step=10000.0)
        avg_balance = c3.number_input("Average Bank Balance (INR)", 0.0, 1e8, 90000.0, step=5000.0)
        cashflow_idx = c4.slider("Cashflow Stability Index", 0.0, 1.0, 0.7)

        c1, c2, c3, c4 = st.columns(4)
        volatility_idx = c1.slider("Transaction Volatility Index", 0.0, 1.0, 0.2)
        has_loan = c2.selectbox("Has Existing Loan", ["No", "Yes"])
        emi = c3.number_input("Monthly Loan EMI (INR)", 0.0, 1e7, 0.0, step=1000.0)
        emi_ontime = c4.slider("EMI On-Time Rate % (if existing loan)", 0.0, 100.0, 90.0)

        c1, c2 = st.columns(2)
        overdraft = c1.slider("Overdraft Usage Ratio", 0.0, 1.0, 0.1)
        credit_hist = c2.number_input("Credit History (months)", 0, 600, 0)

        st.markdown("**Payroll (EPFO)**")
        c1, c2, c3 = st.columns(3)
        payroll = c1.number_input("Monthly Payroll (INR)", 0.0, 1e7, 60000.0, step=5000.0)
        salary_consistency = c2.slider("Salary Consistency %", 0.0, 100.0, 95.0)
        attrition = c3.slider("Employee Attrition Rate %", 0.0, 100.0, 10.0)

        st.markdown("**Invoices, Vendors & Growth**")
        c1, c2, c3 = st.columns(3)
        invoice_delay = c1.number_input("Avg Invoice Payment Delay (days)", 0.0, 365.0, 10.0)
        concentration = c2.slider("Customer Concentration Ratio", 0.0, 1.0, 0.3)
        vendor_timeliness = c3.slider("Vendor Payment Timeliness %", 0.0, 100.0, 90.0)

        c1, c2 = st.columns(2)
        seasonality = c1.slider("Seasonality Index", 0.0, 1.0, 0.15)
        growth = c2.number_input("Revenue Growth Rate %", -100.0, 500.0, 15.0)

        submitted = st.form_submit_button("Generate Financial Health Card")

    if not submitted:
        return None

    return {
        "Customer_Segment": segment, "Business_Type": btype, "Industry": industry, "Location_Category": location,
        "Years_in_Operation": years, "Employee_Count": employees, "Annual_Turnover_INR": turnover,
        "Monthly_GST_Sales_INR": gst_sales, "Monthly_GST_Purchases_INR": gst_purchases,
        "GST_Filing_Timeliness_Pct": gst_timeliness, "GST_Return_Frequency": gst_freq,
        "Monthly_UPI_Inflow_INR": upi_in, "Monthly_UPI_Outflow_INR": upi_out,
        "UPI_Avg_Ticket_Size_INR": upi_ticket, "UPI_Daily_Txn_Count": upi_txn,
        "Transaction_Volatility_Index": volatility_idx,
        "Monthly_Bank_Credits_INR": bank_credits, "Monthly_Bank_Debits_INR": bank_debits,
        "Average_Bank_Balance_INR": avg_balance, "Cashflow_Stability_Index": cashflow_idx,
        "Has_Existing_Loan": has_loan, "Monthly_Loan_EMI_INR": emi,
        "EMI_On_Time_Rate_Pct": emi_ontime if has_loan == "Yes" else None,
        "Overdraft_Usage_Ratio": overdraft,
        "Monthly_Payroll_INR": payroll, "Salary_Consistency_Pct": salary_consistency,
        "Employee_Attrition_Rate_Pct": attrition,
        "Avg_Invoice_Payment_Delay_Days": invoice_delay, "Customer_Concentration_Ratio": concentration,
        "Vendor_Payment_Timeliness_Pct": vendor_timeliness,
        "Seasonality_Index": seasonality, "Revenue_Growth_Rate_Pct": growth,
        "Credit_History_Months": credit_hist,
    }


def render_portfolio():
    df = load_summary()
    st.subheader("Portfolio Overview — 50,000 MSMEs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total MSMEs Scored", f"{len(df):,}")
    c2.metric("Credit Eligible", f"{(df['pred_Credit_Eligible']=='Yes').mean()*100:.1f}%")
    c3.metric("Avg Financial Health Score", f"{df['pred_Financial_Health_Score'].mean():.1f}")
    c4.metric("Anomalous MSMEs", f"{df['is_anomalous'].mean()*100:.2f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Credit Risk Distribution**")
        counts = df["pred_Credit_Risk_Category"].value_counts().reindex(["Low", "Medium", "High"])
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_map=RISK_COLORS,
                     labels={"x": "Risk Category", "y": "Count"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Financial Health Score Distribution**")
        st.plotly_chart(px.histogram(df, x="pred_Financial_Health_Score", nbins=40), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Avg Score by Industry**")
        by_ind = df.groupby("Industry")["pred_Financial_Health_Score"].mean().sort_values()
        st.plotly_chart(px.bar(x=by_ind.values, y=by_ind.index, orientation="h",
                                labels={"x": "Avg Score", "y": ""}), use_container_width=True)
    with c2:
        st.markdown("**Eligibility by Customer Segment (financial inclusion view)**")
        seg = pd.crosstab(df["Customer_Segment"], df["pred_Credit_Eligible"], normalize="index") * 100
        st.plotly_chart(px.bar(seg, barmode="stack", labels={"value": "% of segment", "Customer_Segment": ""}),
                         use_container_width=True)

    st.markdown("**Browse scored MSMEs**")
    st.dataframe(df.head(500), use_container_width=True, height=300)


def main():
    st.title("📊 Cars24 MSME Financial Health Report")
    st.caption("Alternative-data-driven Financial Health Card & explainable credit scoring for NTC/NTB MSMEs.")

    mode = st.sidebar.radio("Mode", ["Lookup existing MSME", "Score a new MSME", "Portfolio overview"])

    if mode == "Lookup existing MSME":
        ids = list_msme_ids(50000)
        query = st.sidebar.text_input("Search MSME_ID", "")
        filtered = [i for i in ids if query.lower() in i.lower()] if query else ids[:500]
        msme_id = st.sidebar.selectbox("Select MSME", filtered)
        if msme_id:
            raw_df = load_raw_df().set_index("MSME_ID")
            record = raw_df.loc[msme_id].to_dict()
            render_health_card(record, msme_id)

    elif mode == "Score a new MSME":
        st.sidebar.info("Fill in the alternative-data fields (as if aggregated via GSTN/UPI/AA/EPFO consent flows) and submit.")
        record = raw_input_form()
        if record:
            render_health_card(record, msme_id="NEW-MSME")

    else:
        render_portfolio()


if __name__ == "__main__":
    main()
