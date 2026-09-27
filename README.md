# Cars24 MSME Financial Health Report

**Created by Nidhi Mehra**

An alternative-data-driven **Financial Health Card** and explainable credit
scoring system for MSMEs with limited or no traditional credit history
(New-to-Credit / New-to-Bank), built on a synthetic 50,000-MSME dataset
combining GST, UPI, banking (Account Aggregator), EPFO payroll, and
invoicing/vendor signals.

Given a business's alternative-data footprint, the system produces:

- An **Overall Financial Health Score (0–100)** plus six dimension scores
  (Business Stability, Cash Flow, Revenue Consistency, Payment Behaviour,
  Business Growth, Compliance)
- A **Probability of Default**, **Credit Risk Category** (Low/Medium/High),
  **Credit Eligibility** (Yes/No), and **Recommended Credit Limit**
- **Anomaly detection** flagging unusual financial-behaviour patterns
- **SHAP-based explainability** for every score/decision
- Rule-based **risk indicators, key insights, and recommendations**

...served over a REST API, visualized in an interactive dashboard, and fully
reproducible from raw data via a training pipeline.

## Repository layout

```
ProjectMSME/
├── data/
│   ├── msme_synthetic_50k.csv        # raw dataset (included, extracted from msme_synthetic_50k.zip)
│   └── processed/health_cards_summary.csv  # batch-scored population (included, regenerable via src/batch_score.py)
├── src/
│   ├── data_pipeline.py              # raw -> feature matrix (ratios + one-hot encoding)
│   ├── train.py                      # trains all 9 supervised models
│   ├── anomaly.py                    # unsupervised anomaly detector (IsolationForest)
│   ├── explain.py                    # SHAP explainability helpers
│   ├── health_card.py                # end-to-end scoring engine (the one seam everything else uses)
│   ├── batch_score.py                # pre-scores the full population for fast lookups
│   └── trends.py                     # reconstructs monthly trend series for dashboard/API charts
├── api/
│   ├── main.py                       # FastAPI REST service
│   └── schemas.py                    # request/response models
├── dashboard/
│   └── app.py                        # Streamlit dashboard
├── models/                           # trained model artifacts (included, regenerable via src/train.py + src/anomaly.py)
├── docs/
│   ├── ARCHITECTURE.md               # end-to-end system architecture, data pipeline, AI/ML workflow, tech stack
│   ├── INTEGRATION_STRATEGY.md       # ULI / OCEN / Account Aggregator / GSTN / banking API integration
│   ├── API_DESIGN.md                 # REST API reference
│   └── MODEL_REPORT.md               # model metrics + explainability/anomaly methodology (auto-generated)
├── requirements.txt
└── MSME Financial Health Card Assignment.pdf   # original problem statement
```

## Setup

The dataset and trained models are already included in this repo (see below),
so you can jump straight to **Run the API** / **Run the dashboard** after
this step — no training required to try it out.

```bash
git clone <this-repo-url> && cd ProjectMSME
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Reproduce the models from scratch (optional)

```bash
python -m src.train          # trains the 6 sub-score + overall score + PD + risk + eligibility + credit-limit models
python -m src.anomaly         # trains the anomaly detector
python -m src.batch_score     # pre-scores all 50,000 MSMEs for fast dashboard/API lookups
```

This writes trained artifacts to `models/` and a fresh `docs/MODEL_REPORT.md`
with accuracy/R²/F1 metrics for every model.

## Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

- Interactive docs: http://localhost:8000/docs
- Try it: `curl http://localhost:8000/api/v1/health-card/MSME0000001`

See `docs/API_DESIGN.md` for the full endpoint reference.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at http://localhost:8501 with three modes: look up an existing MSME by
ID, score a brand-new MSME by filling in its alternative-data profile, or
view the portfolio overview across all 50,000 scored MSMEs.

The dashboard calls the scoring engine directly (`src/health_card.py`) and
does not require the API to be running; the API is a separate integration
surface for other systems (see `docs/INTEGRATION_STRATEGY.md`).

## Deliverables mapping

| Assignment deliverable | Where |
|---|---|
| Solution Architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Financial Health Card | `src/health_card.py` (engine) + dashboard "Lookup"/"Score a new MSME" views |
| AI/ML Model + explainability | `src/train.py`, `src/anomaly.py`, `src/explain.py`; metrics in [`docs/MODEL_REPORT.md`](docs/MODEL_REPORT.md) |
| Dashboard | `dashboard/app.py` (Streamlit) |
| API Design | `api/main.py`, `api/schemas.py`; reference in [`docs/API_DESIGN.md`](docs/API_DESIGN.md) |
| Integration Strategy (ULI/OCEN/AA/GSTN/Banking) | [`docs/INTEGRATION_STRATEGY.md`](docs/INTEGRATION_STRATEGY.md) |

## Notes on the data

`data/msme_synthetic_50k.csv` is a synthetic, cross-sectional (one snapshot
per MSME) dataset — it already includes the six sub-scores, overall score,
PD, risk category, eligibility, and credit limit as generated labels. The
models in this repo are trained to **predict** those labels from the raw
alternative-data fields alone (never from each other or from the labels
themselves), which is the realistic production setup: at scoring time you
only ever have the raw GST/UPI/AA/EPFO/invoice data, not a pre-computed
score. Sub-score models reach R² ≈ 0.999 (they're close to deterministic
functions of a feature subset); the overall score model reaches R² ≈ 0.87
(see `docs/MODEL_REPORT.md` for the full breakdown and why a higher ceiling
isn't achievable from the sub-scores alone — the label carries its own
noise). Trend charts (revenue/cash-flow/GST/UPI/payroll over time) are
reconstructed from each snapshot's growth-rate and seasonality-index fields
since the dataset has no monthly history — see the note in
`docs/API_DESIGN.md` §6.

## GitHub / deployment

Share the GitHub repo URL for this project once pushed, and optionally a
deployed link (e.g., the API on Render/Fly.io and the dashboard on
Streamlit Community Cloud) per the assignment's submission note.
