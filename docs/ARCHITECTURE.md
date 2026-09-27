# Solution Architecture — Cars24 MSME Financial Health Report

## 1. End-to-end system architecture

```mermaid
flowchart LR
    subgraph Sources["Consent-based alternative-data sources"]
        GSTN[GSTN\nGST returns & sales/purchases]
        UPI[UPI / NPCI\nPayment history]
        AA[Account Aggregator\nBank statements]
        EPFO[EPFO\nPayroll data]
        INV[Invoicing / receivables\nvendor systems]
    end

    subgraph Ingestion["Data ingestion layer"]
        CONSENT[Consent & auth manager\n(AA consent artifact, OAuth for GSTN/EPFO)]
        API_ING[Ingestion REST API\nPOST /api/v1/msme/ingest]
        VALID[Schema validation\n(pydantic)]
        STORE[(Raw record store\nOLTP / document DB)]
    end

    subgraph Feature["Feature engineering"]
        FE[src/data_pipeline.py\nratio features + encoding]
        FSTORE[(Feature store\nversioned schema)]
    end

    subgraph MLLayer["AI / ML scoring layer"]
        SUB[6x sub-score regressors\nXGBoost]
        OVR[Overall score regressor]
        PD[PD regressor]
        RISK[Risk-category classifier]
        ELIG[Eligibility classifier]
        LIMIT[Credit-limit regressor]
        ANOM[Anomaly detector\nIsolationForest]
        SHAP[SHAP explainer\nTreeExplainer]
    end

    subgraph Serving["Serving layer"]
        CARD[Health-card engine\nsrc/health_card.py]
        SCORE_API[Scoring REST API\napi/main.py]
        DASH[Dashboard\nStreamlit]
    end

    subgraph Consumers
        LOS[Loan Origination System]
        UW[Underwriter workbench]
        ULI_OCEN[ULI / OCEN loan marketplace]
    end

    GSTN --> CONSENT
    UPI --> CONSENT
    AA --> CONSENT
    EPFO --> CONSENT
    INV --> CONSENT
    CONSENT --> API_ING --> VALID --> STORE
    STORE --> FE --> FSTORE --> SUB & OVR & PD & RISK & ELIG & LIMIT & ANOM
    SUB & OVR & PD & RISK & ELIG & LIMIT & ANOM --> CARD
    SHAP --> CARD
    CARD --> SCORE_API --> DASH
    SCORE_API --> LOS
    SCORE_API --> UW
    SCORE_API --> ULI_OCEN
```

The prototype in this repo implements every box above except `CONSENT`
(consent capture/AA XML parsing) and the production data stores, which are
stubbed as an in-memory dict (`api/main.py::_INGESTED_RECORDS`) and a flat
CSV (`data/msme_synthetic_50k.csv`) respectively — swapping those for a real
AA client and Postgres/feature-store tables is the only change needed to go
from prototype to production, because the scoring code (`src/health_card.py`)
already treats "raw record in, health card out" as its only contract.

## 2. Data ingestion pipeline

1. **Consent capture** — an MSME (or the lending institution on its behalf)
   grants consent via the Account Aggregator framework (for bank data), GSTN
   OAuth (for GST returns), EPFO login (for payroll), and UPI-handle-linked
   PSP data-sharing consent.
2. **Aggregation** — a consent orchestrator (outside this repo's scope, but
   modeled by `api/schemas.py::MSMERawInput`) pulls raw statements/returns
   from each source and normalizes them into the flat feature set this system
   expects: business profile, GST, UPI, banking, payroll, invoicing/vendor,
   and trend fields (28 raw fields, see `src/data_pipeline.py`).
3. **Ingestion API** — `POST /api/v1/msme/ingest` validates the payload
   against a strict schema (`MSMERawInput`, `pydantic`) and persists it.
4. **Feature engineering** — `src/data_pipeline.py::engineer_features` derives
   9 underwriting ratios (GST sales/purchase ratio, UPI net flow ratio, bank
   net flow ratio, EMI-to-credit ratio, payroll-to-turnover ratio, turnover
   per employee, etc.) on top of the 27 raw numeric fields, then one-hot
   encodes the 6 categorical fields against a fixed schema
   (`models/feature_schema.json`) so a single new record always encodes to
   the same columns the models were trained on.

## 3. AI/ML workflow

| Stage | Model | Target | Technique |
|---|---|---|---|
| Dimension scoring | 6× `XGBRegressor` | Business Stability, Cashflow, Revenue Consistency, Payment Behaviour, Business Growth, Compliance (0–100 each) | Gradient-boosted trees, independent models so each dimension's drivers stay separable for explainability |
| Overall scoring | `XGBRegressor` | Financial Health Score (0–100) | Direct model on raw+engineered features (not a re-aggregation of the sub-scores — see `docs/MODEL_REPORT.md` for why) |
| Default risk | `XGBRegressor` | Probability of Default | Continuous regression, clipped to [0,1] |
| Risk segmentation | `XGBClassifier` (3-class) | Credit Risk Category (Low/Medium/High) | `multi:softprob` |
| Credit decisioning | `XGBClassifier` (binary) | Credit Eligible (Yes/No) | binary logistic |
| Credit sizing | `XGBRegressor` | Recommended Credit Limit (INR) | trained on the eligible subset only, `log1p`-transformed target, capped at annual turnover at inference |
| Anomaly detection | `IsolationForest` (unsupervised) | anomaly flag + 0–100 anomaly score | trained on transactional/behavioural features only (volatility, overdraft usage, filing timeliness, attrition, invoice delay, etc.) — there is no labelled "anomaly" column, by design, so this stays unsupervised |
| Explainability | `shap.TreeExplainer` | per-record feature attributions | exact (non-approximated) Shapley values for every XGBoost model above, reused for the overall score, risk category, and eligibility decision |

All models are trained by `python -m src.train` (80/20 split, stratified by
risk category, `random_state=42`) and persisted with `joblib` to `models/`.
`python -m src.anomaly` trains the anomaly detector, and
`python -m src.batch_score` pre-scores the full population for fast
dashboard/API lookups (predictions only — SHAP stays on-demand per record for
latency reasons). See `docs/MODEL_REPORT.md` for accuracy/R²/F1 numbers.

## 4. Integration approach

The scoring engine (`src/health_card.py::generate_health_card`) is the single
seam between "raw alternative data in" and "Financial Health Card out". It is
exposed three ways so it can slot into different institutional workflows:

- **REST API** (`api/main.py`) — for LOS/underwriting-workbench integration,
  ULI/OCEN loan-market participation, or any system-to-system call.
- **Dashboard** (`dashboard/app.py`) — for a credit analyst / relationship
  manager to review a card interactively, including on ad-hoc "what-if" new
  MSME data.
- **Batch** (`src/batch_score.py`) — for portfolio-level monitoring /
  re-scoring on a schedule (e.g., nightly re-score of the active book to
  refresh risk categories from a batch data refresh).

See `docs/INTEGRATION_STRATEGY.md` for how each upstream/downstream system
(ULI, OCEN, Account Aggregator, GSTN, banking APIs) plugs into this seam.

## 5. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Feature engineering / data | pandas, numpy | standard, fast enough at this scale (50k rows in-memory) |
| ML models | XGBoost (`XGBRegressor`/`XGBClassifier`) | strong tabular performance, native missing-value handling (important — `EMI_On_Time_Rate_Pct` is legitimately null for no-loan MSMEs), single explainability strategy (TreeExplainer) across every model |
| Anomaly detection | scikit-learn `IsolationForest` | unsupervised, robust to the mixed-scale transactional features, no labelled anomalies needed |
| Explainability | SHAP (`TreeExplainer`) | exact Shapley values for tree ensembles, fast enough for real-time per-record explanation |
| API | FastAPI + Pydantic + uvicorn | async-ready, automatic OpenAPI docs (`/docs`), strong request validation for the ingestion schema |
| Dashboard | Streamlit + Plotly | fastest path to an interactive analyst-facing UI backed directly by the Python scoring code, no separate frontend build needed |
| Model persistence | joblib | simplest reliable serialization for scikit-learn/XGBoost estimators |
| Deployment target | Docker image(s) for API + dashboard, deployable to any container platform (Render/Fly.io/AWS ECS/Azure Container Apps) | keeps the stack cloud-agnostic; no vendor lock-in |

## 6. Scalability & reliability notes

- Model inference on a single record is sub-50ms (tree ensembles, no deep
  learning); the batch scorer scores all 50,000 MSMEs in a few seconds,
  so the design scales horizontally by simply running more API replicas
  behind a load balancer — there is no shared mutable state except the
  (production-swappable) ingestion store.
- The feature schema is frozen at training time
  (`models/feature_schema.json`) and every inference-time record is
  reindexed against it — this makes the API robust to partial/out-of-order
  fields and prevents silent train/serve skew.
- SHAP explanation is computed on-demand per request rather than batch —
  it's the one part of the pipeline that doesn't trivially scale to 50k
  rows at once, so it's deliberately kept off the batch path.
