# Integration Strategy

How the Financial Health Report plugs into India's digital public
infrastructure (DPI) and existing lending rails. This repo implements the
scoring engine and the REST surface it needs; the sections below describe
the upstream/downstream contracts each integration would use.

## Account Aggregator (AA)

- **Direction:** upstream (data source).
- **What it feeds:** `Monthly_Bank_Credits_INR`, `Monthly_Bank_Debits_INR`,
  `Average_Bank_Balance_INR`, `Cashflow_Stability_Index`,
  `Overdraft_Usage_Ratio`, `Transaction_Volatility_Index`,
  `Has_Existing_Loan` / `Monthly_Loan_EMI_INR` / `EMI_On_Time_Rate_Pct`.
- **Flow:** the lender (Financial Information User, FIU) raises a consent
  request via an AA (e.g., Setu, Finvu, OneMoney); once the MSME approves in
  their AA app, encrypted FI (Financial Information) data — bank statements —
  is pulled and decrypted at the FIU. A translation layer parses the
  AA-standard FI Data XML/JSON into the flat banking fields
  `MSMERawInput` expects, then calls `POST /api/v1/msme/ingest`.
- **Consent lifecycle:** consent has a purpose code, data range, and expiry —
  the ingestion layer should store the consent-artifact ID alongside the raw
  record for audit, and re-request consent on expiry before re-scoring.

## GSTN

- **Direction:** upstream (data source).
- **What it feeds:** `Monthly_GST_Sales_INR`, `Monthly_GST_Purchases_INR`,
  `GST_Filing_Timeliness_Pct`, `GST_Return_Frequency`.
- **Flow:** GSTN's GSP (GST Suvidha Provider) APIs (or the direct GSTN API
  once RBI/GSTN data-sharing consent is in place) return GSTR-1/3B filing
  history and turnover; the same translation layer maps these into the GST
  fields above before ingestion. `GST_Filing_Timeliness_Pct` is derived from
  the gap between due date and actual filing date across recent return
  periods.

## UPI / NPCI

- **Direction:** upstream (data source).
- **What it feeds:** `Monthly_UPI_Inflow_INR`, `Monthly_UPI_Outflow_INR`,
  `UPI_Avg_Ticket_Size_INR`, `UPI_Daily_Txn_Count`.
- **Flow:** UPI transaction history is typically obtained either through the
  AA framework (if the MSME's PSP participates as an FIP) or through a
  merchant-side PSP/payment-aggregator API the business already uses (e.g.
  their UPI QR collections dashboard) with the business's explicit consent.

## EPFO

- **Direction:** upstream (data source).
- **What it feeds:** `Monthly_Payroll_INR`, `Salary_Consistency_Pct`,
  `Employee_Attrition_Rate_Pct`, and indirectly `Employee_Count`.
- **Flow:** EPFO employer login / API access (where available) or
  digilocker-style consent-based document pull of ECR (Electronic Challan
  cum Return) filings, translated into monthly payroll aggregates.

## Invoicing / receivables & vendor systems

- **Direction:** upstream (data source), typically via the MSME's accounting
  software (Tally, Zoho Books, Vyapar) through a consent-based Account
  Aggregator-style connector or a direct OAuth integration.
- **What it feeds:** `Avg_Invoice_Payment_Delay_Days`,
  `Customer_Concentration_Ratio`, `Vendor_Payment_Timeliness_Pct`.

## OCEN (Open Credit Enablement Network)

- **Direction:** downstream (credit distribution).
- **Role of this system:** OCEN standardizes the lender (LSP/LSP-lender)
  handshake for loan origination. This system acts as the **Financial
  Health / underwriting service** a Lending Service Provider (LSP) calls
  before submitting a loan application to a lender on OCEN:
  1. LSP collects consent + alternative data from the MSME.
  2. LSP calls `POST /api/v1/score` (or ingest + `GET /api/v1/health-card/{id}`)
     to get the Financial Health Card, risk category, PD, and recommended
     limit.
  3. LSP attaches the card (score + explainability) as supporting
     underwriting evidence in the OCEN loan application payload.
  4. The lender's own credit policy engine can consume
     `GET /api/v1/credit-recommendation/{id}` directly as one input signal
     alongside its own bureau/policy checks.
- **Contract:** the REST responses in `api/schemas.py` are intentionally flat
  JSON (no nested proprietary types) so they map cleanly onto an OCEN loan
  application's "alternate data assessment" section.

## ULI (Unified Lending Interface)

- **Direction:** downstream (credit distribution) + upstream (additional data
  sources).
- **Role of this system:** ULI (RBI's platform, evolved from the erstwhile
  Public Tech Platform for Frictionless Credit) standardizes pulls from land
  records, satellite data, dairy cooperative data, etc., in addition to the
  financial data sources above. As ULI expands the set of connected data
  sources, this system's ingestion schema (`MSMERawInput`) is the extension
  point — new alternative-data fields become new optional fields on that
  schema and new features in `src/data_pipeline.py`, without changing the
  scoring contract downstream. On the distribution side, a lender on ULI can
  call this system's scoring API the same way an OCEN-connected LSP does.

## Banking APIs (core banking / lender-side)

- **Direction:** downstream (decisioning) + upstream (existing-customer data
  for `Existing-to-Credit` segment MSMEs already banked with the lender).
- **Flow:** a lender's core banking / LOS system calls
  `GET /api/v1/credit-recommendation/{msme_id}` at the point of underwriting,
  and can pass back the actual sanctioned amount for model monitoring
  (planned extension: a `POST /api/v1/feedback` endpoint to close the loop
  between recommended and sanctioned/disbursed amounts for periodic
  recalibration).

## Other consent-based DPI

The architecture treats every new alternative-data source the same way:
land it as raw fields on the ingestion schema, add any derived ratio
features, and retrain — the six-dimension score design means a new data
source usually strengthens one or two existing dimensions (e.g., digital
invoicing data strengthens Payment Behaviour and Revenue Consistency) rather
than requiring a new dimension or a full model redesign.

## Security & compliance notes

- All source integrations above are consent-based per the DEPA (Data
  Empowerment and Protection Architecture) principles underlying AA/ULI/OCEN
  — this system never pulls data without an explicit, auditable consent
  artifact.
- Raw ingested data should be encrypted at rest and access-scoped per
  MSME/lender tenant in a production deployment (the prototype's in-memory
  store is for demo purposes only — see `docs/ARCHITECTURE.md`).
- Explainability outputs (SHAP drivers) should be retained alongside each
  credit decision for RBI's fair-lending / adverse-action-notice
  requirements.
