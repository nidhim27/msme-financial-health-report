"""
REST API for the Cars24 MSME Financial Health Report.

Endpoints map directly to the assignment's "API Design" deliverable:
  - Data ingestion         POST /api/v1/msme/ingest
  - Score generation       POST /api/v1/score
  - Health Card retrieval  GET  /api/v1/health-card/{msme_id}
  - Credit recommendation  GET  /api/v1/credit-recommendation/{msme_id}
  - Dashboard integration  GET  /api/v1/dashboard/{msme_id}
  - Explainability         GET  /api/v1/explain/{msme_id}

Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.schemas import (
    CreditRecommendationResponse, HealthCardResponse, IngestResponse, MSMERawInput,
)
from src.health_card import generate_health_card, generate_health_card_for_id, get_raw_record, list_msme_ids
from src.trends import build_trend_bundle

app = FastAPI(
    title="Cars24 MSME Financial Health Report API",
    description="Alternative-data-driven Financial Health Card, credit scoring, and explainability for MSMEs.\n\n"
                "Created by Nidhi Mehra",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# In-memory store for records ingested at runtime (not part of the seed dataset).
# A production deployment persists this to a database (see docs/ARCHITECTURE.md).
_INGESTED_RECORDS: dict[str, dict] = {}


def _next_synthetic_id() -> str:
    return f"MSME-NEW-{len(_INGESTED_RECORDS) + 1:06d}"


def _resolve_raw_record(msme_id: str) -> dict:
    if msme_id in _INGESTED_RECORDS:
        return _INGESTED_RECORDS[msme_id]
    try:
        return get_raw_record(msme_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MSME_ID '{msme_id}' not found")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/healthz", tags=["system"])
def healthz():
    return {"status": "ok"}


@app.post("/api/v1/msme/ingest", response_model=IngestResponse, tags=["ingestion"])
def ingest_msme(payload: MSMERawInput):
    """Ingest a consent-based raw data record for one MSME (GSTN + UPI + AA +
    EPFO + banking, already aggregated upstream). Stores it for later scoring
    and retrieval; does not compute the score itself (see /api/v1/score)."""
    msme_id = payload.msme_id or _next_synthetic_id()
    record = payload.model_dump(exclude={"msme_id"})
    _INGESTED_RECORDS[msme_id] = record
    return IngestResponse(msme_id=msme_id, status="ingested", message="Record stored; call /api/v1/score to generate the Financial Health Card.")


@app.post("/api/v1/score", response_model=HealthCardResponse, tags=["scoring"])
def score_msme(payload: MSMERawInput, explain: bool = Query(True, description="Include SHAP explainability")):
    """Generate a full Financial Health Card in real time for a raw MSME record
    (ingest + score in a single call)."""
    msme_id = payload.msme_id or _next_synthetic_id()
    record = payload.model_dump(exclude={"msme_id"})
    _INGESTED_RECORDS[msme_id] = record
    card = generate_health_card(record, msme_id=msme_id, explain=explain)
    return card


@app.get("/api/v1/health-card/{msme_id}", response_model=HealthCardResponse, tags=["scoring"])
def get_health_card(msme_id: str, explain: bool = Query(True, description="Include SHAP explainability")):
    """Retrieve (recompute on the fly from stored raw data) the Financial
    Health Card for a known MSME_ID — from the seed dataset or a previously
    ingested record."""
    record = _resolve_raw_record(msme_id)
    return generate_health_card(record, msme_id=msme_id, explain=explain)


@app.get("/api/v1/credit-recommendation/{msme_id}", response_model=CreditRecommendationResponse, tags=["scoring"])
def get_credit_recommendation(msme_id: str):
    record = _resolve_raw_record(msme_id)
    card = generate_health_card(record, msme_id=msme_id, explain=False)
    return CreditRecommendationResponse(
        msme_id=msme_id,
        credit_eligible=card["credit_eligible"],
        credit_risk_category=card["credit_risk_category"],
        probability_of_default=card["probability_of_default"],
        recommended_credit_limit_inr=card["recommended_credit_limit_inr"],
    )


@app.get("/api/v1/explain/{msme_id}", tags=["explainability"])
def explain_msme(msme_id: str):
    record = _resolve_raw_record(msme_id)
    card = generate_health_card(record, msme_id=msme_id, explain=True)
    return card["explainability"]


@app.get("/api/v1/msme/search", tags=["ingestion"])
def search_msme(q: Optional[str] = Query(None, description="Substring match on MSME_ID"), limit: int = 25):
    ids = list_msme_ids(limit=50000)
    if q:
        ids = [i for i in ids if q.lower() in i.lower()]
    return {"count": len(ids), "results": ids[:limit]}


@app.get("/api/v1/dashboard/{msme_id}", tags=["dashboard"])
def dashboard_data(msme_id: str):
    """Aggregated payload the dashboard needs: health card + simulated trend
    series for revenue, cash flow, GST, UPI, and payroll."""
    record = _resolve_raw_record(msme_id)
    card = generate_health_card(record, msme_id=msme_id, explain=True)
    trends = build_trend_bundle(msme_id, record)
    return {"health_card": card, "trends": trends, "generated_on": str(date.today())}
