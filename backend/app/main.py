import os
from contextlib import asynccontextmanager

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictionRequest, PredictionResponse


# ============================================================
# Configuration
# ============================================================

KSERVE_BASE_URL = os.getenv(
    "KSERVE_BASE_URL",
    "http://customer-churn-predictor-00001-private.mlops.svc.cluster.local:80",
)

MODEL_NAME = os.getenv("MODEL_NAME", "customer-churn")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10.0"))

# CORS origins as a comma-separated list
CORS_ORIGINS_RAW = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173",  # default for local dev
)

CORS_ORIGINS = [
    origin.strip()
    for origin in CORS_ORIGINS_RAW.split(",")
    if origin.strip()
]


# ============================================================
# Global HTTP client
# ============================================================

http_client: httpx.AsyncClient | None = None


# ============================================================
# Lifespan Context Manager
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    yield
    await http_client.aclose()


# ============================================================
# FastAPI Application Initialization
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using KServe.",
    version="1.0.0",
    lifespan=lifespan,
)

# ============================================================
# CORS Configuration
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health Check Endpoint
# ============================================================

@app.get("/health")
async def health_check():
    """Check whether the API is running and KServe is reachable."""
    try:
        assert http_client is not None
        resp = await http_client.get(
            f"{KSERVE_BASE_URL}/v1/models/{MODEL_NAME}",
        )
        resp.raise_for_status()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_name": MODEL_NAME,
            "kserve_url": KSERVE_BASE_URL,
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e),
        }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(request: PredictionRequest):
    """Predict whether a customer will churn via KServe."""

    if http_client is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready.",
        )

    # --------------------------------------------------------
    # Convert request to KServe format
    # --------------------------------------------------------
    input_data = {
        "instances": [
            {
                "tenure": request.tenure,
                "monthly_charges": request.monthly_charges,
                "support_calls": request.support_calls,
                "contract_type": request.contract_type,
                "internet_service": request.internet_service,
            }
        ],
    }

    # --------------------------------------------------------
    # Call KServe
    # --------------------------------------------------------
    try:
        resp = await http_client.post(
            f"{KSERVE_BASE_URL}/v1/models/{MODEL_NAME}:predict",
            json=input_data,
        )
        resp.raise_for_status()
        result = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"KServe prediction failed: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"KServe unreachable: {e}",
        )

    # --------------------------------------------------------
    # Parse KServe response
    # --------------------------------------------------------
    predictions = result.get("predictions") or result.get("outputs")
    if not predictions:
        raise HTTPException(
            status_code=500,
            detail="Unexpected KServe response format.",
        )

    prediction_value = int(predictions[0])

    return PredictionResponse(
        prediction=prediction_value,
        churn=bool(prediction_value),
    )