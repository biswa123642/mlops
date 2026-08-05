import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictionRequest, PredictionResponse


KSERVE_BASE_URL = os.getenv(
    "KSERVE_BASE_URL",
    "http://customer-churn.mlops.svc.cluster.local",
)
MODEL_NAME = os.getenv("MODEL_NAME", "customer-churn")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10.0"))

CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "http://localhost:5173")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_RAW.split(",") if origin.strip()]

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    yield
    await http_client.aclose()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using KServe.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    if http_client is None:
        return {
            "status": "error",
            "model_loaded": False,
            "error": "HTTP client not ready.",
        }

    try:
        resp = await http_client.get(f"{KSERVE_BASE_URL}/v1/models/{MODEL_NAME}")
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


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if http_client is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    payload = {
        "instances": [
            {
                "tenure": request.tenure,
                "monthly_charges": request.monthly_charges,
                "support_calls": request.support_calls,
                "contract_type": request.contract_type,
                "internet_service": request.internet_service,
            }
        ]
    }

    try:
        resp = await http_client.post(
            f"{KSERVE_BASE_URL}/v1/models/{MODEL_NAME}:predict",
            json=payload,
        )
        resp.raise_for_status()
        result = resp.json()
        print("KServe raw response:", result)
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"KServe prediction failed: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"KServe unreachable: {e}",
        )

    predictions = result.get("predictions") or result.get("outputs")
    if predictions is None:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected KServe response format: {result}",
        )

    first = predictions[0]
    if isinstance(first, dict) and "data" in first:
        data = first["data"]
        first = data[0] if isinstance(data, list) else data
    elif isinstance(first, list):
        first = first[0]

    try:
        prediction_value = int(first)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to parse prediction from KServe response: {result}",
        )

    return PredictionResponse(
        prediction=prediction_value,
        churn=bool(prediction_value),
    )
