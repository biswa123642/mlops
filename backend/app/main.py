import os

import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)


# ============================================================
# Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

REGISTERED_MODEL_NAME = os.getenv(
    "REGISTERED_MODEL_NAME",
    "customer-churn-model",
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "Production",
)

# MLflow Model Registry URI using alias
#
# Example:
# models:/customer-churn-model@Production
#
# This means the backend always loads
# whichever model version currently has
# the Production alias.
MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "API for predicting customer churn "
        "using an MLflow registered model."
    ),
    version="1.0.0",
)


# ============================================================
# CORS Configuration
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Global Model
# ============================================================

model = None


# ============================================================
# Load MLflow Model
# ============================================================

def load_model():
    """
    Load the model from MLflow Model Registry
    using the configured model alias.
    """

    global model

    print(
        "========================================"
    )

    print(
        "Loading MLflow model..."
    )

    print(
        f"MLflow Tracking URI: "
        f"{MLFLOW_TRACKING_URI}"
    )

    print(
        f"Registered Model Name: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Model Alias: "
        f"{MODEL_ALIAS}"
    )

    print(
        f"Model URI: "
        f"{MODEL_URI}"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    # --------------------------------------------------------
    # Load registered model
    #
    # Example:
    #
    # models:/customer-churn-model@Production
    #
    # The Production alias points to the current
    # production model version.
    # --------------------------------------------------------

    model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    print(
        "MLflow model loaded successfully."
    )

    print(
        "========================================"
    )


# ============================================================
# Startup Event
# ============================================================

@app.on_event("startup")
def startup_event():
    """
    Load the MLflow model when the FastAPI
    application starts.
    """

    load_model()


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check():
    """
    Check whether the API is running
    and the MLflow model is loaded.
    """

    if model is None:

        return {
            "status": "error",
            "model_loaded": False,
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": REGISTERED_MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):
    """
    Predict whether a customer will churn.
    """

    # --------------------------------------------------------
    # Check if model is available
    # --------------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    # --------------------------------------------------------
    # Convert request to DataFrame
    #
    # The MLflow model contains the complete preprocessing
    # pipeline, including:
    #
    # - Numeric preprocessing
    # - Categorical preprocessing
    # - OneHotEncoder
    # - RandomForestClassifier
    #
    # Therefore, the API sends the input data directly
    # to the MLflow model.
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "tenure": request.tenure,
                "monthly_charges": request.monthly_charges,
                "support_calls": request.support_calls,
                "contract_type": request.contract_type,
                "internet_service": request.internet_service,
            }
        ]
    )

    # --------------------------------------------------------
    # Make Prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )

    except Exception as exception:

        print(
            f"Prediction failed: {exception}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Model prediction failed."
            ),
        )

    # --------------------------------------------------------
    # Convert Prediction
    # --------------------------------------------------------

    prediction_value = int(
        prediction[0]
    )

    # --------------------------------------------------------
    # Return Prediction Response
    # --------------------------------------------------------

    return PredictionResponse(
        prediction=prediction_value,
        churn=bool(
            prediction_value
        ),
    )