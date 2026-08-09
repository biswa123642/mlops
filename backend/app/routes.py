import logging

import requests
from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
)
from app.services import PredictionService


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
def health():
    return {
        "status": "Healthy",
    }


@router.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict(
    request: ChurnPredictionRequest,
):
    try:
        result = PredictionService.predict(
            request.model_dump()
        )

        return result

    except requests.exceptions.Timeout:
        logger.exception(
            "Azure ML prediction request timed out."
        )

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Prediction service timed out.",
        )

    except requests.exceptions.HTTPError:
        logger.exception(
            "Azure ML prediction request returned an HTTP error."
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Prediction service returned an error.",
        )

    except requests.exceptions.RequestException:
        logger.exception(
            "Unable to connect to Azure ML prediction service."
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to connect to prediction service.",
        )

    except Exception:
        logger.exception(
            "Unexpected error during prediction."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )