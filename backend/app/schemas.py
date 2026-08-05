from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# Allowed Values (Enums via Literal)
# ============================================================


ContractType = Literal["Month-to-month", "One year", "Two year"]
InternetService = Literal["DSL", "Fiber optic", "No"]


# ============================================================
# Prediction Request
# ============================================================


class PredictionRequest(BaseModel):
    """Input data required for customer churn prediction."""

    tenure: int = Field(
        ...,
        description="Number of months the customer has been with the company.",
        ge=0,
    )

    monthly_charges: float = Field(
        ...,
        description="Customer's monthly charges.",
        ge=0.0,
    )

    support_calls: int = Field(
        ...,
        description="Number of support calls made by the customer.",
        ge=0,
    )

    contract_type: ContractType = Field(
        ...,
        description="Customer's contract type.",
    )

    internet_service: InternetService = Field(
        ...,
        description="Customer's internet service type.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 12,
                    "monthly_charges": 65.5,
                    "support_calls": 2,
                    "contract_type": "Month-to-month",
                    "internet_service": "Fiber optic",
                }
            ]
        }
    }


# ============================================================
# Prediction Response
# ============================================================


class PredictionResponse(BaseModel):
    """Response returned by the prediction API."""

    prediction: int = Field(
        ...,
        description="Predicted churn class. 0 means no churn and 1 means churn.",
    )

    churn: bool = Field(
        ...,
        description="Whether the customer is predicted to churn.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": 1,
                    "churn": True,
                }
            ]
        }
    }