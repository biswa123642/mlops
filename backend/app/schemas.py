from pydantic import BaseModel, Field


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
    contract_type: int = Field(
        ...,
        description="Encoded contract type as a numeric value.",
        ge=0,
    )
    internet_service: int = Field(
        ...,
        description="Encoded internet service type as a numeric value.",
        ge=0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 12,
                    "monthly_charges": 65.5,
                    "support_calls": 2,
                    "contract_type": 0,
                    "internet_service": 1,
                }
            ]
        }
    }


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