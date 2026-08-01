from pydantic import BaseModel, Field


# ============================================================
# Prediction Request
# ============================================================

class PredictionRequest(BaseModel):
    """
    Input data required for customer churn prediction.
    """

    tenure: int = Field(
        ...,
        description=(
            "Number of months the customer "
            "has been with the company."
        ),
        ge=0,
    )

    monthly_charges: float = Field(
        ...,
        description=(
            "Customer's monthly charges."
        ),
        ge=0,
    )

    support_calls: int = Field(
        ...,
        description=(
            "Number of support calls made "
            "by the customer."
        ),
        ge=0,
    )

    contract_type: str = Field(
        ...,
        description=(
            "Customer's contract type."
        ),
    )

    internet_service: str = Field(
        ...,
        description=(
            "Customer's internet service type."
        ),
    )


# ============================================================
# Prediction Response
# ============================================================

class PredictionResponse(BaseModel):
    """
    Response returned by the prediction API.
    """

    prediction: int = Field(
        ...,
        description=(
            "Predicted churn class. "
            "0 means no churn and 1 means churn."
        ),
    )

    churn: bool = Field(
        ...,
        description=(
            "Whether the customer is predicted "
            "to churn."
        ),
    )