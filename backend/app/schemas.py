from pydantic import BaseModel, ConfigDict, Field


class ChurnPredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    tenure: int = Field(
        ...,
        ge=0,
        description="Customer tenure in months",
    )

    monthly_charges: float = Field(
        ...,
        ge=0,
        description="Customer monthly charges",
    )

    support_calls: int = Field(
        ...,
        ge=0,
        description="Number of customer support calls",
    )

    contract_type: int = Field(
        ...,
        ge=0,
        le=2,
        description="Encoded contract type: 0, 1, or 2",
    )

    internet_service: int = Field(
        ...,
        ge=0,
        le=1,
        description="Encoded internet service: 0 or 1",
    )


class ChurnPredictionResponse(BaseModel):
    prediction: list[int] = Field(
        ...,
        description="Predicted churn class",
    )

    probability: list[list[float]] = Field(
        ...,
        description="Probability for each model class",
    )

    churn_probability: list[float] = Field(
        ...,
        description="Probability of churn class 1",
    )