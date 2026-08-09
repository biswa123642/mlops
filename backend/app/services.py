import requests

from app.config import (
    AZURE_ML_API_KEY,
    AZURE_ML_ENDPOINT,
    REQUEST_TIMEOUT,
)


class PredictionService:
    @staticmethod
    def predict(data: dict) -> dict:
        headers = {
            "Authorization": (
                f"Bearer {AZURE_ML_API_KEY}"
            ),
            "Content-Type": "application/json",
        }

        payload = {
            "data": [data],
        }

        response = requests.post(
            url=AZURE_ML_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()