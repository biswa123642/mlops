import json
import logging
import os

import joblib
import pandas as pd


logger = logging.getLogger(__name__)

model = None


FEATURE_COLUMNS = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract_type",
    "internet_service",
]


def init():
    global model

    model_dir = os.getenv("AZUREML_MODEL_DIR")

    if not model_dir:
        raise RuntimeError(
            "AZUREML_MODEL_DIR environment variable is not set."
        )

    model_path = os.path.join(
        model_dir,
        "model",
        "model.pkl",
    )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file was not found at: {model_path}"
        )

    model = joblib.load(model_path)

    logger.info(
        "Customer churn model loaded from %s",
        model_path,
    )


def parse_request(raw_data):
    if isinstance(raw_data, bytes):
        raw_data = raw_data.decode("utf-8")

    if isinstance(raw_data, str):
        payload = json.loads(raw_data)

    elif isinstance(raw_data, dict):
        payload = raw_data

    elif isinstance(raw_data, list):
        payload = raw_data

    else:
        raise ValueError(
            "Request must be JSON text, a dictionary, or a list."
        )

    if isinstance(payload, dict):
        if "data" not in payload:
            raise ValueError(
                "Request object must contain a 'data' field."
            )

        records = payload["data"]

    else:
        records = payload

    if not isinstance(records, list):
        raise ValueError(
            "The 'data' field must contain a list of records."
        )

    if not records:
        raise ValueError(
            "The input data list cannot be empty."
        )

    return records


def validate_and_prepare_data(records):
    dataframe = pd.DataFrame(records)

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    dataframe = dataframe[FEATURE_COLUMNS].copy()

    if dataframe.isnull().any().any():
        raise ValueError(
            "Input contains missing values."
        )

    return dataframe


def get_model_classes():
    if hasattr(model, "classes_"):
        return model.classes_

    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")

        if classifier is not None and hasattr(
            classifier,
            "classes_",
        ):
            return classifier.classes_

    return None


def run(raw_data):
    global model

    if model is None:
        raise RuntimeError(
            "Model has not been loaded. init() was not called."
        )

    try:
        records = parse_request(raw_data)

        dataframe = validate_and_prepare_data(records)

        predictions = model.predict(dataframe)

        response = {
            "prediction": predictions.tolist(),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(dataframe)

            response["probability"] = probabilities.tolist()

            classes = get_model_classes()

            if classes is not None:
                classes = list(classes)

                if 1 in classes:
                    churn_class_index = classes.index(1)

                    response["churn_probability"] = (
                        probabilities[
                            :,
                            churn_class_index,
                        ].tolist()
                    )

        return response

    except Exception:
        logger.exception(
            "Customer churn prediction failed."
        )

        raise