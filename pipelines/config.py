import os

# -----------------------------
# Azure Storage
# -----------------------------

AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")

AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")

AZURE_STORAGE_CONNECTION_STRING = os.getenv(
    "AZURE_STORAGE_CONNECTION_STRING"
)

DATASET_NAME = os.getenv(
    "DATASET_NAME",
    "customer_churn.csv",
)

# -----------------------------
# MLflow
# -----------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT",
    "customer-churn",
)

REGISTERED_MODEL_NAME = os.getenv(
    "REGISTERED_MODEL_NAME",
    "customer-churn-model",
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "Production",
)

# -----------------------------
# KServe
# -----------------------------

KSERVE_NAMESPACE = os.getenv(
    "KSERVE_NAMESPACE",
    "mlops",
)

INFERENCE_SERVICE_NAME = os.getenv(
    "INFERENCE_SERVICE_NAME",
    "customer-churn",
)