import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Azure Storage
# -----------------------------
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")

AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")

AZURE_STORAGE_ACCESS_KEY = os.getenv("AZURE_STORAGE_ACCESS_KEY")

AZURE_STORAGE_SAS_TOKEN = os.getenv("AZURE_STORAGE_SAS_TOKEN")

DATASET_NAME = os.getenv(
    "DATASET_NAME",
    "customer_churn.csv",
)

# -----------------------------
# MLflow
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

MLFLOW_USERNAME = os.getenv(
    "MLFLOW_TRACKING_USERNAME", os.getenv("MLFLOW_USERNAME")
)

MLFLOW_PASSWORD = os.getenv(
    "MLFLOW_TRACKING_PASSWORD", os.getenv("MLFLOW_PASSWORD")
)

ML_PIPELINE_BASE_IMAGE = os.getenv("ML_PIPELINE_BASE_IMAGE")

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
# Kubeflow
# -----------------------------
KUBEFLOW_ENDPOINT = os.getenv("KUBEFLOW_ENDPOINT")

KUBEFLOW_USERNAME = os.getenv("KUBEFLOW_USERNAME")

KUBEFLOW_PASSWORD = os.getenv("KUBEFLOW_PASSWORD")

KUBEFLOW_USER_NAMESPACE = os.getenv(
    "KUBEFLOW_USER_NAMESPACE",
    "kubeflow-user-example-com",
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

KSERVE_SERVICE_ACCOUNT = os.getenv(
    "KSERVE_SERVICE_ACCOUNT",
    "kserve-sa",
)

KSERVE_SECRET_NAME = os.getenv(
    "KSERVE_SECRET_NAME",
    "azure-storage-secret",
)

# =========================================================
# Model Evaluation Thresholds
# =========================================================
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.80"))

MIN_PRECISION = float(os.getenv("MIN_PRECISION", "0.75"))

MIN_RECALL = float(os.getenv("MIN_RECALL", "0.75"))

MIN_F1 = float(os.getenv("MIN_F1", "0.75"))