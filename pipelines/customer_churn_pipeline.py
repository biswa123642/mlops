from kfp import dsl

from components.deploy_kserve_component import create_deploy_kserve_component
from components.evaluate_component import create_evaluate_component
from components.register_model_component import create_register_model_component
from components.train_component import create_train_component

# Import configuration defaults from config.py
from config import (
    AZURE_STORAGE_ACCOUNT,
    AZURE_STORAGE_ACCESS_KEY,
    AZURE_STORAGE_CONTAINER,
    DATASET_NAME,
    EXPERIMENT_NAME,
    INFERENCE_SERVICE_NAME,
    KSERVE_NAMESPACE,
    KSERVE_SECRET_NAME,
    KSERVE_SERVICE_ACCOUNT,
    MIN_ACCURACY,
    MIN_F1,
    MIN_PRECISION,
    MIN_RECALL,
    ML_PIPELINE_BASE_IMAGE,
    MLFLOW_PASSWORD,
    MLFLOW_TRACKING_URI,
    MLFLOW_USERNAME,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
    AZURE_STORAGE_SAS_TOKEN,
)

# =========================================================
# Dynamic Base Image Setup
# =========================================================
BASE_IMAGE = ML_PIPELINE_BASE_IMAGE

# =========================================================
# Component Factory Instantiations
# =========================================================
train_component = create_train_component(base_image=BASE_IMAGE)
evaluate_component = create_evaluate_component(base_image=BASE_IMAGE)
register_model_component = create_register_model_component(base_image=BASE_IMAGE)
deploy_kserve_component = create_deploy_kserve_component(base_image=BASE_IMAGE)


@dsl.pipeline(
    name="customer-churn-ml-pipeline",
    description=(
        "Production ML pipeline for training, evaluating, "
        "registering, and deploying the customer churn model."
    ),
)
def customer_churn_pipeline(
    # ---------------------------------------------------------
    # Azure Blob Storage Parameters
    # ---------------------------------------------------------
    azure_storage_account: str = AZURE_STORAGE_ACCOUNT,
    azure_storage_container: str = AZURE_STORAGE_CONTAINER,
    azure_storage_access_key: str = AZURE_STORAGE_ACCESS_KEY,
    azure_storage_sas_token: str = AZURE_STORAGE_SAS_TOKEN,
    dataset_name: str = DATASET_NAME,
    # ---------------------------------------------------------
    # MLflow Tracking & Registry Parameters
    # ---------------------------------------------------------
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    mlflow_username: str = MLFLOW_USERNAME,
    mlflow_password: str = MLFLOW_PASSWORD,
    experiment_name: str = EXPERIMENT_NAME,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    model_alias: str = MODEL_ALIAS,
    # ---------------------------------------------------------
    # Model Evaluation Thresholds
    # ---------------------------------------------------------
    min_accuracy: float = MIN_ACCURACY,
    min_precision: float = MIN_PRECISION,
    min_recall: float = MIN_RECALL,
    min_f1: float = MIN_F1,
    # ---------------------------------------------------------
    # KServe Serving Parameters
    # ---------------------------------------------------------
    namespace: str = KSERVE_NAMESPACE,
    inference_service_name: str = INFERENCE_SERVICE_NAME,
    service_account_name: str = KSERVE_SERVICE_ACCOUNT,
    kserve_secret_name: str = KSERVE_SECRET_NAME,
):
    # =========================================================
    # 1. TRAIN MODEL
    # =========================================================
    train_task = train_component(
        azure_storage_account=azure_storage_account,
        azure_storage_container=azure_storage_container,
        azure_storage_access_key=azure_storage_access_key,
        dataset_name=dataset_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_username=mlflow_username,
        mlflow_password=mlflow_password,
        experiment_name=experiment_name,
    )
    # Disable caching for training (side effects: MLflow logs, artifacts)
    train_task.set_caching_options(False)

    # =========================================================
    # 2. EVALUATE MODEL
    # =========================================================
    evaluate_task = evaluate_component(
        accuracy=train_task.outputs["accuracy_output"],
        precision=train_task.outputs["precision_output"],
        recall=train_task.outputs["recall_output"],
        f1_score=train_task.outputs["f1_output"],
        min_accuracy=min_accuracy,
        min_precision=min_precision,
        min_recall=min_recall,
        min_f1=min_f1,
    )
    # Optional: disable caching for evaluation
    evaluate_task.set_caching_options(False)

    # =========================================================
    # 3. REGISTER MODEL
    # =========================================================
    register_task = register_model_component(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_username=mlflow_username,
        mlflow_password=mlflow_password,
        azure_storage_account=azure_storage_account,
        azure_storage_access_key=azure_storage_access_key,
        registered_model_name=registered_model_name,
        run_id=train_task.outputs["run_id_output"],
        model_alias=model_alias,
    )
    # Explicit ordering: Register ONLY after evaluation passes threshold checks
    register_task.after(evaluate_task)
    # Disable caching for registration (side effects: MLflow registry)
    register_task.set_caching_options(False)

    # =========================================================
    # 4. DEPLOY MODEL TO KSERVE
    # =========================================================
    deploy_task = deploy_kserve_component(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_username=mlflow_username,
        mlflow_password=mlflow_password,
        azure_storage_account=azure_storage_account,
        azure_storage_access_key=azure_storage_sas_token,
        registered_model_name=registered_model_name,
        model_alias=model_alias,
        model_version=register_task.outputs["model_version_output"],
        model_artifact_uri=register_task.outputs["model_artifact_uri_output"],
        namespace=namespace,
        inference_service_name=inference_service_name,
        service_account_name=service_account_name,
        kserve_secret_name=kserve_secret_name,
    )
    # Explicit ordering: Deploy ONLY after model registration succeeds
    deploy_task.after(register_task)
    # Disable caching for deploy (side effects: KServe resources)
    deploy_task.set_caching_options(False)