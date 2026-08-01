import os
import sys
import time

from kfp import client


def get_required_env(name: str) -> str:

    value = os.getenv(name)

    if not value:

        raise ValueError(
            f"Required environment variable "
            f"'{name}' is not set."
        )

    return value


def main():

    # =========================================================
    # Kubeflow Configuration
    # =========================================================

    kubeflow_endpoint = get_required_env(
        "KUBEFLOW_ENDPOINT"
    )

    pipeline_package = os.getenv(
        "PIPELINE_PACKAGE",
        "compiled/customer_churn_pipeline.yaml",
    )

    pipeline_name = os.getenv(
        "PIPELINE_NAME",
        "customer-churn-ml-pipeline",
    )

    experiment_name = os.getenv(
        "KUBEFLOW_EXPERIMENT_NAME",
        "customer-churn-experiment",
    )

    # =========================================================
    # Pipeline Parameters
    # =========================================================

    azure_storage_account = get_required_env(
        "AZURE_STORAGE_ACCOUNT"
    )

    azure_storage_container = get_required_env(
        "AZURE_STORAGE_CONTAINER"
    )

    dataset_name = os.getenv(
        "DATASET_NAME",
        "customer_churn.csv",
    )

    mlflow_tracking_uri = get_required_env(
        "MLFLOW_TRACKING_URI"
    )

    mlflow_experiment = os.getenv(
        "MLFLOW_EXPERIMENT",
        "customer-churn",
    )

    registered_model_name = os.getenv(
        "REGISTERED_MODEL_NAME",
        "customer-churn-model",
    )

    model_alias = os.getenv(
        "MODEL_ALIAS",
        "Production",
    )

    kserve_namespace = os.getenv(
        "KSERVE_NAMESPACE",
        "mlops",
    )

    inference_service_name = os.getenv(
        "INFERENCE_SERVICE_NAME",
        "customer-churn",
    )

    # =========================================================
    # Evaluation Thresholds
    # =========================================================

    min_accuracy = float(
        os.getenv(
            "MIN_ACCURACY",
            "0.80",
        )
    )

    min_precision = float(
        os.getenv(
            "MIN_PRECISION",
            "0.75",
        )
    )

    min_recall = float(
        os.getenv(
            "MIN_RECALL",
            "0.75",
        )
    )

    min_f1 = float(
        os.getenv(
            "MIN_F1",
            "0.75",
        )
    )

    # =========================================================
    # Validate Pipeline Package
    # =========================================================

    if not os.path.exists(
        pipeline_package
    ):

        raise FileNotFoundError(
            f"Pipeline package not found: "
            f"{pipeline_package}"
        )

    print(
        "=========================================="
    )

    print(
        "Connecting to Kubeflow"
    )

    print(
        f"Endpoint: {kubeflow_endpoint}"
    )

    print(
        "=========================================="
    )

    # =========================================================
    # Create Kubeflow Client
    #
    # The self-hosted GitHub runner is expected to have
    # network access to the Kubeflow endpoint.
    # =========================================================

    kfp_client = client.Client(
        host=kubeflow_endpoint,
    )

    # =========================================================
    # Create or Get Experiment
    # =========================================================

    try:

        experiment = (
            kfp_client.create_experiment(
                name=experiment_name
            )
        )

        print(
            f"Created experiment: "
            f"{experiment.name}"
        )

    except Exception:

        print(
            f"Experiment '{experiment_name}' "
            f"may already exist."
        )

        experiment = (
            kfp_client.get_experiment(
                experiment_name=experiment_name
            )
        )

    # =========================================================
    # Upload / Update Pipeline
    # =========================================================

    print(
        "=========================================="
    )

    print(
        "Uploading Pipeline"
    )

    print(
        f"Pipeline: {pipeline_name}"
    )

    print(
        "=========================================="
    )

    try:

        existing_pipeline = (
            kfp_client.get_pipeline(
                filter=(
                    f'name="{pipeline_name}"'
                )
            )
        )

        pipeline_id = (
            existing_pipeline.pipeline_id
        )

        print(
            f"Existing pipeline found: "
            f"{pipeline_id}"
        )

        pipeline_version = (
            kfp_client.upload_pipeline_version(
                pipeline_package,
                pipeline_id=pipeline_id,
                pipeline_version_name=(
                    f"version-"
                    f"{int(time.time())}"
                ),
            )
        )

        pipeline_version_id = (
            pipeline_version.pipeline_version_id
        )

        print(
            f"Uploaded pipeline version: "
            f"{pipeline_version_id}"
        )

    except Exception:

        print(
            "Pipeline does not exist. "
            "Creating new pipeline."
        )

        pipeline = (
            kfp_client.upload_pipeline(
                pipeline_package,
                pipeline_name=pipeline_name,
            )
        )

        pipeline_id = (
            pipeline.pipeline_id
        )

        pipeline_version_id = None

        print(
            f"Created pipeline: "
            f"{pipeline_id}"
        )

    # =========================================================
    # Pipeline Run Parameters
    # =========================================================

    parameters = {

        "azure_storage_account":
            azure_storage_account,

        "azure_storage_container":
            azure_storage_container,

        "dataset_name":
            dataset_name,

        "mlflow_tracking_uri":
            mlflow_tracking_uri,

        "experiment_name":
            mlflow_experiment,

        "registered_model_name":
            registered_model_name,

        "model_alias":
            model_alias,

        "min_accuracy":
            min_accuracy,

        "min_precision":
            min_precision,

        "min_recall":
            min_recall,

        "min_f1":
            min_f1,

        "namespace":
            kserve_namespace,

        "inference_service_name":
            inference_service_name,

    }

    # =========================================================
    # Start Pipeline Run
    # =========================================================

    print(
        "=========================================="
    )

    print(
        "Starting Kubeflow Pipeline Run"
    )

    print(
        "=========================================="
    )

    run = kfp_client.create_run_from_pipeline_package(

        pipeline_file=pipeline_package,

        arguments=parameters,

        experiment_name=experiment_name,

        run_name=(
            f"customer-churn-"
            f"{int(time.time())}"
        ),

    )

    run_id = (
        run.run_id
    )

    print(
        f"Pipeline Run ID: {run_id}"
    )

    # =========================================================
    # Wait for Pipeline Completion
    # =========================================================

    print(
        "Waiting for pipeline to complete..."
    )

    while True:

        run_details = (
            kfp_client.get_run(
                run_id
            )
        )

        state = (
            run_details.run.status
        )

        print(
            f"Pipeline Status: {state}"
        )

        if state in [
            "SUCCEEDED",
            "FAILED",
            "ERROR",
            "CANCELED",
        ]:

            break

        time.sleep(
            30
        )

    # =========================================================
    # Final Status
    # =========================================================

    print(
        "=========================================="
    )

    print(
        f"Pipeline Finished: {state}"
    )

    print(
        f"Run ID: {run_id}"
    )

    print(
        "=========================================="
    )

    if state != "SUCCEEDED":

        print(
            "ML Pipeline failed."
        )

        sys.exit(1)

    print(
        "ML Pipeline completed successfully."
    )


if __name__ == "__main__":

    main()