from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

from config import (
    AZURE_ML_WORKSPACE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    MODEL_NAME,
)


TRAINING_DISPLAY_NAME = (
    "customer-churn-training"
)


def get_latest_completed_training_job(
    ml_client: MLClient,
):
    completed_jobs = [
        job
        for job in ml_client.jobs.list()
        if job.status == "Completed"
        and job.display_name == TRAINING_DISPLAY_NAME
    ]

    if not completed_jobs:
        raise RuntimeError(
            "No completed customer churn training jobs "
            "were found."
        )

    return max(
        completed_jobs,
        key=lambda job: (
            job.creation_context.created_at
            if job.creation_context
            and job.creation_context.created_at
            else ""
        ),
    )


def main():
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_ML_WORKSPACE,
    )

    latest_job = (
        get_latest_completed_training_job(
            ml_client
        )
    )

    print(
        f"Selected training job: {latest_job.name}"
    )

    job = ml_client.jobs.get(
        latest_job.name
    )

    if not job.outputs:
        raise RuntimeError(
            f"Training job '{latest_job.name}' "
            "does not contain any outputs."
        )

    if "model" not in job.outputs:
        raise RuntimeError(
            f"Training job '{latest_job.name}' "
            "does not contain an output named 'model'."
        )

    model_path = (
        f"azureml://jobs/{latest_job.name}"
        "/outputs/model"
    )

    print(
        f"Registering model from: {model_path}"
    )

    model = Model(
        name=MODEL_NAME,
        path=model_path,
        type=AssetTypes.CUSTOM_MODEL,
        description=(
            "Customer churn prediction model"
        ),
        tags={
            "source_job": latest_job.name,
            "model_type": (
                "random-forest-pipeline"
            ),
        },
    )

    registered_model = (
        ml_client.models.create_or_update(model)
    )

    print("----------------------------------------")
    print("Model registered successfully")
    print(
        f"Model name: {registered_model.name}"
    )
    print(
        f"Model version: {registered_model.version}"
    )
    print("----------------------------------------")

    print("Registered model reference:")
    print(
        f"azureml:{registered_model.name}:"
        f"{registered_model.version}"
    )


if __name__ == "__main__":
    main()