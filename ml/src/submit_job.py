from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from config import (
    AZURE_SUBSCRIPTION_ID,
    AZURE_RESOURCE_GROUP,
    AZURE_ML_WORKSPACE,
    AZURE_ML_COMPUTE,
    DATA_ASSET_NAME,
    DATA_ASSET_VERSION,
    ENVIRONMENT_NAME,
)


def main():
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_ML_WORKSPACE,
    )

    src_dir = Path(__file__).resolve().parent

    job = command(
        code=str(src_dir),
        command=(
            "python train.py "
            "--input-data ${{inputs.input_data}} "
            "--output-dir ${{outputs.model}}"
        ),
        inputs={
            "input_data": Input(
                type=AssetTypes.URI_FILE,
                path=(
                    f"azureml:{DATA_ASSET_NAME}:"
                    f"{DATA_ASSET_VERSION}"
                ),
            )
        },
        outputs={
            "model": Output(
                type=AssetTypes.URI_FOLDER,
                mode="upload",
            )
        },
        environment=ENVIRONMENT_NAME,
        compute=AZURE_ML_COMPUTE,
        experiment_name="customer-churn",
        display_name="customer-churn-training",
        description="Train the customer churn model",
    )

    returned_job = ml_client.jobs.create_or_update(job)

    print("----------------------------------------")
    print("Training job submitted successfully")
    print(f"Job name: {returned_job.name}")
    print(f"Status: {returned_job.status}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()