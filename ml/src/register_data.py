from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

from config import (
    AZURE_SUBSCRIPTION_ID,
    AZURE_RESOURCE_GROUP,
    AZURE_ML_WORKSPACE,
    DATA_ASSET_NAME,
    DATA_ASSET_VERSION,
    DATASTORE_NAME,
)


def main():
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_ML_WORKSPACE,
    )

    data_asset = Data(
        name=DATA_ASSET_NAME,
        version=str(DATA_ASSET_VERSION),
        description="Customer churn dataset",
        type=AssetTypes.URI_FILE,
        path=(
            f"azureml://datastores/{DATASTORE_NAME}"
            "/paths/customer_churn.csv"
        ),
        tags={
            "dataset": "customer-churn",
            "format": "csv",
        },
    )

    registered_data_asset = ml_client.data.create_or_update(data_asset)

    print("----------------------------------------")
    print("Data asset registered successfully")
    print(f"Name: {registered_data_asset.name}")
    print(f"Version: {registered_data_asset.version}")
    print(f"Path: {registered_data_asset.path}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()