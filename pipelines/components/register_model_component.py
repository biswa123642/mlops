from typing import NamedTuple
from kfp import dsl
from kfp.dsl import component


def create_register_model_component(
    base_image: str,
):
    @component(base_image=base_image)
    def register_model_component(
        mlflow_tracking_uri: str,
        registered_model_name: str,
        run_id: str,
        model_alias: str,
        azure_storage_account: str = "",
        azure_storage_access_key: str = "",
        mlflow_username: str = "",
        mlflow_password: str = "",
    ) -> NamedTuple(
        "Outputs",
        [
            ("model_version_output", str),
            ("model_artifact_uri_output", str),
        ],
    ):
        import os
        import mlflow
        from mlflow.tracking import MlflowClient

        # =========================================================
        # 1. Validate Inputs
        # =========================================================
        if not mlflow_tracking_uri:
            raise ValueError("MLflow Tracking URI is required.")

        if not registered_model_name:
            raise ValueError("Registered model name is required.")

        if not run_id:
            raise ValueError("MLflow Run ID is required.")

        # =========================================================
        # 2. Configure Environment (MLflow Auth & Azure Storage)
        # =========================================================
        if mlflow_username and mlflow_password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

        if azure_storage_account and azure_storage_access_key:
            os.environ["AZURE_STORAGE_ACCOUNT"] = azure_storage_account
            os.environ["AZURE_STORAGE_ACCESS_KEY"] = azure_storage_access_key

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = MlflowClient(tracking_uri=mlflow_tracking_uri)

        # =========================================================
        # 3. Build Model URI
        # =========================================================
        model_uri = f"runs:/{run_id}/model"

        print("======================================")
        print("Registering Model")
        print(f"Model Name : {registered_model_name}")
        print(f"Run ID     : {run_id}")
        print(f"Model URI  : {model_uri}")
        print("======================================")

        # =========================================================
        # 4. Register Model
        # =========================================================
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )
        except Exception as exception:
            raise RuntimeError(
                f"Failed to register MLflow model. Error: {exception}"
            ) from exception

        version = str(model_version.version)

        print("Model registered successfully.")
        print(f"New MLflow Model Version: {version}")

        # =========================================================
        # 5. Get Registered Model Version
        # =========================================================
        registered_version = client.get_model_version(
            name=registered_model_name,
            version=version,
        )
        artifact_uri = registered_version.source

        print(f"Model Artifact URI: {artifact_uri}")

        # =========================================================
        # 6. Assign Model Alias
        # =========================================================
        if model_alias:
            print("======================================")
            print("Assigning MLflow Model Alias")
            print(f"Alias   : {model_alias}")
            print(f"Version : {version}")
            print("======================================")

            client.set_registered_model_alias(
                name=registered_model_name,
                alias=model_alias,
                version=version,
            )
            print(f"MLflow Alias '{model_alias}' assigned to version {version}.")

        # =========================================================
        # 7. Verify Model Alias
        # =========================================================
        if model_alias:
            try:
                aliased_model = client.get_model_version_by_alias(
                    name=registered_model_name,
                    alias=model_alias,
                )
            except Exception as exception:
                raise RuntimeError(
                    f"MLflow model alias verification failed. Error: {exception}"
                ) from exception

            aliased_version = str(aliased_model.version)
            aliased_artifact_uri = aliased_model.source

            print("======================================")
            print("MLflow Model Alias Verified")
            print(f"Model Name : {registered_model_name}")
            print(f"Alias      : {model_alias}")
            print(f"Version    : {aliased_version}")
            print(f"Artifact   : {aliased_artifact_uri}")
            print("======================================")

            if aliased_version != version:
                raise RuntimeError(
                    f"MLflow alias verification failed. Expected version {version}, "
                    f"but alias '{model_alias}' points to version {aliased_version}."
                )

            version = aliased_version
            artifact_uri = aliased_artifact_uri

        # =========================================================
        # 8. Validate Final Model Information
        # =========================================================
        if not version:
            raise RuntimeError("MLflow model version is empty.")

        if not artifact_uri:
            raise RuntimeError("MLflow model artifact URI is empty.")

        # =========================================================
        # 9. Return Outputs via NamedTuple
        # =========================================================
        print("======================================")
        print("MODEL REGISTRATION COMPLETED")
        print(f"Model Name : {registered_model_name}")
        print(f"Version    : {version}")
        print(f"Alias      : {model_alias}")
        print(f"Artifact   : {artifact_uri}")
        print("======================================")

        return (version, artifact_uri)

    return register_model_component