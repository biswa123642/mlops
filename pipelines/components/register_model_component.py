from kfp import dsl
from kfp.dsl import component


def create_register_model_component(base_image: str):

    @component(
        base_image=base_image
    )
    def register_model_component(
        mlflow_tracking_uri: str,
        registered_model_name: str,
        run_id: str,
        model_alias: str,
        model_version_output: dsl.OutputPath(str),
        model_artifact_uri_output: dsl.OutputPath(str),
    ):
        import mlflow

        from mlflow.tracking import MlflowClient

        # ---------------------------------------------------------
        # Configure MLflow
        # ---------------------------------------------------------

        if not mlflow_tracking_uri:
            raise ValueError(
                "MLflow Tracking URI is required."
            )

        if not registered_model_name:
            raise ValueError(
                "Registered model name is required."
            )

        if not run_id:
            raise ValueError(
                "MLflow Run ID is required."
            )

        mlflow.set_tracking_uri(
            mlflow_tracking_uri
        )

        client = MlflowClient()

        # ---------------------------------------------------------
        # Model URI
        # ---------------------------------------------------------

        model_uri = (
            f"runs:/{run_id}/model"
        )

        print(
            "======================================"
        )

        print(
            "Registering Model"
        )

        print(
            f"Model Name : {registered_model_name}"
        )

        print(
            f"Run ID     : {run_id}"
        )

        print(
            f"Model URI  : {model_uri}"
        )

        print(
            "======================================"
        )

        # ---------------------------------------------------------
        # Register Model
        # ---------------------------------------------------------

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )

        version = str(
            model_version.version
        )

        print(
            "Model registered successfully."
        )

        print(
            f"Model Version: {version}"
        )

        # ---------------------------------------------------------
        # Get Registered Model Version
        # ---------------------------------------------------------

        registered_version = (
            client.get_model_version(
                name=registered_model_name,
                version=version,
            )
        )

        # ---------------------------------------------------------
        # Get Model Artifact URI
        # ---------------------------------------------------------

        artifact_uri = (
            registered_version.source
        )

        print(
            f"Model Artifact URI: {artifact_uri}"
        )

        # ---------------------------------------------------------
        # Assign MLflow Alias
        # ---------------------------------------------------------

        if model_alias:

            client.set_registered_model_alias(
                name=registered_model_name,
                alias=model_alias,
                version=version,
            )

            print(
                f"MLflow Alias '{model_alias}' "
                f"assigned to version {version}."
            )

        # ---------------------------------------------------------
        # Verify Alias
        # ---------------------------------------------------------

        if model_alias:

            aliased_model = (
                client.get_model_version_by_alias(
                    name=registered_model_name,
                    alias=model_alias,
                )
            )

            print(
                "======================================"
            )

            print(
                "MLflow Model Alias Verified"
            )

            print(
                f"Model Name : "
                f"{registered_model_name}"
            )

            print(
                f"Alias      : "
                f"{model_alias}"
            )

            print(
                f"Version    : "
                f"{aliased_model.version}"
            )

            print(
                f"Artifact   : "
                f"{aliased_model.source}"
            )

            print(
                "======================================"
            )

            artifact_uri = (
                aliased_model.source
            )

        # ---------------------------------------------------------
        # Write Model Version Output
        # ---------------------------------------------------------

        with open(
            model_version_output,
            "w",
        ) as f:

            f.write(
                version
            )

        # ---------------------------------------------------------
        # Write Artifact URI Output
        # ---------------------------------------------------------

        with open(
            model_artifact_uri_output,
            "w",
        ) as f:

            f.write(
                artifact_uri
            )

        # ---------------------------------------------------------
        # Final Output
        # ---------------------------------------------------------

        print(
            "======================================"
        )

        print(
            "MODEL REGISTRATION COMPLETED"
        )

        print(
            f"Model Name : {registered_model_name}"
        )

        print(
            f"Version    : {version}"
        )

        print(
            f"Alias      : {model_alias}"
        )

        print(
            f"Artifact   : {artifact_uri}"
        )

        print(
            "======================================"
        )

    return register_model_component