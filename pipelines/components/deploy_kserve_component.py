from kfp import dsl
from kfp.dsl import component


def create_deploy_kserve_component(base_image: str):

    @component(
        base_image=base_image
    )
    def deploy_kserve_component(
        mlflow_tracking_uri: str,
        registered_model_name: str,
        model_alias: str,
        namespace: str,
        inference_service_name: str,
    ):
        import mlflow

        from kubernetes import client
        from kubernetes import config
        from kubernetes.client.rest import ApiException

        # ---------------------------------------------------------
        # 1. Validate Inputs
        # ---------------------------------------------------------

        if not mlflow_tracking_uri:
            raise ValueError(
                "MLflow Tracking URI is required."
            )

        if not registered_model_name:
            raise ValueError(
                "Registered model name is required."
            )

        if not model_alias:
            raise ValueError(
                "MLflow model alias is required."
            )

        if not namespace:
            raise ValueError(
                "Kubernetes namespace is required."
            )

        if not inference_service_name:
            raise ValueError(
                "KServe InferenceService name is required."
            )

        # ---------------------------------------------------------
        # 2. Configure MLflow
        # ---------------------------------------------------------

        mlflow.set_tracking_uri(
            mlflow_tracking_uri
        )

        client_mlflow = (
            mlflow.tracking.MlflowClient()
        )

        # ---------------------------------------------------------
        # 3. Resolve Model from MLflow Alias
        # ---------------------------------------------------------

        print(
            "======================================"
        )

        print(
            "Resolving MLflow Model"
        )

        print(
            f"Model Name : "
            f"{registered_model_name}"
        )

        print(
            f"Model Alias: "
            f"{model_alias}"
        )

        print(
            "======================================"
        )

        model_version = (
            client_mlflow
            .get_model_version_by_alias(
                name=registered_model_name,
                alias=model_alias,
            )
        )

        model_version_number = (
            model_version.version
        )

        artifact_uri = (
            model_version.source
        )

        print(
            f"Resolved Model Version: "
            f"{model_version_number}"
        )

        print(
            f"Model Artifact URI: "
            f"{artifact_uri}"
        )

        # ---------------------------------------------------------
        # 4. Load Kubernetes Configuration
        # ---------------------------------------------------------

        try:

            config.load_incluster_config()

            print(
                "Using in-cluster Kubernetes configuration."
            )

        except Exception:

            config.load_kube_config()

            print(
                "Using local Kubernetes configuration."
            )

        # ---------------------------------------------------------
        # 5. Kubernetes Custom Objects API
        # ---------------------------------------------------------

        api = client.CustomObjectsApi()

        group = "serving.kserve.io"

        version = "v1beta1"

        plural = "inferenceservices"

        # ---------------------------------------------------------
        # 6. KServe InferenceService Definition
        # ---------------------------------------------------------

        inference_service = {

            "apiVersion":
                "serving.kserve.io/v1beta1",

            "kind":
                "InferenceService",

            "metadata": {

                "name":
                    inference_service_name,

                "namespace":
                    namespace,

                "labels": {

                    "app":
                        inference_service_name,

                    "model":
                        registered_model_name,

                    "mlflow-model-version":
                        str(
                            model_version_number
                        ),

                    "mlflow-model-alias":
                        model_alias,

                },

            },

            "spec": {

                "predictor": {

                    "model": {

                        "modelFormat": {

                            "name":
                                "sklearn",

                        },

                        "storageUri":
                            artifact_uri,

                    },

                },

            },

        }

        # ---------------------------------------------------------
        # 7. Create or Update KServe InferenceService
        # ---------------------------------------------------------

        try:

            api.get_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=inference_service_name,
            )

            print(
                "Existing KServe "
                "InferenceService found."
            )

            api.patch_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=inference_service_name,
                body=inference_service,
            )

            print(
                "KServe InferenceService "
                "updated successfully."
            )

        except ApiException as exception:

            if exception.status == 404:

                api.create_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    body=inference_service,
                )

                print(
                    "KServe InferenceService "
                    "created successfully."
                )

            else:

                raise RuntimeError(
                    "Failed to create or update "
                    "KServe InferenceService. "
                    f"Status: {exception.status}, "
                    f"Reason: {exception.reason}"
                )

        # ---------------------------------------------------------
        # 8. Deployment Information
        # ---------------------------------------------------------

        print(
            "======================================"
        )

        print(
            "KSERVE DEPLOYMENT COMPLETED"
        )

        print(
            f"InferenceService : "
            f"{inference_service_name}"
        )

        print(
            f"Namespace         : "
            f"{namespace}"
        )

        print(
            f"Model             : "
            f"{registered_model_name}"
        )

        print(
            f"MLflow Alias      : "
            f"{model_alias}"
        )

        print(
            f"Model Version     : "
            f"{model_version_number}"
        )

        print(
            f"Artifact URI      : "
            f"{artifact_uri}"
        )

        print(
            "======================================"
        )

    return deploy_kserve_component