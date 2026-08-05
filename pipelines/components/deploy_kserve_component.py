from kfp import dsl
from kfp.dsl import component


def create_deploy_kserve_component(base_image: str):

    @component(base_image=base_image)
    def deploy_kserve_component(
        mlflow_tracking_uri: str,
        registered_model_name: str,
        model_alias: str,
        model_version: str,
        model_artifact_uri: str,
        namespace: str,
        inference_service_name: str,
        service_account_name: str,
        kserve_secret_name: str,
        mlflow_username: str = "",
        mlflow_password: str = "",
        azure_storage_account: str = "",
        azure_storage_access_key: str = "",
    ):
        import os
        from urllib.parse import urlparse
        from kubernetes import client, config

        # ---------------------------------------------------------
        # 1. Set Environment Variables (MLflow auth & Azure storage)
        # ---------------------------------------------------------
        if mlflow_username and mlflow_password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

        if azure_storage_account and azure_storage_access_key:
            os.environ["AZURE_STORAGE_ACCOUNT"] = azure_storage_account
            os.environ["AZURE_STORAGE_ACCESS_KEY"] = azure_storage_access_key

        # ---------------------------------------------------------
        # 2. Convert MLflow wasbs:// URI to KServe-compatible https:// URI
        #    Example:
        #      wasbs://mlops@mlstoregs.blob.core.windows.net/1/.../artifacts/model
        #    -> https://mlstoregs.blob.core.windows.net/mlops/1/.../artifacts/model
        # ---------------------------------------------------------
        kserve_storage_uri = model_artifact_uri
        if model_artifact_uri.startswith("wasbs://"):
            parsed = urlparse(model_artifact_uri)
            # parsed.netloc: "mlops@mlstoregs.blob.core.windows.net"
            container_and_host = parsed.netloc.split("@")
            if len(container_and_host) == 2:
                container = container_and_host[0]
                host = container_and_host[1]  # "mlstoregs.blob.core.windows.net"
                path = parsed.path.lstrip("/")  # "1/.../artifacts/model"
                kserve_storage_uri = f"https://{host}/{container}/{path}"
            else:
                raise ValueError(
                    f"Unexpected wasbs URI format for model_artifact_uri: {model_artifact_uri}"
                )

        print("======================================")
        print("Deploying Model to KServe")
        print(f"InferenceService Name : {inference_service_name}")
        print(f"Namespace              : {namespace}")
        print(f"Service Account        : {service_account_name}")
        print(f"Storage Secret         : {kserve_secret_name}")
        print(f"Model Name             : {registered_model_name}")
        print(f"Model Version          : {model_version}")
        print(f"Model Alias            : {model_alias}")
        print(f"MLflow Artifact URI    : {model_artifact_uri}")
        print(f"KServe Storage URI     : {kserve_storage_uri}")
        print("======================================")

        # ---------------------------------------------------------
        # 3. Initialize In-Cluster Kubernetes Client
        # ---------------------------------------------------------
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        core_api = client.CoreV1Api()
        custom_api = client.CustomObjectsApi()

        # ---------------------------------------------------------
        # 4. Dynamic Secret & ServiceAccount Provisioning
        #    (Requires RBAC allowing default-editor to manage
        #     secrets & serviceaccounts in the target namespace)
        # ---------------------------------------------------------
        if azure_storage_account and azure_storage_access_key:
            secret_manifest = client.V1Secret(
                api_version="v1",
                kind="Secret",
                metadata=client.V1ObjectMeta(
                    name=kserve_secret_name,
                    namespace=namespace,
                ),
                type="Opaque",
                string_data={
                    "AZURE_STORAGE_ACCESS_KEY": azure_storage_access_key,
                },
            )

            try:
                core_api.read_namespaced_secret(kserve_secret_name, namespace)
                core_api.replace_namespaced_secret(
                    kserve_secret_name,
                    namespace,
                    secret_manifest,
                )
                print(
                    f"Updated Secret '{kserve_secret_name}' in namespace '{namespace}'."
                )
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    core_api.create_namespaced_secret(namespace, secret_manifest)
                    print(
                        f"Created Secret '{kserve_secret_name}' in namespace '{namespace}'."
                    )
                else:
                    raise RuntimeError(f"Failed to manage secret: {e}") from e

            try:
                sa = core_api.read_namespaced_service_account(
                    service_account_name,
                    namespace,
                )
                secret_names = [
                    s.name for s in (sa.secrets or []) if s.name is not None
                ]
                if kserve_secret_name not in secret_names:
                    sa.secrets = (sa.secrets or []) + [
                        client.V1ObjectReference(name=kserve_secret_name)
                    ]
                    core_api.replace_namespaced_service_account(
                        service_account_name,
                        namespace,
                        sa,
                    )
                    print(
                        f"Attached secret '{kserve_secret_name}' to ServiceAccount '{service_account_name}'."
                    )
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    sa_manifest = client.V1ServiceAccount(
                        metadata=client.V1ObjectMeta(
                            name=service_account_name,
                            namespace=namespace,
                        ),
                        secrets=[
                            client.V1ObjectReference(name=kserve_secret_name)
                        ],
                    )
                    core_api.create_namespaced_service_account(
                        namespace,
                        sa_manifest,
                    )
                    print(
                        f"Created ServiceAccount '{service_account_name}' with secret '{kserve_secret_name}'."
                    )
                else:
                    raise RuntimeError(
                        f"Failed to manage service account: {e}"
                    ) from e

        # ---------------------------------------------------------
        # 5. Construct KServe InferenceService CRD Spec
        # ---------------------------------------------------------
        isvc_manifest = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": inference_service_name,
                "namespace": namespace,
                "labels": {
                    "app": inference_service_name,
                    "model-version": str(model_version),
                },
            },
            "spec": {
                "predictor": {
                    "serviceAccountName": service_account_name,
                    "model": {
                        "modelFormat": {
                            "name": "sklearn",
                        },
                        "storageUri": kserve_storage_uri,
                    },
                }
            },
        }

        # ---------------------------------------------------------
        # 6. Apply or Update InferenceService
        #    (Requires RBAC for inferenceservices.serving.kserve.io)
        # ---------------------------------------------------------
        group = "serving.kserve.io"
        version = "v1beta1"
        plural = "inferenceservices"

        try:
            existing_isvc = custom_api.get_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=inference_service_name,
            )

            isvc_manifest["metadata"]["resourceVersion"] = existing_isvc["metadata"][
                "resourceVersion"
            ]
            custom_api.replace_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=inference_service_name,
                body=isvc_manifest,
            )
            print(f"Successfully updated InferenceService '{inference_service_name}'.")

        except client.exceptions.ApiException as e:
            if e.status == 404:
                custom_api.create_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    body=isvc_manifest,
                )
                print(
                    f"Successfully created InferenceService '{inference_service_name}'."
                )
            else:
                raise RuntimeError(
                    f"Failed to deploy InferenceService to KServe: {e}"
                ) from e

        print("======================================")
        print("KSERVE DEPLOYMENT TRIGGERED SUCCESSFULLY")
        print("======================================")

    return deploy_kserve_component