import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import urlencode, urlsplit

import kfp
import requests
import urllib3

from config import (
    AZURE_STORAGE_ACCESS_KEY,
    AZURE_STORAGE_ACCOUNT,
    AZURE_STORAGE_CONTAINER,
    DATASET_NAME,
    EXPERIMENT_NAME,
    INFERENCE_SERVICE_NAME,
    KSERVE_NAMESPACE,
    KSERVE_SECRET_NAME,
    KSERVE_SERVICE_ACCOUNT,
    KUBEFLOW_ENDPOINT,
    KUBEFLOW_PASSWORD,
    KUBEFLOW_USER_NAMESPACE,
    KUBEFLOW_USERNAME,
    MIN_ACCURACY,
    MIN_F1,
    MIN_PRECISION,
    MIN_RECALL,
    MLFLOW_PASSWORD,
    MLFLOW_TRACKING_URI,
    MLFLOW_USERNAME,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
)


# =========================================================
# Dex Authentication Manager
# =========================================================
class KFPClientManager:
    """Creates `kfp.Client` instances with Dex authentication."""

    def __init__(
        self,
        api_url: str,
        dex_username: str,
        dex_password: str,
        namespace: str,
        dex_auth_type: str = "local",
        skip_tls_verify: bool = True,
    ):
        self._api_url = api_url
        self._skip_tls_verify = skip_tls_verify
        self._dex_username = dex_username
        self._dex_password = dex_password
        self._dex_auth_type = dex_auth_type
        self.namespace = namespace

        if self._skip_tls_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        if self._dex_auth_type not in ["ldap", "local"]:
            raise ValueError(
                f"Invalid `dex_auth_type` '{self._dex_auth_type}', must be one of: ['ldap', 'local']"
            )

    def _get_session_cookies(self) -> str:
        """Authenticates with Dex / OAuth2-Proxy and returns session cookie string."""
        s = requests.Session()

        # Step 1: Initial GET against the API endpoint
        resp = s.get(
            self._api_url,
            allow_redirects=True,
            verify=not self._skip_tls_verify,
        )

        if resp.status_code == 200:
            pass
        elif resp.status_code == 403:
            url_obj = urlsplit(resp.url)
            url_obj = url_obj._replace(
                path="/oauth2/start", query=urlencode({"rd": url_obj.path})
            )
            resp = s.get(
                url_obj.geturl(),
                allow_redirects=True,
                verify=not self._skip_tls_verify,
            )
        else:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for GET against: {self._api_url}"
            )

        if len(resp.history) == 0:
            return ""

        # Step 2: Rewrite Dex auth path if multi-auth selection is prompted
        url_obj = urlsplit(resp.url)
        if re.search(r"/auth$", url_obj.path):
            url_obj = url_obj._replace(
                path=re.sub(
                    r"/auth$", f"/auth/{self._dex_auth_type}", url_obj.path
                )
            )

        # Step 3: Resolve exact login URL
        if re.search(r"/auth/.*/login$", url_obj.path):
            dex_login_url = url_obj.geturl()
        else:
            resp = s.get(
                url_obj.geturl(),
                allow_redirects=True,
                verify=not self._skip_tls_verify,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for GET against: {url_obj.geturl()}"
                )
            dex_login_url = resp.url

        # Step 4: POST credentials to Dex
        resp = s.post(
            dex_login_url,
            data={"login": self._dex_username, "password": self._dex_password},
            allow_redirects=True,
            verify=not self._skip_tls_verify,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for POST against: {dex_login_url}"
            )

        if len(resp.history) == 0:
            raise RuntimeError(
                f"Login credentials invalid — No redirect after POST to: {dex_login_url}"
            )

        # Step 5: Post authorization approval if redirected to /approval
        url_obj = urlsplit(resp.url)
        if re.search(r"/approval$", url_obj.path):
            dex_approval_url = url_obj.geturl()
            resp = s.post(
                dex_approval_url,
                data={"approval": "approve"},
                allow_redirects=True,
                verify=not self._skip_tls_verify,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for POST against: {url_obj.geturl()}"
                )

        return "; ".join([f"{c.name}={c.value}" for c in s.cookies])

    def create_kfp_client(self) -> kfp.Client:
        """Instantiates an authenticated KFP Client instance."""
        try:
            session_cookies = self._get_session_cookies()
        except Exception as ex:
            raise RuntimeError("Failed to get Dex session cookies") from ex

        original_load_config = kfp.Client._load_config

        def patched_load_config(client_self, *args, **kwargs):
            config = original_load_config(client_self, *args, **kwargs)
            config.verify_ssl = not self._skip_tls_verify
            return config

        kfp.Client._load_config = patched_load_config

        return kfp.Client(
            host=self._api_url,
            cookies=session_cookies,
            namespace=self.namespace,
        )


# =========================================================
# Helpers
# =========================================================
def get_required_value(name: str, value: str | None) -> str:
    if not value:
        raise ValueError(f"Required configuration '{name}' is not set.")
    return value


# =========================================================
# Main Pipeline Workflow
# =========================================================
def main():
    # -----------------------------------------------------
    # 1. Validate Required Configurations
    # -----------------------------------------------------
    kubeflow_endpoint = get_required_value(
        "KUBEFLOW_ENDPOINT", KUBEFLOW_ENDPOINT
    )
    kubeflow_username = get_required_value(
        "KUBEFLOW_USERNAME", KUBEFLOW_USERNAME
    )
    kubeflow_password = get_required_value(
        "KUBEFLOW_PASSWORD", KUBEFLOW_PASSWORD
    )
    user_namespace = get_required_value(
        "KUBEFLOW_USER_NAMESPACE", KUBEFLOW_USER_NAMESPACE
    )

    azure_storage_account = get_required_value(
        "AZURE_STORAGE_ACCOUNT", AZURE_STORAGE_ACCOUNT
    )
    azure_storage_container = get_required_value(
        "AZURE_STORAGE_CONTAINER", AZURE_STORAGE_CONTAINER
    )
    mlflow_tracking_uri = get_required_value(
        "MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI
    )

    pipeline_package = os.getenv(
        "PIPELINE_PACKAGE", "compiled/customer_churn_pipeline.yaml"
    )
    pipeline_name = os.getenv("PIPELINE_NAME", "customer-churn-ml-pipeline")
    kubeflow_experiment_name = os.getenv(
        "KUBEFLOW_EXPERIMENT_NAME", "customer-churn-experiment"
    )

    if not os.path.exists(pipeline_package):
        raise FileNotFoundError(
            f"Pipeline package not found:\n{pipeline_package}\n"
            "Please run compile_pipeline.py first."
        )

    # -----------------------------------------------------
    # 2. Authenticate & Create KFP Client
    # -----------------------------------------------------
    api_url = f"{kubeflow_endpoint.rstrip('/')}/pipeline"
    print("==========================================")
    print("Authenticating with Kubeflow Dex...")
    print(f"API Endpoint   : {api_url}")
    print(f"User Namespace : {user_namespace}")
    print("==========================================")

    client_manager = KFPClientManager(
        api_url=api_url,
        dex_username=kubeflow_username,
        dex_password=kubeflow_password,
        namespace=user_namespace,
        dex_auth_type="local",
        skip_tls_verify=True,
    )

    kfp_client = client_manager.create_kfp_client()
    print("Successfully authenticated and created KFP Client!")

    # -----------------------------------------------------
    # 3. Create or Fetch Experiment
    # -----------------------------------------------------
    try:
        experiment = kfp_client.create_experiment(
            name=kubeflow_experiment_name, namespace=user_namespace
        )
        print(f"Created experiment: {experiment.name}")
    except Exception:
        print(f"Accessing existing experiment '{kubeflow_experiment_name}'...")
        experiment = kfp_client.get_experiment(
            experiment_name=kubeflow_experiment_name, namespace=user_namespace
        )

    # -----------------------------------------------------
    # 4. Upload Pipeline or Version with (v1, v2 + Date)
    # -----------------------------------------------------
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    pipeline_id = kfp_client.get_pipeline_id(pipeline_name)

    if pipeline_id:
        print(f"Existing pipeline found (ID: {pipeline_id})")

        # Query existing versions to find the total count
        try:
            existing_versions = kfp_client.list_pipeline_versions(
                pipeline_id=pipeline_id
            )
            count = (
                len(existing_versions.pipeline_versions)
                if existing_versions.pipeline_versions
                else 0
            )
        except Exception:
            count = 0

        next_version_num = count + 1
        version_name = f"v{next_version_num}_{date_str}"

        pipeline_version = kfp_client.upload_pipeline_version(
            pipeline_package,
            pipeline_version_name=version_name,
            pipeline_id=pipeline_id,
        )
        print(f"Uploaded new pipeline version: {version_name}")
    else:
        print("Pipeline entry not found. Creating new pipeline entry...")
        version_name = f"v1_{date_str}"
        pipeline = kfp_client.upload_pipeline(
            pipeline_package,
            pipeline_name=pipeline_name,
            description=f"Initial pipeline release {version_name}",
        )
        print(f"Created pipeline with version: {version_name}")

    # -----------------------------------------------------
    # 5. Define Parameters & Trigger Run
    # -----------------------------------------------------
    parameters = {
        "azure_storage_account": azure_storage_account,
        "azure_storage_container": azure_storage_container,
        "azure_storage_access_key": AZURE_STORAGE_ACCESS_KEY or "",
        "dataset_name": DATASET_NAME,
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "mlflow_username": MLFLOW_USERNAME or "",
        "mlflow_password": MLFLOW_PASSWORD or "",
        "experiment_name": EXPERIMENT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "min_accuracy": MIN_ACCURACY,
        "min_precision": MIN_PRECISION,
        "min_recall": MIN_RECALL,
        "min_f1": MIN_F1,
        "namespace": KSERVE_NAMESPACE,
        "inference_service_name": INFERENCE_SERVICE_NAME,
        "service_account_name": KSERVE_SERVICE_ACCOUNT,
        "kserve_secret_name": KSERVE_SECRET_NAME,
    }

    run_name = f"churn-run-{version_name}"

    print("==========================================")
    print(f"Starting Kubeflow Pipeline Run: {run_name}")
    print("==========================================")

    run = kfp_client.create_run_from_pipeline_package(
        pipeline_file=pipeline_package,
        arguments=parameters,
        experiment_name=kubeflow_experiment_name,
        run_name=run_name,
        namespace=user_namespace,
    )
    run_id = run.run_id
    print(f"Pipeline Run Started! Run ID: {run_id}")

    # -----------------------------------------------------
    # 6. Poll Execution Status
    # -----------------------------------------------------
    print("Monitoring pipeline progress...")
    while True:
        run_details = kfp_client.get_run(run_id)
        raw_state = getattr(
            run_details, "state", getattr(run_details, "status", "RUNNING")
        )
        state = str(raw_state).upper()
        print(f"Current Pipeline Status: {state}")

        if any(
            term in state
            for term in ["SUCCEEDED", "FAILED", "ERROR", "CANCELED", "SKIPPED"]
        ):
            break

        time.sleep(30)

    print("==========================================")
    print(f"Pipeline Finished with Status: {state}")
    print("==========================================")

    if "SUCCEEDED" not in state:
        print("Pipeline run failed.")
        sys.exit(1)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()