from typing import NamedTuple
from kfp import dsl
from kfp.dsl import component


def create_train_component(
    base_image: str,
):
    @component(base_image=base_image)
    def train_component(
        azure_storage_account: str,
        azure_storage_container: str,
        dataset_name: str,
        mlflow_tracking_uri: str,
        experiment_name: str,
        azure_storage_access_key: str = "",
        mlflow_username: str = "",
        mlflow_password: str = "",
    ) -> NamedTuple(
        "Outputs",
        [
            ("run_id_output", str),
            ("accuracy_output", float),
            ("precision_output", float),
            ("recall_output", float),
            ("f1_output", float),
        ],
    ):
        import io
        import os
        import pandas as pd
        import mlflow
        import mlflow.sklearn

        from azure.storage.blob import BlobClient
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import train_test_split

        # =========================================================
        # 1. Validate Core Inputs
        # =========================================================
        if not azure_storage_account:
            raise ValueError("Azure Storage Account name is required.")

        if not azure_storage_container:
            raise ValueError("Azure Storage Container name is required.")

        if not dataset_name:
            raise ValueError("Dataset name is required.")

        if not mlflow_tracking_uri:
            raise ValueError("MLflow Tracking URI is required.")

        if not experiment_name:
            raise ValueError("MLflow Experiment Name is required.")

        # Access key is required now (no DefaultAzureCredential fallback)
        if not azure_storage_access_key:
            raise ValueError(
                "Azure Storage Access Key is required for dataset download."
            )

        # =========================================================
        # 2. Configure Environment (MLflow Auth & Azure Storage)
        # =========================================================
        if mlflow_username and mlflow_password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

        os.environ["AZURE_STORAGE_ACCOUNT"] = azure_storage_account
        os.environ["AZURE_STORAGE_ACCESS_KEY"] = azure_storage_access_key

        # =========================================================
        # 3. Download Dataset from Azure Blob Storage
        # =========================================================
        blob_url = (
            f"https://{azure_storage_account}.blob.core.windows.net/"
            f"{azure_storage_container}/{dataset_name}"
        )

        print("======================================")
        print("Downloading Dataset from Azure Storage")
        print(f"Blob URL: {blob_url}")
        print("======================================")

        try:
            # Always use access key; no DefaultAzureCredential
            blob_client = BlobClient.from_blob_url(
                blob_url=blob_url,
                credential=azure_storage_access_key,
            )

            download_stream = blob_client.download_blob()
            data_bytes = download_stream.readall()
            dataframe = pd.read_csv(io.BytesIO(data_bytes))

            print("Dataset downloaded successfully.")
            print(f"Dataset Shape: {dataframe.shape}")

        except Exception as exception:
            raise RuntimeError(
                f"Failed to download dataset from Azure Blob Storage. Error: {exception}"
            ) from exception

        if dataframe.empty:
            raise ValueError("Downloaded dataset is empty.")

        # =========================================================
        # 4. Prepare Features and Target
        # =========================================================
        target_column = "churn"

        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column]

        # One-hot encode categorical features if present
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if len(y.unique()) > 1 else None,
        )

        print("======================================")
        print("Dataset Split Details")
        print(f"Train Shape : {X_train.shape}")
        print(f"Test Shape  : {X_test.shape}")
        print("======================================")

        # =========================================================
        # 5. Configure MLflow Tracking
        # =========================================================
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        # =========================================================
        # 6. Model Training & Evaluation inside MLflow Run
        # =========================================================
        print("Starting MLflow Run...")

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            print("======================================")
            print("MLflow Training Started")
            print(f"Experiment : {experiment_name}")
            print(f"Run ID     : {run_id}")
            print("======================================")

            n_estimators = 100
            max_depth = 10
            random_state = 42

            # Log Hyperparameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", random_state)

            # Train Model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            model.fit(X_train, y_train)

            # Make Predictions
            y_pred = model.predict(X_test)

            # Calculate Evaluation Metrics
            accuracy = float(accuracy_score(y_test, y_pred))
            precision = float(precision_score(y_test, y_pred, zero_division=0))
            recall = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))

            # Log Metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            print("======================================")
            print("Model Evaluation Results")
            print(f"Accuracy  : {accuracy:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1 Score  : {f1:.4f}")
            print("======================================")

            # Log Model Artifact
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )

            print("Model artifact successfully logged to MLflow / Azure Blob.")

        print("======================================")
        print("TRAINING COMPONENT COMPLETED")
        print(f"Run ID : {run_id}")
        print("======================================")

        # =========================================================
        # 7. Return Outputs via NamedTuple
        # =========================================================
        return (run_id, accuracy, precision, recall, f1)

    return train_component