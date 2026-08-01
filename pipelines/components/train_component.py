from kfp import dsl
from kfp.dsl import component


def create_train_component(base_image: str):

    @component(
        base_image=base_image
    )
    def train_component(
        azure_storage_account: str,
        azure_storage_container: str,
        dataset_name: str,
        mlflow_tracking_uri: str,
        experiment_name: str,
        run_id_output: dsl.OutputPath(str),
        accuracy_output: dsl.OutputPath(float),
        precision_output: dsl.OutputPath(float),
        recall_output: dsl.OutputPath(float),
        f1_output: dsl.OutputPath(float),
    ):
        import io

        import mlflow
        import mlflow.sklearn
        import pandas as pd

        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobClient

        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        # ---------------------------------------------------------
        # Validate inputs
        # ---------------------------------------------------------

        if not azure_storage_account:
            raise ValueError(
                "Azure Storage Account is required."
            )

        if not azure_storage_container:
            raise ValueError(
                "Azure Storage Container is required."
            )

        if not dataset_name:
            raise ValueError(
                "Dataset name is required."
            )

        if not mlflow_tracking_uri:
            raise ValueError(
                "MLflow Tracking URI is required."
            )

        # ---------------------------------------------------------
        # Download dataset from Azure Blob Storage
        # ---------------------------------------------------------

        account_url = (
            f"https://{azure_storage_account}.blob.core.windows.net"
        )

        print(
            f"Downloading dataset: {dataset_name}"
        )

        credential = DefaultAzureCredential()

        blob_client = BlobClient(
            account_url=account_url,
            container_name=azure_storage_container,
            blob_name=dataset_name,
            credential=credential,
        )

        blob_data = (
            blob_client
            .download_blob()
            .readall()
        )

        df = pd.read_csv(
            io.BytesIO(blob_data)
        )

        print(
            "Dataset loaded successfully."
        )

        print(
            f"Rows: {len(df)}"
        )

        print(
            f"Columns: {len(df.columns)}"
        )

        # ---------------------------------------------------------
        # Validate target column
        # ---------------------------------------------------------

        target_column = "Churn"

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                "was not found in the dataset."
            )

        # ---------------------------------------------------------
        # Prepare features and target
        # ---------------------------------------------------------

        y = df[target_column]

        X = df.drop(
            columns=[target_column]
        )

        # ---------------------------------------------------------
        # Convert target to numeric
        # ---------------------------------------------------------

        if y.dtype == "object":

            y = (
                y.astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "yes": 1,
                        "no": 0,
                        "true": 1,
                        "false": 0,
                        "1": 1,
                        "0": 0,
                    }
                )
            )

        y = pd.to_numeric(
            y,
            errors="coerce",
        )

        if y.isna().any():
            raise ValueError(
                "Target column contains "
                "invalid or missing values."
            )

        # ---------------------------------------------------------
        # Identify feature types
        # ---------------------------------------------------------

        numeric_features = (
            X.select_dtypes(
                include=[
                    "int64",
                    "float64",
                    "int32",
                    "float32",
                ]
            )
            .columns
            .tolist()
        )

        categorical_features = (
            X.select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool",
                ]
            )
            .columns
            .tolist()
        )

        # ---------------------------------------------------------
        # Numeric preprocessing
        # ---------------------------------------------------------

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                )
            ]
        )

        # ---------------------------------------------------------
        # Categorical preprocessing
        # ---------------------------------------------------------

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        # ---------------------------------------------------------
        # Preprocessor
        # ---------------------------------------------------------

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features,
                ),
            ]
        )

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        model_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        # ---------------------------------------------------------
        # Train / Test split
        # ---------------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )
        )

        # ---------------------------------------------------------
        # Configure MLflow
        # ---------------------------------------------------------

        mlflow.set_tracking_uri(
            mlflow_tracking_uri
        )

        mlflow.set_experiment(
            experiment_name
        )

        # ---------------------------------------------------------
        # Start MLflow Run
        # ---------------------------------------------------------

        with mlflow.start_run() as run:

            run_id = run.info.run_id

            print(
                f"MLflow Run ID: {run_id}"
            )

            # -----------------------------------------------------
            # Train
            # -----------------------------------------------------

            model_pipeline.fit(
                X_train,
                y_train,
            )

            # -----------------------------------------------------
            # Predict
            # -----------------------------------------------------

            predictions = (
                model_pipeline.predict(
                    X_test
                )
            )

            # -----------------------------------------------------
            # Calculate metrics
            # -----------------------------------------------------

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0,
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0,
            )

            # -----------------------------------------------------
            # Log parameters
            # -----------------------------------------------------

            mlflow.log_param(
                "algorithm",
                "RandomForestClassifier",
            )

            mlflow.log_param(
                "n_estimators",
                200,
            )

            mlflow.log_param(
                "random_state",
                42,
            )

            mlflow.log_param(
                "dataset",
                dataset_name,
            )

            # -----------------------------------------------------
            # Log metrics
            # -----------------------------------------------------

            mlflow.log_metric(
                "accuracy",
                accuracy,
            )

            mlflow.log_metric(
                "precision",
                precision,
            )

            mlflow.log_metric(
                "recall",
                recall,
            )

            mlflow.log_metric(
                "f1_score",
                f1,
            )

            # -----------------------------------------------------
            # Log model to MLflow
            # -----------------------------------------------------

            mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                artifact_path="model",
            )

            print(
                "Model logged successfully to MLflow."
            )

            print(
                f"Artifact URI: "
                f"{mlflow.get_artifact_uri()}"
            )

        # ---------------------------------------------------------
        # Write KFP outputs
        # ---------------------------------------------------------

        with open(
            run_id_output,
            "w",
        ) as f:

            f.write(
                run_id
            )

        with open(
            accuracy_output,
            "w",
        ) as f:

            f.write(
                str(accuracy)
            )

        with open(
            precision_output,
            "w",
        ) as f:

            f.write(
                str(precision)
            )

        with open(
            recall_output,
            "w",
        ) as f:

            f.write(
                str(recall)
            )

        with open(
            f1_output,
            "w",
        ) as f:

            f.write(
                str(f1)
            )

        # ---------------------------------------------------------
        # Final Output
        # ---------------------------------------------------------

        print(
            "======================================"
        )

        print(
            "TRAINING COMPLETED SUCCESSFULLY"
        )

        print(
            f"Run ID    : {run_id}"
        )

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print(
            "======================================"
        )

    return train_component