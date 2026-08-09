import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract_type",
    "internet_service",
]

TARGET_COLUMN = "churn"

NUMERIC_COLUMNS = [
    "tenure",
    "monthly_charges",
    "support_calls",
]

CATEGORICAL_COLUMNS = [
    "contract_type",
    "internet_service",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the customer churn model"
    )

    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to the input CSV file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where model.pkl will be saved",
    )

    return parser.parse_args()


def load_data(input_path):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input file was not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    df = df[required_columns].copy()

    if df.isnull().any().any():
        raise ValueError(
            "Input dataset contains missing values."
        )

    invalid_targets = set(df[TARGET_COLUMN].unique()) - {0, 1}

    if invalid_targets:
        raise ValueError(
            f"Invalid churn values found: {invalid_targets}. "
            "Expected only 0 and 1."
        )

    return df


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    args = parse_arguments()

    df = load_data(args.input_data)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The target column must contain both classes: 0 and 1."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()

    mlflow.sklearn.autolog(
        log_models=False,
        log_input_examples=False,
        log_model_signatures=False,
    )

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
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

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 200,
                "test_size": 0.2,
                "random_state": 42,
                "feature_count": len(FEATURE_COLUMNS),
                "training_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        mlflow.set_tag(
            "model_name",
            "customer-churn-model",
        )

        os.makedirs(args.output_dir, exist_ok=True)

        model_path = os.path.join(
            args.output_dir,
            "model.pkl",
        )

        joblib.dump(pipeline, model_path)

        print("----------------------------------------")
        print("Training completed successfully")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Model saved to: {model_path}")
        print("----------------------------------------")


if __name__ == "__main__":
    main()