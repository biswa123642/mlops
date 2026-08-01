import os

from kfp import dsl

from components.train_component import (
    create_train_component,
)

from components.evaluate_component import (
    create_evaluate_component,
)

from components.register_model_component import (
    create_register_model_component,
)

from components.deploy_kserve_component import (
    create_deploy_kserve_component,
)


# =========================================================
# Dynamic ACR Base Image
# =========================================================

BASE_IMAGE = os.environ[
    "ML_PIPELINE_BASE_IMAGE"
]


# =========================================================
# Create Components Using Dynamic ACR Image
# =========================================================

train_component = create_train_component(
    base_image=BASE_IMAGE
)

evaluate_component = create_evaluate_component(
    base_image=BASE_IMAGE
)

register_model_component = (
    create_register_model_component(
        base_image=BASE_IMAGE
    )
)

deploy_kserve_component = (
    create_deploy_kserve_component(
        base_image=BASE_IMAGE
    )
)


@dsl.pipeline(
    name="customer-churn-ml-pipeline",
    description=(
        "Production ML pipeline for training, evaluating, "
        "registering, and deploying the customer churn model."
    ),
)
def customer_churn_pipeline(
    # ---------------------------------------------------------
    # Azure Blob Storage
    # ---------------------------------------------------------

    azure_storage_account: str,

    azure_storage_container: str,

    dataset_name: str = "customer_churn.csv",

    # ---------------------------------------------------------
    # MLflow
    # ---------------------------------------------------------

    mlflow_tracking_uri: str = "",

    experiment_name: str = "customer-churn",

    registered_model_name: str = "customer-churn-model",

    model_alias: str = "Production",

    # ---------------------------------------------------------
    # Model Evaluation Thresholds
    # ---------------------------------------------------------

    min_accuracy: float = 0.80,

    min_precision: float = 0.75,

    min_recall: float = 0.75,

    min_f1: float = 0.75,

    # ---------------------------------------------------------
    # KServe
    # ---------------------------------------------------------

    namespace: str = "mlops",

    inference_service_name: str = "customer-churn",
):

    # =========================================================
    # 1. TRAIN MODEL
    # =========================================================

    train_task = train_component(

        azure_storage_account=(
            azure_storage_account
        ),

        azure_storage_container=(
            azure_storage_container
        ),

        dataset_name=(
            dataset_name
        ),

        mlflow_tracking_uri=(
            mlflow_tracking_uri
        ),

        experiment_name=(
            experiment_name
        ),
    )

    # =========================================================
    # 2. EVALUATE MODEL
    # =========================================================

    evaluate_task = evaluate_component(

        accuracy=(
            train_task.outputs[
                "accuracy_output"
            ]
        ),

        precision=(
            train_task.outputs[
                "precision_output"
            ]
        ),

        recall=(
            train_task.outputs[
                "recall_output"
            ]
        ),

        f1_score=(
            train_task.outputs[
                "f1_output"
            ]
        ),

        min_accuracy=(
            min_accuracy
        ),

        min_precision=(
            min_precision
        ),

        min_recall=(
            min_recall
        ),

        min_f1=(
            min_f1
        ),
    )

    # =========================================================
    # 3. REGISTER MODEL
    #
    # This task runs only if evaluation succeeds.
    # =========================================================

    register_task = register_model_component(

        mlflow_tracking_uri=(
            mlflow_tracking_uri
        ),

        registered_model_name=(
            registered_model_name
        ),

        run_id=(
            train_task.outputs[
                "run_id_output"
            ]
        ),

        model_alias=(
            model_alias
        ),
    )

    register_task.after(
        evaluate_task
    )

    # =========================================================
    # 4. DEPLOY MODEL TO KSERVE
    #
    # This task runs only after successful registration.
    # =========================================================

    deploy_task = deploy_kserve_component(

        mlflow_tracking_uri=(
            mlflow_tracking_uri
        ),

        registered_model_name=(
            registered_model_name
        ),

        model_alias=(
            model_alias
        ),

        namespace=(
            namespace
        ),

        inference_service_name=(
            inference_service_name
        ),
    )

    deploy_task.after(
        register_task
    )