from pathlib import Path

from kfp import compiler

from customer_churn_pipeline import (
    customer_churn_pipeline,
)


def main():

    # ---------------------------------------------------------
    # Pipeline Output Directory
    # ---------------------------------------------------------

    output_dir = Path(
        "compiled"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Compiled Pipeline Package
    # ---------------------------------------------------------

    output_file = (
        output_dir
        / "customer_churn_pipeline.yaml"
    )

    # ---------------------------------------------------------
    # Compile Kubeflow Pipeline
    # ---------------------------------------------------------

    print(
        "======================================"
    )

    print(
        "Compiling Kubeflow Pipeline"
    )

    print(
        f"Output: {output_file}"
    )

    print(
        "======================================"
    )

    compiler.Compiler().compile(

        pipeline_func=(
            customer_churn_pipeline
        ),

        package_path=str(
            output_file
        ),
    )

    # ---------------------------------------------------------
    # Success
    # ---------------------------------------------------------

    print(
        "======================================"
    )

    print(
        "Kubeflow Pipeline Compiled Successfully"
    )

    print(
        f"Pipeline Package: "
        f"{output_file}"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":

    main()