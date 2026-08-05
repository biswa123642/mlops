import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from kfp import compiler

# ---------------------------------------------------------
# 1. Load Environment Variables First
# ---------------------------------------------------------
load_dotenv()

# Ensure base image environment variable exists prior to pipeline loading
DEFAULT_BASE_IMAGE = "myregistry.azurecr.io/mlops/customer-churn:latest"

if "ML_PIPELINE_BASE_IMAGE" not in os.environ:
    print("[WARNING] 'ML_PIPELINE_BASE_IMAGE' is not set in .env or system environment.")
    print(f"[INFO] Falling back to default container image: {DEFAULT_BASE_IMAGE}")
    os.environ["ML_PIPELINE_BASE_IMAGE"] = DEFAULT_BASE_IMAGE

# ---------------------------------------------------------
# 2. Import Pipeline Definition
# ---------------------------------------------------------
try:
    from customer_churn_pipeline import customer_churn_pipeline
except ImportError as e:
    print(f"[ERROR] Failed to import 'customer_churn_pipeline': {e}")
    sys.exit(1)


def main():
    # ---------------------------------------------------------
    # 3. Output Directory & Artifact Path
    # ---------------------------------------------------------
    output_dir = Path("compiled")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "customer_churn_pipeline.yaml"

    # ---------------------------------------------------------
    # 4. Compile Kubeflow Pipeline
    # ---------------------------------------------------------
    print("======================================")
    print("Compiling Kubeflow Pipeline")
    print(f"Base Image : {os.environ.get('ML_PIPELINE_BASE_IMAGE')}")
    print(f"Target File: {output_file}")
    print("======================================")

    try:
        compiler.Compiler().compile(
            pipeline_func=customer_churn_pipeline,
            package_path=str(output_file),
        )
    except Exception as e:
        print("======================================")
        print("COMPILATION FAILED")
        print(f"Error: {e}")
        print("======================================")
        raise e

    # ---------------------------------------------------------
    # 5. Success Confirmation
    # ---------------------------------------------------------
    print("======================================")
    print("Kubeflow Pipeline Compiled Successfully!")
    print(f"Pipeline Package: {output_file.resolve()}")
    print("======================================")


if __name__ == "__main__":
    main()