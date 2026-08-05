from typing import NamedTuple
from kfp import dsl
from kfp.dsl import component


def create_evaluate_component(base_image: str):

    @component(base_image=base_image)
    def evaluate_component(
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        min_accuracy: float,
        min_precision: float,
        min_recall: float,
        min_f1: float,
        evaluation_status_output: dsl.OutputPath(str),
    ) -> NamedTuple("Outputs", [("evaluation_passed", bool)]):
        import json

        # ---------------------------------------------------------
        # Model Evaluation Logging
        # ---------------------------------------------------------
        print("======================================")
        print("Model Evaluation")
        print("======================================")
        print(f"Accuracy         : {accuracy:.4f} (Min: {min_accuracy:.4f})")
        print(f"Precision        : {precision:.4f} (Min: {min_precision:.4f})")
        print(f"Recall           : {recall:.4f} (Min: {min_recall:.4f})")
        print(f"F1 Score         : {f1_score:.4f} (Min: {min_f1:.4f})")

        # ---------------------------------------------------------
        # Evaluate Metrics
        # ---------------------------------------------------------
        accuracy_passed = accuracy >= min_accuracy
        precision_passed = precision >= min_precision
        recall_passed = recall >= min_recall
        f1_passed = f1_score >= min_f1

        evaluation_passed = all(
            [
                accuracy_passed,
                precision_passed,
                recall_passed,
                f1_passed,
            ]
        )

        # ---------------------------------------------------------
        # Evaluation Result Summary
        # ---------------------------------------------------------
        evaluation_result = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "min_accuracy": min_accuracy,
            "min_precision": min_precision,
            "min_recall": min_recall,
            "min_f1": min_f1,
            "accuracy_passed": accuracy_passed,
            "precision_passed": precision_passed,
            "recall_passed": recall_passed,
            "f1_passed": f1_passed,
            "evaluation_passed": evaluation_passed,
        }

        # Write Evaluation Output artifact JSON file
        with open(evaluation_status_output, "w") as f:
            json.dump(evaluation_result, f, indent=2)

        # ---------------------------------------------------------
        # Fail Pipeline if Model is Below Threshold
        # ---------------------------------------------------------
        if not evaluation_passed:
            print("======================================")
            print("MODEL EVALUATION FAILED")
            print("The model did not meet the minimum quality thresholds.")
            print("The pipeline will stop.")
            print("======================================")
            raise RuntimeError(
                "Model evaluation failed. Model does not meet required quality thresholds."
            )

        print("======================================")
        print("MODEL EVALUATION PASSED")
        print("The model meets all required quality thresholds.")
        print("======================================")

        return (evaluation_passed,)

    return evaluate_component