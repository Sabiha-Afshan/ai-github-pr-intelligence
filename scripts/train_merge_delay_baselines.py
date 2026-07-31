"""Train and compare Model 2 baseline classifiers."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.discovery import load_dataset
from src.models.merge_delay_evaluation import (
    compare_validation_models,
)
from src.models.merge_delay_training import (
    load_preprocessed_split,
    train_logistic_regression,
    train_random_forest,
    validate_split_compatibility,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    PROJECT_ROOT,
    REPORTS_DIRECTORY,
)

MODEL_DIRECTORY = PROJECT_ROOT / "models" / "merge_delay"

TRAIN_DATA_PATH = PROCESSED_DATA_DIRECTORY / "model2_preprocessed_train.csv"

VALIDATION_DATA_PATH = PROCESSED_DATA_DIRECTORY / "model2_preprocessed_validation.csv"

TEST_DATA_PATH = PROCESSED_DATA_DIRECTORY / "model2_preprocessed_test.csv"

LOGISTIC_MODEL_PATH = MODEL_DIRECTORY / "merge_delay_logistic_regression.joblib"

RANDOM_FOREST_MODEL_PATH = MODEL_DIRECTORY / "merge_delay_random_forest.joblib"

COMPARISON_PATH = REPORTS_DIRECTORY / "stage_6d_validation_model_comparison.csv"

TRAINING_METRICS_PATH = REPORTS_DIRECTORY / "stage_6d_training_metrics.csv"

VALIDATION_PREDICTIONS_PATH = REPORTS_DIRECTORY / "stage_6d_validation_predictions.csv"

LOGISTIC_THRESHOLDS_PATH = REPORTS_DIRECTORY / "stage_6d_logistic_thresholds.csv"

RANDOM_FOREST_THRESHOLDS_PATH = (
    REPORTS_DIRECTORY / "stage_6d_random_forest_thresholds.csv"
)

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_6d_completion_report.json"


def save_json(
    payload: Any,
    output_path: Path,
) -> None:
    """Save JSON atomically."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            payload,
            output_file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    temporary_path.replace(output_path)


def main() -> int:
    """Run Stage 6D baseline training and validation."""

    stage_6c_path = REPORTS_DIRECTORY / "stage_6c_completion_report.json"

    required_paths = [
        stage_6c_path,
        TRAIN_DATA_PATH,
        VALIDATION_DATA_PATH,
        TEST_DATA_PATH,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 6D files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    with stage_6c_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        stage_6c_report = json.load(report_file)

    if not stage_6c_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 6C did not pass verification.")
        return 1

    test_file_metadata_before = {
        "path": str(TEST_DATA_PATH),
        "size_bytes": (TEST_DATA_PATH.stat().st_size),
        "modified_time_ns": (TEST_DATA_PATH.stat().st_mtime_ns),
    }

    try:
        train_dataframe = load_dataset(TRAIN_DATA_PATH)

        validation_dataframe = load_dataset(VALIDATION_DATA_PATH)

        training_split = load_preprocessed_split(
            train_dataframe,
            expected_split="train",
        )

        validation_split = load_preprocessed_split(
            validation_dataframe,
            expected_split="validation",
        )

        split_validation = validate_split_compatibility(
            training_split=(training_split),
            validation_split=(validation_split),
        )

        if not split_validation["validation_passed"]:
            raise ValueError(
                "Training and validation splits failed compatibility checks."
            )

        logistic_result = train_logistic_regression(
            training_split=(training_split),
            validation_split=(validation_split),
        )

        random_forest_result = train_random_forest(
            training_split=(training_split),
            validation_split=(validation_split),
        )

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        logistic_result.model,
        LOGISTIC_MODEL_PATH,
    )

    joblib.dump(
        random_forest_result.model,
        RANDOM_FOREST_MODEL_PATH,
    )

    validation_comparison = compare_validation_models(
        [
            logistic_result.validation_metrics,
            random_forest_result.validation_metrics,
        ]
    )

    training_metrics = pd.DataFrame(
        [
            logistic_result.training_metrics,
            random_forest_result.training_metrics,
        ]
    )

    validation_predictions = pd.concat(
        [
            logistic_result.validation_predictions,
            random_forest_result.validation_predictions,
        ],
        ignore_index=True,
    )

    validation_comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8",
    )

    training_metrics.to_csv(
        TRAINING_METRICS_PATH,
        index=False,
        encoding="utf-8",
    )

    validation_predictions.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    logistic_result.threshold_table.to_csv(
        LOGISTIC_THRESHOLDS_PATH,
        index=False,
        encoding="utf-8",
    )

    random_forest_result.threshold_table.to_csv(
        RANDOM_FOREST_THRESHOLDS_PATH,
        index=False,
        encoding="utf-8",
    )

    test_file_metadata_after = {
        "path": str(TEST_DATA_PATH),
        "size_bytes": (TEST_DATA_PATH.stat().st_size),
        "modified_time_ns": (TEST_DATA_PATH.stat().st_mtime_ns),
    }

    test_untouched = test_file_metadata_before == test_file_metadata_after

    nominated_row = (
        validation_comparison.loc[validation_comparison["nominated_candidate"]]
        .iloc[0]
        .to_dict()
    )

    artifact_validation = {
        "logistic_model_exists": (LOGISTIC_MODEL_PATH.exists()),
        "random_forest_model_exists": (RANDOM_FOREST_MODEL_PATH.exists()),
        "comparison_exists": (COMPARISON_PATH.exists()),
        "training_metrics_exists": (TRAINING_METRICS_PATH.exists()),
        "validation_predictions_exist": (VALIDATION_PREDICTIONS_PATH.exists()),
        "logistic_thresholds_exist": (LOGISTIC_THRESHOLDS_PATH.exists()),
        "random_forest_thresholds_exist": (RANDOM_FOREST_THRESHOLDS_PATH.exists()),
        "test_split_untouched": (test_untouched),
    }

    artifact_validation["validation_passed"] = all(artifact_validation.values())

    training_metric_lookup = training_metrics.set_index("model_name").to_dict(
        orient="index"
    )

    overfitting_summary = {}

    for model_name in (
        "merge_delay_logistic_regression",
        "merge_delay_random_forest",
    ):
        training_roc_auc = float(training_metric_lookup[model_name]["roc_auc"])

        validation_roc_auc = float(
            validation_comparison.loc[
                validation_comparison["model_name"] == model_name,
                "roc_auc",
            ].iloc[0]
        )

        overfitting_summary[model_name] = {
            "training_roc_auc": (training_roc_auc),
            "validation_roc_auc": (validation_roc_auc),
            "roc_auc_gap": (training_roc_auc - validation_roc_auc),
        }

    overall_passed = (
        split_validation["validation_passed"]
        and len(validation_comparison) == 2
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 6D",
        "model": ("Model 2 — Merge-delay prediction"),
        "training_rows": len(training_split.features),
        "validation_rows": len(validation_split.features),
        "test_rows": 38,
        "feature_count": len(training_split.features.columns),
        "split_validation": (split_validation),
        "validation_comparison": (validation_comparison.to_dict(orient="records")),
        "nominated_candidate": (nominated_row),
        "overfitting_summary": (overfitting_summary),
        "artifact_validation": (artifact_validation),
        "test_policy": {
            "test_dataset_loaded": False,
            "test_predictions_generated": False,
            "test_metrics_calculated": False,
            "test_split_untouched": (test_untouched),
        },
        "selection_status": (
            "Validation nominee only. No model is locked at Stage 6D."
        ),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    print("Stage 6D Model 2 baseline training and validation")
    print("=" * 100)

    print()
    print("Split validation:")

    print(
        json.dumps(
            split_validation,
            indent=2,
        )
    )

    print()
    print("Validation model comparison:")

    print(
        validation_comparison[
            [
                "validation_rank",
                "model_name",
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
                "nominated_candidate",
            ]
        ].to_string(index=False)
    )

    print()
    print("Training-to-validation ROC-AUC gaps:")

    print(
        json.dumps(
            overfitting_summary,
            indent=2,
        )
    )

    print()
    print("Nominated validation candidate:")

    print(
        json.dumps(
            nominated_row,
            indent=2,
        )
    )

    print()
    print("Artifact validation:")

    print(
        json.dumps(
            artifact_validation,
            indent=2,
        )
    )

    print()
    print(
        "Test dataset loaded:",
        False,
    )

    print(
        "Test predictions generated:",
        False,
    )

    print(
        "Test split untouched:",
        test_untouched,
    )

    print()
    print(
        "Overall Stage 6D verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
