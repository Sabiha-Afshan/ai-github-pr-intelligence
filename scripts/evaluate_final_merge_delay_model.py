"""Evaluate the locked Model 2 on the final chronological test set."""

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.data.discovery import load_dataset
from src.models.merge_delay_final_evaluation import (
    build_classification_report_table,
    build_error_table,
    build_validation_test_comparison,
    evaluate_locked_model,
    load_final_test_split,
    validate_final_outputs,
    validate_locked_configuration,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    PROJECT_ROOT,
    REPORTS_DIRECTORY,
)

MODEL_DIRECTORY = PROJECT_ROOT / "models" / "merge_delay"

LOCKED_MODEL_PATH = MODEL_DIRECTORY / "locked_merge_delay_model.joblib"

FINAL_MODEL_PATH = MODEL_DIRECTORY / "final_merge_delay_model.joblib"

TEST_DATA_PATH = PROCESSED_DATA_DIRECTORY / "model2_preprocessed_test.csv"

LOCKED_CONFIGURATION_PATH = (
    REPORTS_DIRECTORY / "stage_6e_locked_model_configuration.json"
)

STAGE_6E_COMPLETION_PATH = REPORTS_DIRECTORY / "stage_6e_completion_report.json"

TEST_PREDICTIONS_PATH = REPORTS_DIRECTORY / "stage_6f_test_predictions.csv"

TEST_METRICS_PATH = REPORTS_DIRECTORY / "stage_6f_test_metrics.json"

CLASSIFICATION_REPORT_PATH = REPORTS_DIRECTORY / "stage_6f_classification_report.csv"

ERROR_REPORT_PATH = REPORTS_DIRECTORY / "stage_6f_test_errors.csv"

HIGH_CONFIDENCE_ERROR_PATH = REPORTS_DIRECTORY / "stage_6f_high_confidence_errors.csv"

VALIDATION_TEST_COMPARISON_PATH = (
    REPORTS_DIRECTORY / "stage_6f_validation_test_comparison.csv"
)

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_6f_completion_report.json"


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
    """Run the final Model 2 test evaluation."""

    required_paths = [
        LOCKED_MODEL_PATH,
        TEST_DATA_PATH,
        LOCKED_CONFIGURATION_PATH,
        STAGE_6E_COMPLETION_PATH,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 6F files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    with STAGE_6E_COMPLETION_PATH.open(
        "r",
        encoding="utf-8",
    ) as completion_file:
        stage_6e_report = json.load(completion_file)

    if not stage_6e_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 6E did not pass verification.")
        return 1

    if not stage_6e_report.get(
        "model_locked",
        False,
    ):
        print("FAIL: Model 2 was not locked in Stage 6E.")
        return 1

    with LOCKED_CONFIGURATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as configuration_file:
        configuration = json.load(configuration_file)

    configuration_validation = validate_locked_configuration(configuration)

    if not configuration_validation["validation_passed"]:
        print("FAIL: Locked configuration did not pass validation.")

        print(
            json.dumps(
                configuration_validation,
                indent=2,
            )
        )

        return 1

    locked_threshold = float(configuration["threshold"])

    feature_names = [str(feature) for feature in configuration["features"]]

    model_name = str(configuration["model_name"])

    model_metadata_before = {
        "size_bytes": (LOCKED_MODEL_PATH.stat().st_size),
        "modified_time_ns": (LOCKED_MODEL_PATH.stat().st_mtime_ns),
    }

    try:
        test_dataframe = load_dataset(TEST_DATA_PATH)

        (
            test_identifiers,
            y_test,
            x_test,
        ) = load_final_test_split(
            dataframe=test_dataframe,
            feature_names=feature_names,
        )

        locked_model = joblib.load(LOCKED_MODEL_PATH)

        (
            test_metrics,
            test_predictions,
        ) = evaluate_locked_model(
            model=locked_model,
            identifiers=test_identifiers,
            targets=y_test,
            features=x_test,
            threshold=locked_threshold,
            model_name=model_name,
        )

        classification_table = build_classification_report_table(test_predictions)

        error_table = build_error_table(test_predictions)

        high_confidence_errors = (
            error_table.loc[error_table["high_confidence_error"].astype(bool)]
            .copy()
            .reset_index(drop=True)
        )

        validation_test_comparison = build_validation_test_comparison(
            validation_metrics=(configuration["validation_metrics"]),
            test_metrics=test_metrics,
        )

        output_validation = validate_final_outputs(
            metrics=test_metrics,
            predictions=test_predictions,
            classification_table=(classification_table),
            error_table=error_table,
            expected_threshold=(locked_threshold),
        )

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    model_metadata_after = {
        "size_bytes": (LOCKED_MODEL_PATH.stat().st_size),
        "modified_time_ns": (LOCKED_MODEL_PATH.stat().st_mtime_ns),
    }

    locked_model_unchanged = model_metadata_before == model_metadata_after

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_predictions.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    classification_table.to_csv(
        CLASSIFICATION_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    error_table.to_csv(
        ERROR_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    high_confidence_errors.to_csv(
        HIGH_CONFIDENCE_ERROR_PATH,
        index=False,
        encoding="utf-8",
    )

    validation_test_comparison.to_csv(
        VALIDATION_TEST_COMPARISON_PATH,
        index=False,
        encoding="utf-8",
    )

    save_json(
        test_metrics,
        TEST_METRICS_PATH,
    )

    shutil.copy2(
        LOCKED_MODEL_PATH,
        FINAL_MODEL_PATH,
    )

    artifact_validation = {
        "final_model_exists": (FINAL_MODEL_PATH.exists()),
        "test_predictions_exist": (TEST_PREDICTIONS_PATH.exists()),
        "test_metrics_exist": (TEST_METRICS_PATH.exists()),
        "classification_report_exists": (CLASSIFICATION_REPORT_PATH.exists()),
        "error_report_exists": (ERROR_REPORT_PATH.exists()),
        "high_confidence_error_report_exists": (HIGH_CONFIDENCE_ERROR_PATH.exists()),
        "validation_test_comparison_exists": (VALIDATION_TEST_COMPARISON_PATH.exists()),
        "locked_model_unchanged": (locked_model_unchanged),
        "threshold_changed": False,
        "features_changed": False,
        "model_retrained": False,
    }

    artifact_validation["validation_passed"] = (
        artifact_validation["final_model_exists"]
        and artifact_validation["test_predictions_exist"]
        and artifact_validation["test_metrics_exist"]
        and artifact_validation["classification_report_exists"]
        and artifact_validation["error_report_exists"]
        and artifact_validation["high_confidence_error_report_exists"]
        and artifact_validation["validation_test_comparison_exists"]
        and artifact_validation["locked_model_unchanged"]
        and not artifact_validation["threshold_changed"]
        and not artifact_validation["features_changed"]
        and not artifact_validation["model_retrained"]
    )

    overall_passed = (
        configuration_validation["validation_passed"]
        and output_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 6F",
        "model": ("Model 2 — Merge-delay prediction"),
        "model_name": model_name,
        "model_family": configuration["model_family"],
        "target": configuration["target"],
        "threshold": locked_threshold,
        "test_rows": len(test_predictions),
        "feature_count": len(feature_names),
        "test_metrics": test_metrics,
        "validation_test_comparison": (
            validation_test_comparison.to_dict(orient="records")
        ),
        "error_count": len(error_table),
        "high_confidence_error_count": len(high_confidence_errors),
        "configuration_validation": (configuration_validation),
        "output_validation": (output_validation),
        "artifact_validation": (artifact_validation),
        "final_model_path": str(FINAL_MODEL_PATH),
        "final_evaluation_policy": {
            "model_retrained": False,
            "features_changed": False,
            "threshold_changed": False,
            "test_used_for_model_selection": False,
            "test_used_for_threshold_selection": False,
            "test_used_for_final_evaluation": True,
            "future_changes_require_new_holdout_data": True,
        },
        "interpretation_warning": (
            "The test split contains only 38 chronological "
            "records. Report exact metrics but do not imply "
            "universal performance across repositories or "
            "future time periods."
        ),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    print("Stage 6F final held-out Model 2 evaluation")
    print("=" * 100)

    print()
    print("Locked configuration validation:")

    print(
        json.dumps(
            configuration_validation,
            indent=2,
        )
    )

    print()
    print("Final chronological test metrics:")

    print(
        json.dumps(
            test_metrics,
            indent=2,
        )
    )

    print()
    print("Validation-to-test comparison:")

    print(validation_test_comparison.to_string(index=False))

    print()
    print("Test classification report:")

    print(classification_table.to_string(index=False))

    print()
    print("Test error summary:")

    print(
        json.dumps(
            {
                "total_test_rows": len(test_predictions),
                "correct_predictions": int(
                    test_predictions["prediction_correct"].astype(bool).sum()
                ),
                "incorrect_predictions": len(error_table),
                "false_positives": int(test_metrics["false_positive"]),
                "false_negatives": int(test_metrics["false_negative"]),
                "high_confidence_errors": len(high_confidence_errors),
            },
            indent=2,
        )
    )

    print()
    print("Output validation:")

    print(
        json.dumps(
            output_validation,
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
        "Model retrained:",
        False,
    )

    print(
        "Features changed:",
        False,
    )

    print(
        "Threshold changed:",
        False,
    )

    print()
    print(
        "Overall Stage 6F verification passed:",
        overall_passed,
    )

    print()
    print("Final Model 2 artifact:")
    print(FINAL_MODEL_PATH)

    print()
    print("Completion report:")
    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
