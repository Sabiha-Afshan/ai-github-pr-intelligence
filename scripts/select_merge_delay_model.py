"""Select and lock the final Model 2 candidate using validation only."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.data.discovery import load_dataset
from src.models.merge_delay_selection import (
    build_all_candidates,
    build_candidate_comparison,
    build_family_summary,
    evaluate_candidate,
    select_locked_candidate,
)
from src.models.merge_delay_training import (
    load_preprocessed_split,
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

LOCKED_MODEL_PATH = MODEL_DIRECTORY / "locked_merge_delay_model.joblib"

CANDIDATE_COMPARISON_PATH = REPORTS_DIRECTORY / "stage_6e_candidate_comparison.csv"

FAMILY_SUMMARY_PATH = REPORTS_DIRECTORY / "stage_6e_model_family_summary.csv"

SELECTED_THRESHOLDS_PATH = (
    REPORTS_DIRECTORY / "stage_6e_selected_candidate_thresholds.csv"
)

SELECTED_PREDICTIONS_PATH = (
    REPORTS_DIRECTORY / "stage_6e_selected_validation_predictions.csv"
)

LOCKED_CONFIGURATION_PATH = (
    REPORTS_DIRECTORY / "stage_6e_locked_model_configuration.json"
)

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_6e_completion_report.json"


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
    """Run Stage 6E controlled model selection."""

    stage_6d_path = REPORTS_DIRECTORY / "stage_6d_completion_report.json"

    required_paths = [
        stage_6d_path,
        TRAIN_DATA_PATH,
        VALIDATION_DATA_PATH,
        TEST_DATA_PATH,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 6E files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    with stage_6d_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        stage_6d_report = json.load(report_file)

    if not stage_6d_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 6D did not pass verification.")
        return 1

    test_metadata_before = {
        "size_bytes": TEST_DATA_PATH.stat().st_size,
        "modified_time_ns": (TEST_DATA_PATH.stat().st_mtime_ns),
    }

    try:
        training_dataframe = load_dataset(TRAIN_DATA_PATH)

        validation_dataframe = load_dataset(VALIDATION_DATA_PATH)

        training_split = load_preprocessed_split(
            training_dataframe,
            expected_split="train",
        )

        validation_split = load_preprocessed_split(
            validation_dataframe,
            expected_split="validation",
        )

        split_validation = validate_split_compatibility(
            training_split=training_split,
            validation_split=(validation_split),
        )

        if not split_validation["validation_passed"]:
            raise ValueError("Training and validation schemas do not match.")

        candidates = build_all_candidates()

        evaluated_candidates = []

        for candidate in candidates:
            print(
                "Evaluating:",
                candidate.candidate_id,
            )

            evaluated_candidates.append(
                evaluate_candidate(
                    candidate=candidate,
                    training_split=training_split,
                    validation_split=(validation_split),
                )
            )

        comparison = build_candidate_comparison(evaluated_candidates)

        family_summary = build_family_summary(comparison)

        locked_candidate = select_locked_candidate(
            evaluated_candidates=(evaluated_candidates),
            comparison=comparison,
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
        locked_candidate.trained_model,
        LOCKED_MODEL_PATH,
    )

    comparison.to_csv(
        CANDIDATE_COMPARISON_PATH,
        index=False,
        encoding="utf-8",
    )

    family_summary.to_csv(
        FAMILY_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    locked_candidate.threshold_table.to_csv(
        SELECTED_THRESHOLDS_PATH,
        index=False,
        encoding="utf-8",
    )

    locked_candidate.validation_predictions.to_csv(
        SELECTED_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    selected_row = (
        comparison.loc[comparison["selected_for_locking"].astype(bool)]
        .iloc[0]
        .to_dict()
    )

    feature_names = list(training_split.features.columns)

    locked_configuration = {
        "generated_at": datetime.now(UTC),
        "model_name": (locked_candidate.candidate.candidate_id),
        "model_family": (locked_candidate.candidate.model_family),
        "model_path": str(LOCKED_MODEL_PATH),
        "threshold": (locked_candidate.selected_threshold),
        "target": ("merge_delay_target"),
        "target_definition": ("1 when merge_hours exceeds 48 hours; otherwise 0"),
        "population": ("Merged pull requests only"),
        "feature_count": len(feature_names),
        "features": feature_names,
        "parameters": (locked_candidate.candidate.parameters),
        "validation_metrics": (locked_candidate.validation_metrics),
        "selection_policy": (
            "Highest validation balanced accuracy, "
            "then F1, recall, ROC-AUC, average precision, "
            "smaller ROC-AUC gap and threshold proximity "
            "to 0.50."
        ),
        "test_used_for_selection": False,
    }

    save_json(
        locked_configuration,
        LOCKED_CONFIGURATION_PATH,
    )

    test_metadata_after = {
        "size_bytes": TEST_DATA_PATH.stat().st_size,
        "modified_time_ns": (TEST_DATA_PATH.stat().st_mtime_ns),
    }

    test_untouched = test_metadata_before == test_metadata_after

    artifact_validation = {
        "locked_model_exists": (LOCKED_MODEL_PATH.exists()),
        "candidate_comparison_exists": (CANDIDATE_COMPARISON_PATH.exists()),
        "family_summary_exists": (FAMILY_SUMMARY_PATH.exists()),
        "selected_thresholds_exist": (SELECTED_THRESHOLDS_PATH.exists()),
        "selected_predictions_exist": (SELECTED_PREDICTIONS_PATH.exists()),
        "locked_configuration_exists": (LOCKED_CONFIGURATION_PATH.exists()),
        "test_split_untouched": (test_untouched),
    }

    artifact_validation["validation_passed"] = all(artifact_validation.values())

    overall_passed = (
        split_validation["validation_passed"]
        and len(comparison) == 19
        and comparison["selected_for_locking"].sum() == 1
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 6E",
        "model": ("Model 2 — Merge-delay prediction"),
        "candidate_count": len(comparison),
        "logistic_candidate_count": int(
            (comparison["model_family"] == "logistic_regression").sum()
        ),
        "random_forest_candidate_count": int(
            (comparison["model_family"] == "random_forest").sum()
        ),
        "hist_gradient_boosting_candidate_count": int(
            (comparison["model_family"] == "hist_gradient_boosting").sum()
        ),
        "selected_candidate": (selected_row),
        "locked_configuration": (locked_configuration),
        "split_validation": (split_validation),
        "artifact_validation": (artifact_validation),
        "test_policy": {
            "test_dataset_loaded": False,
            "test_predictions_generated": False,
            "test_metrics_calculated": False,
            "test_split_untouched": (test_untouched),
        },
        "model_locked": True,
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    display_columns = [
        "validation_rank",
        "candidate_id",
        "model_family",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "roc_auc_gap",
        "overfitting_warning",
        "selected_for_locking",
    ]

    print()
    print("Stage 6E controlled Model 2 selection")
    print("=" * 110)

    print()
    print("Top validation candidates:")

    print(comparison[display_columns].head(10).to_string(index=False))

    print()
    print("Strongest candidate by family:")

    print(family_summary[display_columns].to_string(index=False))

    print()
    print("Locked Model 2 configuration:")

    print(
        json.dumps(
            locked_configuration,
            indent=2,
            default=str,
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
        "Overall Stage 6E verification passed:",
        overall_passed,
    )

    print()
    print("Locked model:")
    print(LOCKED_MODEL_PATH)

    print()
    print("Completion report:")
    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
