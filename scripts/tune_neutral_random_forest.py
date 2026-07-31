"""Tune and inspect the contributor-neutral Random Forest."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.discovery import load_dataset
from src.models.feature_schema import (
    PR_IDENTIFIER_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)
from src.models.merge_outcome_tuning import (
    FINAL_MODEL_NAME,
    calculate_permutation_importance,
    create_selected_predictions,
    get_selected_candidate,
    tune_neutral_random_forest,
    validate_tuning_results,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    PROJECT_ROOT,
    REPORTS_DIRECTORY,
)

MODEL_DIRECTORY = PROJECT_ROOT / "models" / "merge_outcome"


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


def load_model_split(
    file_path: Path,
    expected_split: str,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """Load one preprocessed model split."""

    if not file_path.exists():
        raise FileNotFoundError(f"Missing model split: {file_path}")

    dataframe = load_dataset(file_path)

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        TARGET_COLUMN,
        SPLIT_COLUMN,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"{file_path.name} is missing: {sorted(missing_columns)}")

    split_values = set(
        dataframe[SPLIT_COLUMN].astype(str).str.strip().str.lower().unique()
    )

    if split_values != {expected_split}:
        raise ValueError(
            f"{file_path.name} contains unexpected split values: {sorted(split_values)}"
        )

    identifiers = dataframe[PR_IDENTIFIER_COLUMN].copy()

    targets = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="raise",
    ).astype(int)

    features = dataframe.drop(
        columns=[
            PR_IDENTIFIER_COLUMN,
            TARGET_COLUMN,
            SPLIT_COLUMN,
        ]
    ).astype(float)

    return (
        identifiers,
        targets,
        features,
    )


def main() -> int:
    """Run Stage 5E tuning."""

    train_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_train.csv"

    validation_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_validation.csv"

    test_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_test.csv"

    stage_5d_path = REPORTS_DIRECTORY / "stage_5d_completion_report.json"

    if not stage_5d_path.exists():
        print("FAIL: Stage 5D completion report is missing.")
        return 1

    with stage_5d_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        stage_5d_report = json.load(report_file)

    if not stage_5d_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 5D did not pass.")
        return 1

    if not test_path.exists():
        print("FAIL: Reserved test file is missing.")
        return 1

    try:
        (
            train_identifiers,
            y_train,
            x_train,
        ) = load_model_split(
            train_path,
            expected_split="train",
        )

        (
            validation_identifiers,
            y_validation,
            x_validation,
        ) = load_model_split(
            validation_path,
            expected_split="validation",
        )
    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    if len(train_identifiers) != 440:
        print("FAIL: Expected 440 training identifiers.")
        return 1

    if len(validation_identifiers) != 84:
        print("FAIL: Expected 84 validation identifiers.")
        return 1

    (
        results,
        comparison,
    ) = tune_neutral_random_forest(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    selected_result = get_selected_candidate(
        results=results,
        comparison=comparison,
    )

    importance_table = calculate_permutation_importance(
        selected_result=(selected_result),
        x_validation=x_validation,
        y_validation=y_validation,
        repeats=30,
    )

    predictions = create_selected_predictions(
        selected_result=(selected_result),
        validation_identifiers=(validation_identifiers),
        y_validation=y_validation,
    )

    threshold_tables = []

    for result in results.values():
        threshold_table = result.threshold_table.copy()

        threshold_table.insert(
            0,
            "candidate_id",
            result.candidate_id,
        )

        threshold_table.insert(
            1,
            "feature_set_name",
            result.feature_set_name,
        )

        threshold_table.insert(
            2,
            "configuration_name",
            result.parameters["configuration_name"],
        )

        threshold_table["selected_for_candidate"] = threshold_table["threshold"].round(
            6
        ) == round(
            result.selected_threshold,
            6,
        )

        threshold_tables.append(threshold_table)

    all_thresholds = pd.concat(
        threshold_tables,
        ignore_index=True,
    )

    tuning_validation = validate_tuning_results(
        results=results,
        comparison=comparison,
        expected_validation_rows=84,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_model_path = MODEL_DIRECTORY / f"{FINAL_MODEL_NAME}.joblib"

    joblib.dump(
        selected_result.model,
        selected_model_path,
    )

    selected_feature_path = REPORTS_DIRECTORY / "stage_5e_selected_features.csv"

    comparison_path = REPORTS_DIRECTORY / "stage_5e_tuning_comparison.csv"

    thresholds_path = REPORTS_DIRECTORY / "stage_5e_threshold_analysis.csv"

    importance_path = REPORTS_DIRECTORY / "stage_5e_permutation_importance.csv"

    predictions_path = REPORTS_DIRECTORY / "stage_5e_validation_predictions.csv"

    completion_path = REPORTS_DIRECTORY / "stage_5e_completion_report.json"

    selected_features = pd.DataFrame(
        {
            "feature_position": range(
                1,
                len(selected_result.feature_names) + 1,
            ),
            "feature": (selected_result.feature_names),
        }
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8",
    )

    all_thresholds.to_csv(
        thresholds_path,
        index=False,
        encoding="utf-8",
    )

    importance_table.to_csv(
        importance_path,
        index=False,
        encoding="utf-8",
    )

    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
    )

    selected_features.to_csv(
        selected_feature_path,
        index=False,
        encoding="utf-8",
    )

    artifact_validation = {
        "selected_model_exists": (selected_model_path.exists()),
        "selected_model_size_bytes": (
            selected_model_path.stat().st_size if selected_model_path.exists() else 0
        ),
        "comparison_exists": (comparison_path.exists()),
        "thresholds_exist": (thresholds_path.exists()),
        "importance_exists": (importance_path.exists()),
        "predictions_exist": (predictions_path.exists()),
        "selected_features_exist": (selected_feature_path.exists()),
        "test_set_used": False,
    }

    artifact_validation["validation_passed"] = (
        artifact_validation["selected_model_exists"]
        and artifact_validation["selected_model_size_bytes"] > 0
        and artifact_validation["comparison_exists"]
        and artifact_validation["thresholds_exist"]
        and artifact_validation["importance_exists"]
        and artifact_validation["predictions_exist"]
        and artifact_validation["selected_features_exist"]
        and not artifact_validation["test_set_used"]
    )

    overall_passed = (
        tuning_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    baseline_metrics = stage_5d_report["fairness_comparison"]

    baseline_roc_auc = float(baseline_metrics["neutral_random_forest_roc_auc"])

    baseline_accuracy = float(baseline_metrics["neutral_random_forest_accuracy"])

    selected_metrics = selected_result.validation_metrics

    improvement_summary = {
        "stage_5d_roc_auc": (baseline_roc_auc),
        "stage_5e_roc_auc": float(selected_metrics["roc_auc"]),
        "roc_auc_change": float(selected_metrics["roc_auc"] - baseline_roc_auc),
        "stage_5d_accuracy": (baseline_accuracy),
        "stage_5e_accuracy": float(selected_metrics["accuracy"]),
        "accuracy_change": float(selected_metrics["accuracy"] - baseline_accuracy),
    }

    completion_report = {
        "generated_at": (datetime.now(UTC)),
        "stage": "Stage 5E",
        "target": TARGET_COLUMN,
        "training_rows": len(x_train),
        "validation_rows": len(x_validation),
        "test_rows_reserved": 76,
        "test_set_used": False,
        "candidate_count": len(results),
        "selected_candidate_id": (selected_result.candidate_id),
        "selected_feature_set": (selected_result.feature_set_name),
        "selected_feature_count": len(selected_result.feature_names),
        "selected_features": (selected_result.feature_names),
        "selected_parameters": (selected_result.parameters),
        "selected_threshold": (selected_result.selected_threshold),
        "selected_validation_metrics": (selected_metrics),
        "training_metrics": (selected_result.training_metrics),
        "improvement_summary": (improvement_summary),
        "selected_model_path": str(selected_model_path),
        "tuning_validation": (tuning_validation),
        "artifact_validation": (artifact_validation),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_path,
    )

    display_columns = [
        "validation_rank",
        "candidate_id",
        "feature_set_name",
        "configuration_name",
        "feature_count",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "threshold",
        "train_roc_auc",
        "roc_auc_gap",
        "selected_candidate",
    ]

    print("Stage 5E contributor-neutral Random Forest tuning")
    print("=" * 120)

    print()
    print("Top validation candidates:")

    print(comparison[display_columns].head(12).to_string(index=False))

    print()
    print("Selected configuration:")

    print(
        json.dumps(
            {
                "candidate_id": (selected_result.candidate_id),
                "feature_set_name": (selected_result.feature_set_name),
                "feature_count": len(selected_result.feature_names),
                "parameters": (selected_result.parameters),
                "threshold": (selected_result.selected_threshold),
                "validation_metrics": (selected_metrics),
            },
            indent=2,
            default=str,
        )
    )

    print()
    print("Improvement over Stage 5D:")

    print(
        json.dumps(
            improvement_summary,
            indent=2,
        )
    )

    print()
    print("Top validation permutation importance:")

    print(importance_table.head(15).to_string(index=False))

    print()
    print("Tuning validation:")

    print(
        json.dumps(
            tuning_validation,
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
        "Overall Stage 5E verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(completion_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
