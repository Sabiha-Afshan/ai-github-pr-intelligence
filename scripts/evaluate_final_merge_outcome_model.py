"""Evaluate the locked merge-outcome model on the held-out test set."""

import json
import shutil
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
from src.models.final_model_evaluation import (
    compare_validation_and_test,
    evaluate_locked_model,
    identify_high_confidence_errors,
    validate_final_test_results,
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
    """Save a JSON report atomically."""

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


def load_test_split(
    file_path: Path,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """Load the held-out test split."""

    if not file_path.exists():
        raise FileNotFoundError(f"Test split is missing: {file_path}")

    dataframe = load_dataset(file_path)

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        TARGET_COLUMN,
        SPLIT_COLUMN,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Test dataset is missing columns: {sorted(missing_columns)}")

    split_values = set(
        dataframe[SPLIT_COLUMN].astype(str).str.strip().str.lower().unique()
    )

    if split_values != {"test"}:
        raise ValueError(
            f"Expected only the test split, but found: {sorted(split_values)}"
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
    """Run the one-time Stage 5F test evaluation."""

    stage_5e2_path = REPORTS_DIRECTORY / "stage_5e2_completion_report.json"

    locked_configuration_path = (
        REPORTS_DIRECTORY / "stage_5e2_locked_model_configuration.json"
    )

    locked_model_path = MODEL_DIRECTORY / "locked_merge_outcome_model.joblib"

    test_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_test.csv"

    required_paths = [
        stage_5e2_path,
        locked_configuration_path,
        locked_model_path,
        test_path,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 5F files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    with stage_5e2_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        stage_5e2_report = json.load(report_file)

    if not stage_5e2_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 5E2 did not pass verification.")
        return 1

    with locked_configuration_path.open(
        "r",
        encoding="utf-8",
    ) as configuration_file:
        locked_configuration = json.load(configuration_file)

    if locked_configuration.get(
        "test_set_used",
        True,
    ):
        print(
            "FAIL: Locked configuration indicates that the test set was already used."
        )
        return 1

    required_configuration_fields = {
        "model_name",
        "features",
        "feature_count",
        "threshold",
        "validation_metrics",
    }

    missing_configuration_fields = required_configuration_fields - set(
        locked_configuration
    )

    if missing_configuration_fields:
        print(
            "FAIL: Locked configuration is missing fields: "
            f"{sorted(missing_configuration_fields)}"
        )
        return 1

    model_name = str(locked_configuration["model_name"])

    selected_features = [str(feature) for feature in locked_configuration["features"]]

    selected_threshold = float(locked_configuration["threshold"])

    validation_metrics = locked_configuration["validation_metrics"]

    if len(selected_features) != int(locked_configuration["feature_count"]):
        print("FAIL: Locked feature count does not match the saved feature list.")
        return 1

    try:
        (
            test_identifiers,
            y_test,
            x_test,
        ) = load_test_split(test_path)

        model = joblib.load(locked_model_path)

        (
            test_metrics,
            predictions,
            classification_table,
        ) = evaluate_locked_model(
            model_name=model_name,
            model=model,
            features=x_test,
            targets=y_test,
            identifiers=test_identifiers,
            selected_features=selected_features,
            selected_threshold=selected_threshold,
        )

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    comparison = compare_validation_and_test(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )

    high_confidence_errors = identify_high_confidence_errors(
        predictions,
        minimum_confidence=0.75,
    )

    result_validation = validate_final_test_results(
        metrics=test_metrics,
        predictions=predictions,
        expected_rows=76,
    )

    predictions_path = REPORTS_DIRECTORY / "stage_5f_test_predictions.csv"

    metrics_path = REPORTS_DIRECTORY / "stage_5f_test_metrics.json"

    classification_path = REPORTS_DIRECTORY / "stage_5f_classification_report.csv"

    comparison_path = REPORTS_DIRECTORY / "stage_5f_validation_test_comparison.csv"

    error_path = REPORTS_DIRECTORY / "stage_5f_high_confidence_errors.csv"

    completion_path = REPORTS_DIRECTORY / "stage_5f_completion_report.json"

    final_model_path = (
        MODEL_DIRECTORY / "final_contributor_neutral_merge_outcome_model.joblib"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
    )

    classification_table.to_csv(
        classification_path,
        index=False,
        encoding="utf-8",
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8",
    )

    high_confidence_errors.to_csv(
        error_path,
        index=False,
        encoding="utf-8",
    )

    save_json(
        test_metrics,
        metrics_path,
    )

    shutil.copyfile(
        locked_model_path,
        final_model_path,
    )

    artifact_validation = {
        "final_model_exists": (final_model_path.exists()),
        "final_model_size_bytes": (
            final_model_path.stat().st_size if final_model_path.exists() else 0
        ),
        "predictions_exist": (predictions_path.exists()),
        "metrics_exist": (metrics_path.exists()),
        "classification_report_exists": (classification_path.exists()),
        "comparison_exists": (comparison_path.exists()),
        "error_report_exists": (error_path.exists()),
    }

    artifact_validation["validation_passed"] = (
        artifact_validation["final_model_exists"]
        and artifact_validation["final_model_size_bytes"] > 0
        and all(
            value
            for key, value in artifact_validation.items()
            if key.endswith("_exists")
        )
    )

    overall_passed = (
        result_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 5F",
        "model_name": model_name,
        "model_role": ("Final contributor-neutral merge-outcome model"),
        "test_evaluated_once": True,
        "test_rows": len(x_test),
        "feature_count": len(selected_features),
        "selected_threshold": (selected_threshold),
        "threshold_source": ("Locked Stage 5E2 validation configuration"),
        "validation_metrics": (validation_metrics),
        "test_metrics": test_metrics,
        "high_confidence_error_count": len(high_confidence_errors),
        "result_validation": (result_validation),
        "artifact_validation": (artifact_validation),
        "locked_model_source": str(locked_model_path),
        "final_model_path": str(final_model_path),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_path,
    )

    display_metrics = {
        key: value
        for key, value in test_metrics.items()
        if key
        not in {
            "model_name",
            "evaluation_split",
            "threshold_source",
        }
    }

    print("Stage 5F final held-out test evaluation")
    print("=" * 88)

    print()
    print("Locked model configuration:")

    print(
        json.dumps(
            {
                "model": model_name,
                "test_rows": len(x_test),
                "feature_count": len(selected_features),
                "threshold": (selected_threshold),
                "threshold_source": ("Locked Stage 5E2 validation"),
            },
            indent=2,
        )
    )

    print()
    print("Final test metrics:")

    print(
        json.dumps(
            display_metrics,
            indent=2,
        )
    )

    print()
    print("Validation versus test comparison:")

    print(comparison.to_string(index=False))

    print()
    print(
        "High-confidence test errors:",
        len(high_confidence_errors),
    )

    print()
    print("Result validation:")

    print(
        json.dumps(
            result_validation,
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
        "Overall Stage 5F verification passed:",
        overall_passed,
    )

    print()
    print("Final model:")
    print(final_model_path)

    print()
    print("Completion report:")
    print(completion_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
