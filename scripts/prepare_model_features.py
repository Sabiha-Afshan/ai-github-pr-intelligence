"""Fit and save the Stage 5C preprocessing pipeline."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from src.data.discovery import load_dataset
from src.models.preprocessing import (
    build_transformed_output,
    create_chronological_splits,
    fit_and_transform_model_splits,
    get_model_feature_columns,
    validate_approved_modelling_dataset,
    validate_transformed_splits,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    PROJECT_ROOT,
    REPORTS_DIRECTORY,
)

MODEL_ARTIFACT_DIRECTORY = PROJECT_ROOT / "models"


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


def main() -> int:
    """Run Stage 5C preprocessing."""

    approved_dataset_path = (
        PROCESSED_DATA_DIRECTORY / "model1_merge_outcome_approved_features.csv"
    )

    stage_5b_report_path = REPORTS_DIRECTORY / "stage_5b_completion_report.json"

    if not approved_dataset_path.exists():
        print("FAIL: Approved feature dataset is missing:")
        print(approved_dataset_path)
        return 1

    if not stage_5b_report_path.exists():
        print("FAIL: Stage 5B completion report is missing:")
        print(stage_5b_report_path)
        return 1

    with stage_5b_report_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        stage_5b_report = json.load(report_file)

    if not stage_5b_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 5B did not pass verification.")
        return 1

    dataframe = load_dataset(approved_dataset_path)

    source_validation = validate_approved_modelling_dataset(dataframe)

    if not source_validation["validation_passed"]:
        print("FAIL: Approved modelling dataset did not pass validation.")

        print(
            json.dumps(
                source_validation,
                indent=2,
            )
        )

        return 1

    feature_columns = get_model_feature_columns(dataframe)

    splits = create_chronological_splits(dataframe)

    (
        preprocessor,
        feature_groups,
        transformed,
    ) = fit_and_transform_model_splits(
        splits=splits,
        feature_columns=feature_columns,
    )

    transformed_validation = validate_transformed_splits(
        transformed=transformed,
        expected_feature_count=len(feature_columns),
    )

    train_output = build_transformed_output(
        identifiers=(transformed.train_identifiers),
        targets=transformed.y_train,
        features=transformed.x_train,
        split_name="train",
    )

    validation_output = build_transformed_output(
        identifiers=(transformed.validation_identifiers),
        targets=(transformed.y_validation),
        features=(transformed.x_validation),
        split_name="validation",
    )

    test_output = build_transformed_output(
        identifiers=(transformed.test_identifiers),
        targets=transformed.y_test,
        features=transformed.x_test,
        split_name="test",
    )

    train_output_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_train.csv"

    validation_output_path = (
        PROCESSED_DATA_DIRECTORY / "model1_preprocessed_validation.csv"
    )

    test_output_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_test.csv"

    feature_manifest_path = REPORTS_DIRECTORY / "stage_5c_feature_manifest.csv"

    completion_report_path = REPORTS_DIRECTORY / "stage_5c_completion_report.json"

    preprocessing_artifact_path = (
        MODEL_ARTIFACT_DIRECTORY / "model1_preprocessor.joblib"
    )

    MODEL_ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_output.to_csv(
        train_output_path,
        index=False,
        encoding="utf-8",
    )

    validation_output.to_csv(
        validation_output_path,
        index=False,
        encoding="utf-8",
    )

    test_output.to_csv(
        test_output_path,
        index=False,
        encoding="utf-8",
    )

    joblib.dump(
        preprocessor,
        preprocessing_artifact_path,
    )

    feature_manifest_records = []

    binary_feature_set = set(feature_groups.binary_features)

    for position, feature_name in enumerate(
        transformed.x_train.columns,
        start=1,
    ):
        feature_manifest_records.append(
            {
                "position": position,
                "feature": feature_name,
                "feature_type": (
                    "binary" if feature_name in binary_feature_set else "continuous"
                ),
                "imputation_strategy": (
                    "most_frequent" if feature_name in binary_feature_set else "median"
                ),
                "scaled": (feature_name not in binary_feature_set),
            }
        )

    feature_manifest = pd.DataFrame(feature_manifest_records)

    feature_manifest.to_csv(
        feature_manifest_path,
        index=False,
        encoding="utf-8",
    )

    artifact_validation = {
        "preprocessor_exists": (preprocessing_artifact_path.exists()),
        "preprocessor_size_bytes": (
            preprocessing_artifact_path.stat().st_size
            if preprocessing_artifact_path.exists()
            else 0
        ),
        "train_output_exists": (train_output_path.exists()),
        "validation_output_exists": (validation_output_path.exists()),
        "test_output_exists": (test_output_path.exists()),
        "feature_manifest_exists": (feature_manifest_path.exists()),
    }

    artifact_validation["validation_passed"] = (
        all(
            value
            for key, value in artifact_validation.items()
            if key.endswith("_exists")
        )
        and artifact_validation["preprocessor_size_bytes"] > 0
    )

    overall_passed = (
        source_validation["validation_passed"]
        and transformed_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": (datetime.now(UTC)),
        "stage": "Stage 5C",
        "scikit_learn_version": (sklearn.__version__),
        "source_dataset": str(approved_dataset_path),
        "preprocessor_artifact": str(preprocessing_artifact_path),
        "source_validation": (source_validation),
        "binary_feature_count": len(feature_groups.binary_features),
        "continuous_feature_count": len(feature_groups.continuous_features),
        "binary_features": (feature_groups.binary_features),
        "continuous_features": (feature_groups.continuous_features),
        "transformed_validation": (transformed_validation),
        "artifact_validation": (artifact_validation),
        "generated_files": {
            "train": str(train_output_path),
            "validation": str(validation_output_path),
            "test": str(test_output_path),
            "feature_manifest": str(feature_manifest_path),
        },
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_report_path,
    )

    print("Stage 5C train-only preprocessing")
    print("=" * 76)

    print()
    print("Source dataset validation:")
    print(
        json.dumps(
            source_validation,
            indent=2,
        )
    )

    print()
    print("Feature classification:")
    print(
        json.dumps(
            {
                "binary_feature_count": len(feature_groups.binary_features),
                "continuous_feature_count": len(feature_groups.continuous_features),
            },
            indent=2,
        )
    )

    print()
    print("Transformed split validation:")
    print(
        json.dumps(
            transformed_validation,
            indent=2,
        )
    )

    print()
    print("Saved-artifact validation:")
    print(
        json.dumps(
            artifact_validation,
            indent=2,
        )
    )

    print()
    print(
        "Overall Stage 5C verification passed:",
        overall_passed,
    )

    print()
    print("Preprocessor:")
    print(preprocessing_artifact_path)

    print()
    print("Completion report:")
    print(completion_report_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
