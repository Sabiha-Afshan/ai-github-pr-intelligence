"""Prepare train-only preprocessed features for Model 2."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.discovery import load_dataset
from src.models.merge_delay_preprocessing import (
    TARGET_COLUMN,
    build_output_split,
    prepare_model2_data,
    validate_prepared_data,
    validate_source_dataset,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    PROJECT_ROOT,
    REPORTS_DIRECTORY,
)

SOURCE_DATASET_PATH = (
    PROCESSED_DATA_DIRECTORY / "model2_merge_delay_approved_features.csv"
)

MODEL_DIRECTORY = PROJECT_ROOT / "models" / "merge_delay"

PREPROCESSOR_PATH = MODEL_DIRECTORY / "model2_preprocessor.joblib"

TRAIN_OUTPUT_PATH = PROCESSED_DATA_DIRECTORY / "model2_preprocessed_train.csv"

VALIDATION_OUTPUT_PATH = PROCESSED_DATA_DIRECTORY / "model2_preprocessed_validation.csv"

TEST_OUTPUT_PATH = PROCESSED_DATA_DIRECTORY / "model2_preprocessed_test.csv"

FEATURE_MANIFEST_PATH = REPORTS_DIRECTORY / "stage_6c_feature_manifest.csv"

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_6c_completion_report.json"


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


def build_feature_manifest(
    feature_names: list[str],
    binary_features: list[str],
    continuous_features: list[str],
) -> pd.DataFrame:
    """Create the Model 2 feature manifest."""

    binary_set = set(binary_features)

    continuous_set = set(continuous_features)

    records = []

    for position, feature in enumerate(
        feature_names,
        start=1,
    ):
        if feature in binary_set:
            feature_type = "binary"
            preprocessing = "Most-frequent imputation"
        elif feature in continuous_set:
            feature_type = "continuous"
            preprocessing = "Median imputation and training-fitted standardization"
        else:
            raise ValueError(f"Feature is missing from type classification: {feature}")

        records.append(
            {
                "feature_position": (position),
                "feature": feature,
                "feature_type": (feature_type),
                "preprocessing": (preprocessing),
                "fit_source": ("Training split only"),
            }
        )

    return pd.DataFrame(records)


def main() -> int:
    """Run Stage 6C preprocessing."""

    stage_6b_path = REPORTS_DIRECTORY / "stage_6b_completion_report.json"

    required_paths = [
        SOURCE_DATASET_PATH,
        stage_6b_path,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 6C files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    with stage_6b_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        stage_6b_report = json.load(report_file)

    if not stage_6b_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 6B did not pass verification.")
        return 1

    try:
        source_dataframe = load_dataset(SOURCE_DATASET_PATH)

        source_validation = validate_source_dataset(source_dataframe)

        if not source_validation["validation_passed"]:
            raise ValueError(
                f"Approved Model 2 dataset failed validation: {source_validation}"
            )

        prepared = prepare_model2_data(source_dataframe)

        transformed_validation = validate_prepared_data(prepared)

        train_output = build_output_split(
            identifiers=(prepared.train_identifiers),
            targets=prepared.y_train,
            features=prepared.x_train,
            split_name="train",
        )

        validation_output = build_output_split(
            identifiers=(prepared.validation_identifiers),
            targets=(prepared.y_validation),
            features=(prepared.x_validation),
            split_name="validation",
        )

        test_output = build_output_split(
            identifiers=(prepared.test_identifiers),
            targets=prepared.y_test,
            features=prepared.x_test,
            split_name="test",
        )

        feature_manifest = build_feature_manifest(
            feature_names=(prepared.feature_names),
            binary_features=(prepared.binary_features),
            continuous_features=(prepared.continuous_features),
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

    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        prepared.preprocessor,
        PREPROCESSOR_PATH,
    )

    train_output.to_csv(
        TRAIN_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    validation_output.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    test_output.to_csv(
        TEST_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    feature_manifest.to_csv(
        FEATURE_MANIFEST_PATH,
        index=False,
        encoding="utf-8",
    )

    artifact_validation = {
        "preprocessor_exists": (PREPROCESSOR_PATH.exists()),
        "preprocessor_size_bytes": (
            PREPROCESSOR_PATH.stat().st_size if PREPROCESSOR_PATH.exists() else 0
        ),
        "train_output_exists": (TRAIN_OUTPUT_PATH.exists()),
        "validation_output_exists": (VALIDATION_OUTPUT_PATH.exists()),
        "test_output_exists": (TEST_OUTPUT_PATH.exists()),
        "feature_manifest_exists": (FEATURE_MANIFEST_PATH.exists()),
    }

    artifact_validation["validation_passed"] = (
        artifact_validation["preprocessor_exists"]
        and artifact_validation["preprocessor_size_bytes"] > 0
        and artifact_validation["train_output_exists"]
        and artifact_validation["validation_output_exists"]
        and artifact_validation["test_output_exists"]
        and artifact_validation["feature_manifest_exists"]
    )

    overall_passed = (
        source_validation["validation_passed"]
        and transformed_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 6C",
        "model": ("Model 2 — Merge-delay prediction"),
        "source_dataset": str(SOURCE_DATASET_PATH),
        "target": TARGET_COLUMN,
        "training_rows": len(train_output),
        "validation_rows": len(validation_output),
        "test_rows": len(test_output),
        "feature_count": len(prepared.feature_names),
        "binary_feature_count": len(prepared.binary_features),
        "continuous_feature_count": len(prepared.continuous_features),
        "binary_features": (prepared.binary_features),
        "continuous_features": (prepared.continuous_features),
        "source_validation": (source_validation),
        "transformed_validation": (transformed_validation),
        "artifact_validation": (artifact_validation),
        "preprocessing_policy": {
            "fit_split": ("Training only"),
            "binary_imputation": ("Most frequent"),
            "continuous_imputation": ("Median"),
            "continuous_scaling": ("StandardScaler fitted on training only"),
            "validation_used_for_fit": (False),
            "test_used_for_fit": (False),
        },
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    print("Stage 6C Model 2 train-only preprocessing")
    print("=" * 92)

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
                "binary_feature_count": len(prepared.binary_features),
                "continuous_feature_count": len(prepared.continuous_features),
                "total_feature_count": len(prepared.feature_names),
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
        "Overall Stage 6C verification passed:",
        overall_passed,
    )

    print()
    print("Preprocessor:")
    print(PREPROCESSOR_PATH)

    print()
    print("Completion report:")
    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
