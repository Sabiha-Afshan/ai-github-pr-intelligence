"""Train the corrected Stage 5D merge-outcome models."""

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
from src.models.merge_outcome_training import (
    AUTHOR_RANDOM_FOREST_MODEL_NAME,
    NEUTRAL_RANDOM_FOREST_MODEL_NAME,
    build_model_comparison,
    train_merge_outcome_models,
    validate_training_results,
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


def load_preprocessed_split(
    path: Path,
    expected_split: str,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """Load one preprocessed model split."""

    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessed split: {path}")

    dataframe = load_dataset(path)

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        TARGET_COLUMN,
        SPLIT_COLUMN,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing_columns)}")

    split_values = set(
        dataframe[SPLIT_COLUMN].astype(str).str.strip().str.lower().unique()
    )

    if split_values != {expected_split}:
        raise ValueError(
            f"{path.name} contains unexpected split values: {sorted(split_values)}"
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


def load_author_association() -> pd.DataFrame:
    """Load author association from the authoritative dataset."""

    detailed_path = (
        PROCESSED_DATA_DIRECTORY / "pallets_flask_time_matched_600_detailed.csv"
    )

    if not detailed_path.exists():
        raise FileNotFoundError(
            f"Author-association source is missing: {detailed_path}"
        )

    dataframe = load_dataset(detailed_path)

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        "author_association",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Detailed dataset is missing columns: {sorted(missing_columns)}"
        )

    author_data = dataframe[
        [
            PR_IDENTIFIER_COLUMN,
            "author_association",
        ]
    ].copy()

    duplicate_count = int(author_data.duplicated(subset=[PR_IDENTIFIER_COLUMN]).sum())

    if duplicate_count > 0:
        raise ValueError(
            f"Author-association data contains {duplicate_count} duplicate PR numbers."
        )

    author_data["author_association"] = (
        author_data["author_association"]
        .astype("string")
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
    )

    return author_data


def attach_author_association(
    identifiers: pd.Series,
    numeric_features: pd.DataFrame,
    author_data: pd.DataFrame,
) -> pd.DataFrame:
    """Attach author association to one model split."""

    feature_data = numeric_features.copy()

    feature_data.insert(
        0,
        PR_IDENTIFIER_COLUMN,
        identifiers.reset_index(drop=True),
    )

    merged = feature_data.merge(
        author_data,
        on=PR_IDENTIFIER_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if merged["author_association"].isna().any():
        missing_count = int(merged["author_association"].isna().sum())

        raise ValueError(
            f"Author association is missing for {missing_count} model rows."
        )

    return merged.drop(columns=[PR_IDENTIFIER_COLUMN])


def main() -> int:
    """Run corrected Stage 5D model training."""

    train_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_train.csv"

    validation_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_validation.csv"

    test_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_test.csv"

    stage_5c_path = REPORTS_DIRECTORY / "stage_5c_completion_report.json"

    if not stage_5c_path.exists():
        print("FAIL: Stage 5C completion report is missing.")
        return 1

    with stage_5c_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        stage_5c_report = json.load(report_file)

    if not stage_5c_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 5C did not pass verification.")
        return 1

    if not test_path.exists():
        print("FAIL: Untouched test split is missing.")
        return 1

    try:
        (
            train_identifiers,
            y_train,
            x_train,
        ) = load_preprocessed_split(
            train_path,
            expected_split="train",
        )

        (
            validation_identifiers,
            y_validation,
            x_validation,
        ) = load_preprocessed_split(
            validation_path,
            expected_split="validation",
        )

        author_data = load_author_association()

        author_x_train = attach_author_association(
            identifiers=(train_identifiers),
            numeric_features=x_train,
            author_data=author_data,
        )

        author_x_validation = attach_author_association(
            identifiers=(validation_identifiers),
            numeric_features=(x_validation),
            author_data=author_data,
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    if len(x_train) != 440:
        print("FAIL: Expected 440 training rows.")
        return 1

    if len(x_validation) != 84:
        print("FAIL: Expected 84 validation rows.")
        return 1

    if len(x_train.columns) != 61:
        print("FAIL: Expected 61 neutral features.")
        return 1

    results = train_merge_outcome_models(
        neutral_x_train=x_train,
        neutral_y_train=y_train,
        neutral_x_validation=(x_validation),
        neutral_y_validation=(y_validation),
        validation_identifiers=(validation_identifiers),
        author_x_train=(author_x_train),
        author_x_validation=(author_x_validation),
    )

    comparison = build_model_comparison(results)

    result_validation = validate_training_results(
        results=results,
        expected_validation_rows=(len(x_validation)),
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_paths = {}

    for model_name, result in results.items():
        model_path = MODEL_DIRECTORY / f"{model_name}.joblib"

        joblib.dump(
            result.model,
            model_path,
        )

        model_paths[model_name] = str(model_path)

    threshold_records = []

    prediction_tables = []

    for result in results.values():
        threshold_records.append(result.threshold_table)

        prediction_tables.append(result.predictions)

    thresholds = pd.concat(
        threshold_records,
        ignore_index=True,
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    comparison_path = REPORTS_DIRECTORY / "stage_5d_validation_model_comparison.csv"

    threshold_path = REPORTS_DIRECTORY / "stage_5d_validation_thresholds.csv"

    prediction_path = REPORTS_DIRECTORY / "stage_5d_validation_predictions.csv"

    completion_path = REPORTS_DIRECTORY / "stage_5d_completion_report.json"

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8",
    )

    thresholds.to_csv(
        threshold_path,
        index=False,
        encoding="utf-8",
    )

    predictions.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8",
    )

    final_model_name = NEUTRAL_RANDOM_FOREST_MODEL_NAME

    artifact_validation = {
        "saved_model_count": len(model_paths),
        "all_models_exist": all(
            Path(path).exists() and Path(path).stat().st_size > 0
            for path in model_paths.values()
        ),
        "comparison_exists": (comparison_path.exists()),
        "threshold_report_exists": (threshold_path.exists()),
        "prediction_report_exists": (prediction_path.exists()),
        "test_set_used": False,
    }

    artifact_validation["validation_passed"] = (
        artifact_validation["saved_model_count"] == 3
        and artifact_validation["all_models_exist"]
        and artifact_validation["comparison_exists"]
        and artifact_validation["threshold_report_exists"]
        and artifact_validation["prediction_report_exists"]
        and not artifact_validation["test_set_used"]
    )

    overall_passed = (
        result_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    neutral_forest_metrics = results[
        NEUTRAL_RANDOM_FOREST_MODEL_NAME
    ].validation_metrics

    author_forest_metrics = results[AUTHOR_RANDOM_FOREST_MODEL_NAME].validation_metrics

    fairness_comparison = {
        "neutral_random_forest_roc_auc": (neutral_forest_metrics["roc_auc"]),
        "author_benchmark_roc_auc": (author_forest_metrics["roc_auc"]),
        "roc_auc_difference": (
            author_forest_metrics["roc_auc"] - neutral_forest_metrics["roc_auc"]
        ),
        "neutral_random_forest_accuracy": (neutral_forest_metrics["accuracy"]),
        "author_benchmark_accuracy": (author_forest_metrics["accuracy"]),
        "accuracy_difference": (
            author_forest_metrics["accuracy"] - neutral_forest_metrics["accuracy"]
        ),
        "interpretation": (
            "The author-association model is used "
            "only to measure repository-specific "
            "shortcut effects. It is not eligible "
            "for final application use."
        ),
    }

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 5D",
        "target": TARGET_COLUMN,
        "target_definition": {
            "0": "Closed without merge",
            "1": "Merged",
        },
        "training_rows": len(x_train),
        "validation_rows": len(x_validation),
        "test_rows_reserved": 76,
        "test_set_used": False,
        "neutral_feature_count": len(x_train.columns),
        "trained_models": list(results),
        "final_application_candidate": (final_model_name),
        "final_model_selection_note": (
            "The contributor-neutral Random Forest "
            "is the designated application candidate. "
            "The author-association model is retained "
            "only for fairness and shortcut analysis."
        ),
        "selected_thresholds": {
            model_name: (result.selected_threshold)
            for model_name, result in results.items()
        },
        "model_paths": model_paths,
        "fairness_comparison": (fairness_comparison),
        "result_validation": (result_validation),
        "artifact_validation": (artifact_validation),
        "validation_metrics": (comparison.to_dict(orient="records")),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_path,
    )

    display_columns = [
        "validation_rank",
        "model_name",
        "model_role",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "threshold",
        "eligible_for_final_application",
    ]

    print("Stage 5D merge-outcome model training")
    print("=" * 110)

    print()
    print("Training design:")
    print(
        json.dumps(
            {
                "training_rows": len(x_train),
                "validation_rows": len(x_validation),
                "test_rows_reserved": 76,
                "test_set_used": False,
                "neutral_feature_count": len(x_train.columns),
            },
            indent=2,
        )
    )

    print()
    print("Validation model comparison:")

    print(comparison[display_columns].to_string(index=False))

    print()
    print("Fairness and shortcut comparison:")

    print(
        json.dumps(
            fairness_comparison,
            indent=2,
        )
    )

    print()
    print("Training-result validation:")

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
        "Final application candidate:",
        final_model_name,
    )

    print()
    print(
        "Overall Stage 5D verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(completion_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
