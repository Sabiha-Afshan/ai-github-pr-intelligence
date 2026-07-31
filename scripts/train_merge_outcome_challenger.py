"""Train and compare the Histogram Gradient Boosting challenger."""

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
from src.models.merge_outcome_challenger import (
    CHALLENGER_MODEL_NAME,
    compare_challenger_with_random_forest,
    create_challenger_predictions,
    fit_selected_challenger,
    get_selected_cv_candidate,
    run_challenger_cross_validation,
    validate_challenger_outputs,
)
from src.models.merge_outcome_tuning import (
    FINAL_MODEL_NAME,
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
        raise ValueError(
            f"{file_path.name} is missing columns: {sorted(missing_columns)}"
        )

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
    """Run Stage 5E2 challenger training."""

    train_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_train.csv"

    validation_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_validation.csv"

    test_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_test.csv"

    stage_5e_path = REPORTS_DIRECTORY / "stage_5e_completion_report.json"

    random_forest_model_path = MODEL_DIRECTORY / f"{FINAL_MODEL_NAME}.joblib"

    required_paths = [
        train_path,
        validation_path,
        test_path,
        stage_5e_path,
        random_forest_model_path,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required challenger files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    with stage_5e_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        stage_5e_report = json.load(report_file)

    if not stage_5e_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 5E did not pass verification.")
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
        print("FAIL: Expected 440 training rows.")
        return 1

    if len(validation_identifiers) != 84:
        print("FAIL: Expected 84 validation rows.")
        return 1

    if list(x_train.columns) != list(x_validation.columns):
        print("FAIL: Train and validation features differ.")
        return 1

    cv_results, cv_comparison = run_challenger_cross_validation(
        x_train=x_train,
        y_train=y_train,
    )

    selected_cv_candidate = get_selected_cv_candidate(
        results=cv_results,
        comparison=cv_comparison,
    )

    challenger_result = fit_selected_challenger(
        selected_candidate=(selected_cv_candidate),
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    challenger_predictions = create_challenger_predictions(
        result=challenger_result,
        validation_identifiers=(validation_identifiers),
        y_validation=y_validation,
    )

    random_forest_metrics = stage_5e_report["selected_validation_metrics"]

    model_selection = compare_challenger_with_random_forest(
        challenger_metrics=(challenger_result.validation_metrics),
        random_forest_metrics=(random_forest_metrics),
        minimum_roc_auc_improvement=0.02,
    )

    output_validation = validate_challenger_outputs(
        cv_results=cv_results,
        cv_comparison=cv_comparison,
        challenger_result=(challenger_result),
        expected_validation_rows=84,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    challenger_model_path = MODEL_DIRECTORY / f"{CHALLENGER_MODEL_NAME}.joblib"

    joblib.dump(
        challenger_result.model,
        challenger_model_path,
    )

    challenger_features_path = REPORTS_DIRECTORY / "stage_5e2_selected_features.csv"

    cv_comparison_path = REPORTS_DIRECTORY / "stage_5e2_temporal_cv_comparison.csv"

    threshold_path = REPORTS_DIRECTORY / "stage_5e2_threshold_analysis.csv"

    predictions_path = REPORTS_DIRECTORY / "stage_5e2_validation_predictions.csv"

    model_comparison_path = REPORTS_DIRECTORY / "stage_5e2_model_selection.json"

    completion_path = REPORTS_DIRECTORY / "stage_5e2_completion_report.json"

    locked_model_path = MODEL_DIRECTORY / "locked_merge_outcome_model.joblib"

    locked_configuration_path = (
        REPORTS_DIRECTORY / "stage_5e2_locked_model_configuration.json"
    )

    selected_features_dataframe = pd.DataFrame(
        {
            "feature_position": range(
                1,
                len(challenger_result.feature_names) + 1,
            ),
            "feature": (challenger_result.feature_names),
        }
    )

    selected_features_dataframe.to_csv(
        challenger_features_path,
        index=False,
        encoding="utf-8",
    )

    cv_comparison.to_csv(
        cv_comparison_path,
        index=False,
        encoding="utf-8",
    )

    challenger_result.threshold_table.to_csv(
        threshold_path,
        index=False,
        encoding="utf-8",
    )

    challenger_predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
    )

    save_json(
        model_selection,
        model_comparison_path,
    )

    if model_selection["challenger_selected"]:
        shutil.copyfile(
            challenger_model_path,
            locked_model_path,
        )

        locked_features = challenger_result.feature_names

        locked_threshold = challenger_result.selected_threshold

        locked_model_name = CHALLENGER_MODEL_NAME

        locked_source_model_path = challenger_model_path

        locked_validation_metrics = challenger_result.validation_metrics

    else:
        shutil.copyfile(
            random_forest_model_path,
            locked_model_path,
        )

        locked_features = stage_5e_report["selected_features"]

        locked_threshold = float(stage_5e_report["selected_threshold"])

        locked_model_name = FINAL_MODEL_NAME

        locked_source_model_path = random_forest_model_path

        locked_validation_metrics = random_forest_metrics

    locked_configuration = {
        "locked_at": datetime.now(UTC),
        "model_name": locked_model_name,
        "source_model_path": str(locked_source_model_path),
        "locked_model_path": str(locked_model_path),
        "feature_count": len(locked_features),
        "features": locked_features,
        "threshold": (locked_threshold),
        "threshold_source": ("Validation data before final test evaluation"),
        "validation_metrics": (locked_validation_metrics),
        "test_set_used": False,
        "selection_rule": {
            "minimum_roc_auc_improvement": 0.02,
            "secondary_rule": (
                "Balanced accuracy improvement of at least "
                "0.02, no F1 reduction and no ROC-AUC reduction."
            ),
        },
    }

    save_json(
        locked_configuration,
        locked_configuration_path,
    )

    artifact_validation = {
        "challenger_model_exists": (challenger_model_path.exists()),
        "challenger_model_size_bytes": (
            challenger_model_path.stat().st_size
            if challenger_model_path.exists()
            else 0
        ),
        "cv_comparison_exists": (cv_comparison_path.exists()),
        "threshold_report_exists": (threshold_path.exists()),
        "predictions_exist": (predictions_path.exists()),
        "model_selection_exists": (model_comparison_path.exists()),
        "locked_model_exists": (locked_model_path.exists()),
        "locked_configuration_exists": (locked_configuration_path.exists()),
        "test_set_used": False,
    }

    artifact_validation["validation_passed"] = (
        artifact_validation["challenger_model_exists"]
        and artifact_validation["challenger_model_size_bytes"] > 0
        and artifact_validation["cv_comparison_exists"]
        and artifact_validation["threshold_report_exists"]
        and artifact_validation["predictions_exist"]
        and artifact_validation["model_selection_exists"]
        and artifact_validation["locked_model_exists"]
        and artifact_validation["locked_configuration_exists"]
        and not artifact_validation["test_set_used"]
    )

    overall_passed = (
        output_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 5E2",
        "target": TARGET_COLUMN,
        "training_rows": len(x_train),
        "validation_rows": len(x_validation),
        "test_rows_reserved": 76,
        "test_set_used": False,
        "cross_validation_method": (
            "Five-fold expanding-window temporal cross-validation on training rows only"
        ),
        "challenger_candidate_count": len(cv_results),
        "selected_cv_candidate": (selected_cv_candidate.candidate_id),
        "selected_feature_set": (challenger_result.feature_set_name),
        "selected_feature_count": len(challenger_result.feature_names),
        "selected_parameters": (challenger_result.parameters),
        "temporal_cv_mean_roc_auc": (selected_cv_candidate.mean_cv_roc_auc),
        "temporal_cv_std_roc_auc": (selected_cv_candidate.std_cv_roc_auc),
        "challenger_validation_metrics": (challenger_result.validation_metrics),
        "challenger_training_metrics": (challenger_result.training_metrics),
        "model_selection": (model_selection),
        "locked_model": (locked_configuration),
        "output_validation": (output_validation),
        "artifact_validation": (artifact_validation),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_path,
    )

    display_columns = [
        "cv_rank",
        "candidate_id",
        "feature_set_name",
        "configuration_name",
        "feature_count",
        "mean_cv_roc_auc",
        "std_cv_roc_auc",
        "minimum_cv_roc_auc",
        "valid_fold_count",
        "selected_candidate",
    ]

    print("Stage 5E2 Histogram Gradient Boosting challenger")
    print("=" * 118)

    print()
    print("Top temporal cross-validation candidates:")

    print(cv_comparison[display_columns].head(12).to_string(index=False))

    print()
    print("Selected challenger configuration:")

    print(
        json.dumps(
            {
                "candidate_id": (selected_cv_candidate.candidate_id),
                "feature_set": (challenger_result.feature_set_name),
                "feature_count": len(challenger_result.feature_names),
                "parameters": (challenger_result.parameters),
                "mean_temporal_cv_roc_auc": (selected_cv_candidate.mean_cv_roc_auc),
                "std_temporal_cv_roc_auc": (selected_cv_candidate.std_cv_roc_auc),
                "validation_threshold": (challenger_result.selected_threshold),
                "validation_metrics": (challenger_result.validation_metrics),
            },
            indent=2,
            default=str,
        )
    )

    print()
    print("Random Forest versus challenger:")

    print(
        json.dumps(
            model_selection,
            indent=2,
        )
    )

    print()
    print("Locked final model before test evaluation:")

    print(
        json.dumps(
            {
                "model_name": (locked_model_name),
                "feature_count": len(locked_features),
                "threshold": (locked_threshold),
                "test_set_used": False,
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
        "Overall Stage 5E2 verification passed:",
        overall_passed,
    )

    print()
    print("Locked configuration:")

    print(locked_configuration_path)

    print()
    print("Completion report:")

    print(completion_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
