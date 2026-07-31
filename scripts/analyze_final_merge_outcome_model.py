"""Run final explainability and error analysis for Model 1."""

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
from src.models.merge_outcome_explainability import (
    build_confidence_band_table,
    build_error_analysis_table,
    build_error_feature_profile,
    build_error_type_summary,
    build_probability_calibration_table,
    calculate_explainability_summary,
    calculate_random_forest_importance,
    calculate_test_permutation_importance,
    combine_importance_tables,
    validate_explainability_outputs,
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


def load_test_features(
    file_path: Path,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """Load the final held-out test features."""

    if not file_path.exists():
        raise FileNotFoundError(f"Test feature file is missing: {file_path}")

    dataframe = load_dataset(file_path)

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        TARGET_COLUMN,
        SPLIT_COLUMN,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Test feature file is missing columns: {sorted(missing_columns)}"
        )

    split_values = set(
        dataframe[SPLIT_COLUMN].astype(str).str.strip().str.lower().unique()
    )

    if split_values != {"test"}:
        raise ValueError(f"Expected only test rows, but found: {sorted(split_values)}")

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
    """Run Stage 5G explainability and error analysis."""

    stage_5f_path = REPORTS_DIRECTORY / "stage_5f_completion_report.json"

    locked_configuration_path = (
        REPORTS_DIRECTORY / "stage_5e2_locked_model_configuration.json"
    )

    final_model_path = (
        MODEL_DIRECTORY / "final_contributor_neutral_merge_outcome_model.joblib"
    )

    test_features_path = PROCESSED_DATA_DIRECTORY / "model1_preprocessed_test.csv"

    test_predictions_path = REPORTS_DIRECTORY / "stage_5f_test_predictions.csv"

    required_paths = [
        stage_5f_path,
        locked_configuration_path,
        final_model_path,
        test_features_path,
        test_predictions_path,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 5G files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    with stage_5f_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        stage_5f_report = json.load(report_file)

    if not stage_5f_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 5F did not pass verification.")
        return 1

    with locked_configuration_path.open(
        "r",
        encoding="utf-8",
    ) as configuration_file:
        locked_configuration = json.load(configuration_file)

    selected_features = [str(feature) for feature in locked_configuration["features"]]

    try:
        (
            test_identifiers,
            y_test,
            x_test,
        ) = load_test_features(test_features_path)

        predictions = load_dataset(test_predictions_path)

        model = joblib.load(final_model_path)

        impurity_importance = calculate_random_forest_importance(
            model=model,
            feature_names=selected_features,
        )

        permutation_importance_table = calculate_test_permutation_importance(
            model=model,
            features=x_test,
            targets=y_test,
            feature_names=selected_features,
            repeats=50,
            random_state=42,
        )

        combined_importance = combine_importance_tables(
            impurity_importance=(impurity_importance),
            permutation_importance_table=(permutation_importance_table),
        )

        calibration_table = build_probability_calibration_table(
            predictions=predictions,
            bin_count=10,
        )

        confidence_table = build_confidence_band_table(predictions=predictions)

        error_analysis = build_error_analysis_table(
            predictions=predictions,
            test_features=x_test,
            identifiers=test_identifiers,
        )

        error_type_summary = build_error_type_summary(error_analysis)

        error_feature_profile = build_error_feature_profile(
            error_analysis=error_analysis,
            selected_features=selected_features,
        )

        explainability_summary = calculate_explainability_summary(
            predictions=predictions,
            combined_importance=(combined_importance),
        )

        result_validation = validate_explainability_outputs(
            combined_importance=(combined_importance),
            calibration_table=(calibration_table),
            confidence_table=(confidence_table),
            error_analysis=error_analysis,
            expected_feature_count=len(selected_features),
            expected_row_count=76,
        )

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    impurity_path = REPORTS_DIRECTORY / "stage_5g_impurity_feature_importance.csv"

    permutation_path = REPORTS_DIRECTORY / "stage_5g_permutation_feature_importance.csv"

    combined_path = REPORTS_DIRECTORY / "stage_5g_combined_feature_importance.csv"

    calibration_path = REPORTS_DIRECTORY / "stage_5g_probability_calibration.csv"

    confidence_path = REPORTS_DIRECTORY / "stage_5g_confidence_band_analysis.csv"

    error_analysis_path = REPORTS_DIRECTORY / "stage_5g_enriched_error_analysis.csv"

    error_summary_path = REPORTS_DIRECTORY / "stage_5g_error_type_summary.csv"

    error_profile_path = REPORTS_DIRECTORY / "stage_5g_error_feature_profile.csv"

    completion_path = REPORTS_DIRECTORY / "stage_5g_completion_report.json"

    impurity_importance.to_csv(
        impurity_path,
        index=False,
        encoding="utf-8",
    )

    permutation_importance_table.to_csv(
        permutation_path,
        index=False,
        encoding="utf-8",
    )

    combined_importance.to_csv(
        combined_path,
        index=False,
        encoding="utf-8",
    )

    calibration_table.to_csv(
        calibration_path,
        index=False,
        encoding="utf-8",
    )

    confidence_table.to_csv(
        confidence_path,
        index=False,
        encoding="utf-8",
    )

    error_analysis.to_csv(
        error_analysis_path,
        index=False,
        encoding="utf-8",
    )

    error_type_summary.to_csv(
        error_summary_path,
        index=False,
        encoding="utf-8",
    )

    error_feature_profile.to_csv(
        error_profile_path,
        index=False,
        encoding="utf-8",
    )

    artifact_validation = {
        "impurity_importance_exists": (impurity_path.exists()),
        "permutation_importance_exists": (permutation_path.exists()),
        "combined_importance_exists": (combined_path.exists()),
        "calibration_report_exists": (calibration_path.exists()),
        "confidence_report_exists": (confidence_path.exists()),
        "error_analysis_exists": (error_analysis_path.exists()),
        "error_summary_exists": (error_summary_path.exists()),
        "error_profile_exists": (error_profile_path.exists()),
        "model_retrained": False,
        "model_reselected": False,
        "threshold_changed": False,
    }

    artifact_validation["validation_passed"] = (
        all(
            value
            for key, value in artifact_validation.items()
            if key.endswith("_exists")
        )
        and not artifact_validation["model_retrained"]
        and not artifact_validation["model_reselected"]
        and not artifact_validation["threshold_changed"]
    )

    overall_passed = (
        result_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 5G",
        "model_name": locked_configuration["model_name"],
        "analysis_type": ("Post-hoc explainability and final held-out error analysis"),
        "model_retrained": False,
        "model_reselected": False,
        "threshold_changed": False,
        "test_rows": len(predictions),
        "feature_count": len(selected_features),
        "explainability_summary": (explainability_summary),
        "result_validation": (result_validation),
        "artifact_validation": (artifact_validation),
        "interpretation_warning": (
            "Feature importance identifies predictive "
            "association, not causal impact. Correlated "
            "features may share or distort importance."
        ),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_path,
    )

    print("Stage 5G final model explainability and error analysis")
    print("=" * 100)

    print()
    print("Final model summary:")

    print(
        json.dumps(
            explainability_summary,
            indent=2,
        )
    )

    print()
    print("Top combined features:")

    print(
        combined_importance[
            [
                "combined_rank",
                "feature",
                "impurity_importance",
                "permutation_importance_mean",
                "permutation_importance_std",
                "average_rank",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print()
    print("Prediction outcome summary:")

    print(error_type_summary.to_string(index=False))

    print()
    print("Confidence-band analysis:")

    print(confidence_table.to_string(index=False))

    print()
    print("Probability calibration:")

    print(calibration_table.to_string(index=False))

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
        "Overall Stage 5G verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(completion_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
