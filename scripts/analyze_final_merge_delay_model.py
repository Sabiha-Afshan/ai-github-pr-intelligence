"""Run explainability and error analysis for final Model 2."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.data.discovery import load_dataset
from src.models.merge_delay_explainability import (
    build_error_analysis,
    build_probability_band_summary,
    calculate_local_logit_contributions,
    calculate_logistic_coefficient_importance,
    calculate_test_permutation_importance,
    combine_explainability_tables,
    validate_explainability_outputs,
)
from src.models.merge_delay_final_evaluation import (
    load_final_test_split,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    PROJECT_ROOT,
    REPORTS_DIRECTORY,
)

MODEL_DIRECTORY = PROJECT_ROOT / "models" / "merge_delay"

FINAL_MODEL_PATH = MODEL_DIRECTORY / "final_merge_delay_model.joblib"

TEST_DATA_PATH = PROCESSED_DATA_DIRECTORY / "model2_preprocessed_test.csv"

TEST_PREDICTIONS_PATH = REPORTS_DIRECTORY / "stage_6f_test_predictions.csv"

LOCKED_CONFIGURATION_PATH = (
    REPORTS_DIRECTORY / "stage_6e_locked_model_configuration.json"
)

STAGE_6F_COMPLETION_PATH = REPORTS_DIRECTORY / "stage_6f_completion_report.json"

COEFFICIENT_REPORT_PATH = REPORTS_DIRECTORY / "stage_6g_logistic_coefficients.csv"

PERMUTATION_REPORT_PATH = REPORTS_DIRECTORY / "stage_6g_permutation_importance.csv"

COMBINED_REPORT_PATH = REPORTS_DIRECTORY / "stage_6g_combined_explainability.csv"

LOCAL_CONTRIBUTION_PATH = REPORTS_DIRECTORY / "stage_6g_local_contributions.csv"

ERROR_REPORT_PATH = REPORTS_DIRECTORY / "stage_6g_error_analysis.csv"

ERROR_FEATURE_REPORT_PATH = REPORTS_DIRECTORY / "stage_6g_error_feature_summary.csv"

PROBABILITY_BAND_PATH = REPORTS_DIRECTORY / "stage_6g_probability_band_summary.csv"

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_6g_completion_report.json"


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
    """Run Stage 6G explainability."""

    required_paths = [
        FINAL_MODEL_PATH,
        TEST_DATA_PATH,
        TEST_PREDICTIONS_PATH,
        LOCKED_CONFIGURATION_PATH,
        STAGE_6F_COMPLETION_PATH,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 6G files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    with STAGE_6F_COMPLETION_PATH.open(
        "r",
        encoding="utf-8",
    ) as completion_file:
        stage_6f_report = json.load(completion_file)

    if not stage_6f_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 6F did not pass verification.")
        return 1

    with LOCKED_CONFIGURATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as configuration_file:
        configuration = json.load(configuration_file)

    feature_names = [str(feature) for feature in configuration["features"]]

    try:
        test_dataframe = load_dataset(TEST_DATA_PATH)

        predictions = load_dataset(TEST_PREDICTIONS_PATH)

        (
            identifiers,
            targets,
            features,
        ) = load_final_test_split(
            dataframe=test_dataframe,
            feature_names=feature_names,
        )

        model = joblib.load(FINAL_MODEL_PATH)

        coefficient_table = calculate_logistic_coefficient_importance(
            model=model,
            feature_names=feature_names,
        )

        permutation_table = calculate_test_permutation_importance(
            model=model,
            features=features,
            targets=targets,
            feature_names=feature_names,
            repeats=50,
            random_state=42,
        )

        combined_table = combine_explainability_tables(
            coefficient_table=(coefficient_table),
            permutation_table=(permutation_table),
        )

        local_contributions = calculate_local_logit_contributions(
            model=model,
            features=features,
            identifiers=identifiers,
            actual_targets=targets,
            predicted_targets=(predictions["predicted_target"].astype(int)),
            probabilities=(predictions["delay_probability"].astype(float)),
            top_feature_count=10,
        )

        (
            error_table,
            error_feature_summary,
        ) = build_error_analysis(
            predictions=predictions,
            local_contributions=(local_contributions),
        )

        probability_bands = build_probability_band_summary(predictions)

        output_validation = validate_explainability_outputs(
            coefficient_table=(coefficient_table),
            permutation_table=(permutation_table),
            combined_table=(combined_table),
            local_contributions=(local_contributions),
            errors=error_table,
            probability_bands=(probability_bands),
            expected_feature_count=58,
            expected_test_rows=38,
            top_feature_count=10,
        )

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    coefficient_table.to_csv(
        COEFFICIENT_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    permutation_table.to_csv(
        PERMUTATION_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    combined_table.to_csv(
        COMBINED_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    local_contributions.to_csv(
        LOCAL_CONTRIBUTION_PATH,
        index=False,
        encoding="utf-8",
    )

    error_table.to_csv(
        ERROR_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    error_feature_summary.to_csv(
        ERROR_FEATURE_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    probability_bands.to_csv(
        PROBABILITY_BAND_PATH,
        index=False,
        encoding="utf-8",
    )

    artifact_validation = {
        "coefficient_report_exists": (COEFFICIENT_REPORT_PATH.exists()),
        "permutation_report_exists": (PERMUTATION_REPORT_PATH.exists()),
        "combined_report_exists": (COMBINED_REPORT_PATH.exists()),
        "local_contribution_report_exists": (LOCAL_CONTRIBUTION_PATH.exists()),
        "error_report_exists": (ERROR_REPORT_PATH.exists()),
        "error_feature_report_exists": (ERROR_FEATURE_REPORT_PATH.exists()),
        "probability_band_report_exists": (PROBABILITY_BAND_PATH.exists()),
        "model_retrained": False,
        "model_reselected": False,
        "threshold_changed": False,
        "features_changed": False,
    }

    artifact_validation["validation_passed"] = bool(
        all(
            value
            for key, value in artifact_validation.items()
            if key.endswith("_exists")
        )
        and not artifact_validation["model_retrained"]
        and not artifact_validation["model_reselected"]
        and not artifact_validation["threshold_changed"]
        and not artifact_validation["features_changed"]
    )

    strongest_positive_features = (
        coefficient_table.loc[coefficient_table["coefficient"] > 0]
        .sort_values(
            "coefficient",
            ascending=False,
        )
        .head(10)[
            [
                "feature",
                "coefficient",
                "odds_ratio",
            ]
        ]
        .to_dict(orient="records")
    )

    strongest_negative_features = (
        coefficient_table.loc[coefficient_table["coefficient"] < 0]
        .sort_values(
            "coefficient",
            ascending=True,
        )
        .head(10)[
            [
                "feature",
                "coefficient",
                "odds_ratio",
            ]
        ]
        .to_dict(orient="records")
    )

    top_combined_features = combined_table.head(10)["feature"].tolist()

    overall_passed = bool(
        output_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 6G",
        "model": ("Model 2 — Merge-delay prediction"),
        "model_name": configuration["model_name"],
        "model_family": configuration["model_family"],
        "feature_count": len(feature_names),
        "test_rows": len(predictions),
        "error_count": len(error_table),
        "high_confidence_error_count": int(
            error_table["high_confidence_error"].astype(bool).sum()
        ),
        "top_combined_features": (top_combined_features),
        "strongest_positive_features": (strongest_positive_features),
        "strongest_negative_features": (strongest_negative_features),
        "output_validation": (output_validation),
        "artifact_validation": (artifact_validation),
        "interpretation_policy": {
            "coefficient_meaning": (
                "Effect on log odds for a one-standard-deviation "
                "change in a continuous feature or a zero-to-one "
                "change in a binary feature."
            ),
            "causal_claims_allowed": False,
            "permutation_importance_use": ("Post-hoc interpretation only"),
            "test_used_for_retraining": False,
            "test_used_for_reselection": False,
            "threshold_changed": False,
        },
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    print("Stage 6G Model 2 explainability and error analysis")
    print("=" * 104)

    print()
    print("Strongest features increasing delay risk:")

    print(
        coefficient_table.loc[
            coefficient_table["coefficient"] > 0,
            [
                "coefficient_rank",
                "feature",
                "coefficient",
                "odds_ratio",
            ],
        ]
        .sort_values(
            "coefficient",
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )

    print()
    print("Strongest features reducing delay risk:")

    print(
        coefficient_table.loc[
            coefficient_table["coefficient"] < 0,
            [
                "coefficient_rank",
                "feature",
                "coefficient",
                "odds_ratio",
            ],
        ]
        .sort_values("coefficient")
        .head(15)
        .to_string(index=False)
    )

    print()
    print("Top combined explainability features:")

    print(
        combined_table[
            [
                "combined_rank",
                "feature",
                "coefficient",
                "odds_ratio",
                "permutation_importance_mean",
                "average_rank",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print()
    print("Probability-band summary:")

    print(probability_bands.to_string(index=False))

    print()
    print("Error analysis:")

    print(error_table.to_string(index=False))

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
        "Model reselected:",
        False,
    )

    print(
        "Threshold changed:",
        False,
    )

    print()
    print(
        "Overall Stage 6G verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
