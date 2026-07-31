"""Build the unified structured PR intelligence dataset."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.data.discovery import load_dataset
from src.intelligence.unified_pr_intelligence import (
    build_priority_summary,
    build_unified_dataset,
    score_binary_model,
    validate_unified_outputs,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    PROJECT_ROOT,
    REPORTS_DIRECTORY,
)

CORE_DATASET_PATH = (
    PROCESSED_DATA_DIRECTORY / "pallets_flask_corrected_600_feature_engineered.csv"
)

MODEL1_SCORING_DATASET_PATH = (
    PROCESSED_DATA_DIRECTORY / "model1_merge_outcome_approved_features.csv"
)

MODEL2_SCORING_DATASET_PATH = (
    PROCESSED_DATA_DIRECTORY / "model2_merge_delay_approved_features.csv"
)

POLICY_DATASET_PATH = PROCESSED_DATA_DIRECTORY / "pr_policy_intelligence.csv"

MODEL1_PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "model1_preprocessor.joblib"

MODEL1_FINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "merge_outcome"
    / "final_contributor_neutral_merge_outcome_model.joblib"
)

MODEL1_CONFIGURATION_PATH = (
    REPORTS_DIRECTORY / "stage_5e2_locked_model_configuration.json"
)

MODEL2_PREPROCESSOR_PATH = (
    PROJECT_ROOT / "models" / "merge_delay" / "model2_preprocessor.joblib"
)

MODEL2_FINAL_MODEL_PATH = (
    PROJECT_ROOT / "models" / "merge_delay" / "final_merge_delay_model.joblib"
)

MODEL2_CONFIGURATION_PATH = (
    REPORTS_DIRECTORY / "stage_6e_locked_model_configuration.json"
)

STAGE_7A_COMPLETION_PATH = REPORTS_DIRECTORY / "stage_7a_completion_report.json"

OUTPUT_DATASET_PATH = PROCESSED_DATA_DIRECTORY / "unified_pr_intelligence.csv"

MODEL1_SCORE_PATH = REPORTS_DIRECTORY / "stage_7b_model1_all_pr_scores.csv"

MODEL2_SCORE_PATH = REPORTS_DIRECTORY / "stage_7b_model2_merged_pr_scores.csv"

PRIORITY_SUMMARY_PATH = REPORTS_DIRECTORY / "stage_7b_review_priority_summary.csv"

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_7b_completion_report.json"


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


def load_json(
    input_path: Path,
) -> dict[str, Any]:
    """Load one JSON file."""

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        result = json.load(input_file)

    if not isinstance(
        result,
        dict,
    ):
        raise ValueError(f"Expected a JSON object in {input_path}.")

    return result


def main() -> int:
    """Run Stage 7B unified intelligence generation."""

    required_paths = [
        CORE_DATASET_PATH,
        MODEL1_SCORING_DATASET_PATH,
        MODEL2_SCORING_DATASET_PATH,
        POLICY_DATASET_PATH,
        MODEL1_PREPROCESSOR_PATH,
        MODEL1_FINAL_MODEL_PATH,
        MODEL1_CONFIGURATION_PATH,
        MODEL2_PREPROCESSOR_PATH,
        MODEL2_FINAL_MODEL_PATH,
        MODEL2_CONFIGURATION_PATH,
        STAGE_7A_COMPLETION_PATH,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 7B files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    try:
        stage_7a_report = load_json(STAGE_7A_COMPLETION_PATH)

        if not stage_7a_report.get(
            "overall_verification_passed",
            False,
        ):
            raise ValueError("Stage 7A did not pass verification.")

        model1_configuration = load_json(MODEL1_CONFIGURATION_PATH)

        model2_configuration = load_json(MODEL2_CONFIGURATION_PATH)

        core_dataframe = load_dataset(CORE_DATASET_PATH)

        model1_scoring_dataframe = load_dataset(MODEL1_SCORING_DATASET_PATH)

        model2_scoring_dataframe = load_dataset(MODEL2_SCORING_DATASET_PATH)

        policy_dataframe = load_dataset(POLICY_DATASET_PATH)

        model1_preprocessor = joblib.load(MODEL1_PREPROCESSOR_PATH)

        model1_model = joblib.load(MODEL1_FINAL_MODEL_PATH)

        model2_preprocessor = joblib.load(MODEL2_PREPROCESSOR_PATH)

        model2_model = joblib.load(MODEL2_FINAL_MODEL_PATH)

        model1_scores = score_binary_model(
            dataframe=(model1_scoring_dataframe),
            model=model1_model,
            preprocessor=(model1_preprocessor),
            configuration=(model1_configuration),
            probability_column=("merge_probability"),
            prediction_column=("merge_prediction"),
            confidence_column=("merge_prediction_confidence"),
            threshold_column=("merge_prediction_threshold"),
        )

        model2_scores = score_binary_model(
            dataframe=(model2_scoring_dataframe),
            model=model2_model,
            preprocessor=(model2_preprocessor),
            configuration=(model2_configuration),
            probability_column=("delay_probability"),
            prediction_column=("delay_prediction"),
            confidence_column=("delay_prediction_confidence"),
            threshold_column=("delay_prediction_threshold"),
        )

        unified_dataframe = build_unified_dataset(
            core_dataframe=(core_dataframe),
            merge_scores=(model1_scores),
            delay_scores=(model2_scores),
            policy_scores=(policy_dataframe),
        )

        priority_summary = build_priority_summary(unified_dataframe)

        output_validation = validate_unified_outputs(
            core_dataframe=(core_dataframe),
            unified_dataframe=(unified_dataframe),
            merge_scores=(model1_scores),
            delay_scores=(model2_scores),
            policy_scores=(policy_dataframe),
        )

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    unified_dataframe.to_csv(
        OUTPUT_DATASET_PATH,
        index=False,
        encoding="utf-8",
    )

    model1_scores.to_csv(
        MODEL1_SCORE_PATH,
        index=False,
        encoding="utf-8",
    )

    model2_scores.to_csv(
        MODEL2_SCORE_PATH,
        index=False,
        encoding="utf-8",
    )

    priority_summary.to_csv(
        PRIORITY_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    artifact_validation = {
        "unified_dataset_exists": (OUTPUT_DATASET_PATH.exists()),
        "model1_score_report_exists": (MODEL1_SCORE_PATH.exists()),
        "model2_score_report_exists": (MODEL2_SCORE_PATH.exists()),
        "priority_summary_exists": (PRIORITY_SUMMARY_PATH.exists()),
        "model1_retrained": False,
        "model2_retrained": False,
        "model1_threshold_changed": False,
        "model2_threshold_changed": False,
    }

    artifact_validation["validation_passed"] = bool(
        artifact_validation["unified_dataset_exists"]
        and artifact_validation["model1_score_report_exists"]
        and artifact_validation["model2_score_report_exists"]
        and artifact_validation["priority_summary_exists"]
        and not artifact_validation["model1_retrained"]
        and not artifact_validation["model2_retrained"]
        and not artifact_validation["model1_threshold_changed"]
        and not artifact_validation["model2_threshold_changed"]
    )

    priority_counts = unified_dataframe["review_priority"].value_counts().to_dict()

    overall_passed = bool(
        output_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 7B",
        "component": ("Unified structured PR intelligence dataset"),
        "source_pr_count": len(core_dataframe),
        "model1_score_count": len(model1_scores),
        "model2_score_count": len(model2_scores),
        "policy_score_count": len(policy_dataframe),
        "unified_row_count": len(unified_dataframe),
        "priority_counts": (priority_counts),
        "output_validation": (output_validation),
        "artifact_validation": (artifact_validation),
        "model_usage_policy": {
            "model1_retrained": False,
            "model2_retrained": False,
            "model1_threshold_changed": False,
            "model2_threshold_changed": False,
            "model1_usage": ("Inference across all 600 historical PR snapshots"),
            "model2_usage": ("Inference only for the 300 merged-PR population"),
            "automatic_merge_action": False,
            "human_review_required": True,
        },
        "review_priority_policy": {
            "policy_score_weight": 0.65,
            "maximum_merge_blocker_component": 20,
            "maximum_delay_component": 15,
            "priority_bands": [
                "Routine",
                "Moderate",
                "High",
                "Critical",
            ],
            "purpose": ("Review prioritisation and decision support only"),
        },
        "output_dataset": str(OUTPUT_DATASET_PATH),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    display_columns = [
        "pr_number",
        "review_priority",
        "review_priority_score",
        "policy_risk_band",
        "policy_risk_score",
        "merge_probability",
        "merge_prediction",
        "delay_score_available",
        "delay_probability",
        "delay_prediction",
        "triggered_rule_count",
        "manual_review_required",
    ]

    print("Stage 7B unified PR intelligence dataset")
    print("=" * 108)

    print()
    print("Scoring coverage:")

    print(
        json.dumps(
            {
                "core_pr_count": len(core_dataframe),
                "model1_score_count": len(model1_scores),
                "model2_score_count": len(model2_scores),
                "policy_score_count": len(policy_dataframe),
            },
            indent=2,
        )
    )

    print()
    print("Review-priority summary:")

    print(priority_summary.to_string(index=False))

    print()
    print("Highest-priority PRs:")

    print(unified_dataframe[display_columns].head(20).to_string(index=False))

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
        "Overall Stage 7B verification passed:",
        overall_passed,
    )

    print()
    print("Unified intelligence dataset:")

    print(OUTPUT_DATASET_PATH)

    print()
    print("Completion report:")

    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
