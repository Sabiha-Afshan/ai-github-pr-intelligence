"""Build the deterministic PR policy-intelligence layer."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data.discovery import load_dataset
from src.rules.pr_policy_rules import (
    build_risk_band_summary,
    build_rule_definitions,
    build_rule_summary,
    evaluate_pr_rules,
    validate_rule_outputs,
    validate_source_dataset,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    REPORTS_DIRECTORY,
)

SOURCE_DATASET_PATH = (
    PROCESSED_DATA_DIRECTORY / "pallets_flask_corrected_600_feature_engineered.csv"
)

OUTPUT_DATASET_PATH = PROCESSED_DATA_DIRECTORY / "pr_policy_intelligence.csv"

LONG_RESULTS_PATH = REPORTS_DIRECTORY / "stage_7a_rule_results_long.csv"

RULE_SUMMARY_PATH = REPORTS_DIRECTORY / "stage_7a_rule_summary.csv"

RISK_BAND_SUMMARY_PATH = REPORTS_DIRECTORY / "stage_7a_risk_band_summary.csv"

RULE_CATALOGUE_PATH = REPORTS_DIRECTORY / "stage_7a_rule_catalogue.csv"

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_7a_completion_report.json"


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
    """Run Stage 7A deterministic rule evaluation."""

    if not SOURCE_DATASET_PATH.exists():
        print("FAIL: Stage 7A source dataset is missing:")
        print(SOURCE_DATASET_PATH)
        return 1

    try:
        source_dataframe = load_dataset(SOURCE_DATASET_PATH)

        source_validation = validate_source_dataset(source_dataframe)

        if not source_validation["validation_passed"]:
            raise ValueError(
                f"Policy-rule source data failed validation: {source_validation}"
            )

        rules = build_rule_definitions()

        (
            policy_summary,
            long_results,
        ) = evaluate_pr_rules(
            dataframe=source_dataframe,
            rules=rules,
        )

        rule_summary = build_rule_summary(long_results)

        risk_band_summary = build_risk_band_summary(policy_summary)

        output_validation = validate_rule_outputs(
            source_dataframe=(source_dataframe),
            summary_results=(policy_summary),
            long_results=(long_results),
            rule_count=len(rules),
        )

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    rule_catalogue = [
        {
            "rule_id": (rule.rule_id),
            "rule_name": (rule.rule_name),
            "category": (rule.category),
            "severity": (rule.severity),
            "weight": (rule.weight),
            "description": (rule.description),
            "recommendation": (rule.recommendation),
        }
        for rule in rules
    ]

    import pandas as pd

    rule_catalogue_dataframe = pd.DataFrame(rule_catalogue)

    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy_summary.to_csv(
        OUTPUT_DATASET_PATH,
        index=False,
        encoding="utf-8",
    )

    long_results.to_csv(
        LONG_RESULTS_PATH,
        index=False,
        encoding="utf-8",
    )

    rule_summary.to_csv(
        RULE_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    risk_band_summary.to_csv(
        RISK_BAND_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    rule_catalogue_dataframe.to_csv(
        RULE_CATALOGUE_PATH,
        index=False,
        encoding="utf-8",
    )

    artifact_validation = {
        "policy_dataset_exists": (OUTPUT_DATASET_PATH.exists()),
        "long_results_exist": (LONG_RESULTS_PATH.exists()),
        "rule_summary_exists": (RULE_SUMMARY_PATH.exists()),
        "risk_band_summary_exists": (RISK_BAND_SUMMARY_PATH.exists()),
        "rule_catalogue_exists": (RULE_CATALOGUE_PATH.exists()),
    }

    artifact_validation["validation_passed"] = bool(all(artifact_validation.values()))

    manual_review_count = int(
        policy_summary["manual_review_required"].astype(bool).sum()
    )

    high_or_critical_count = int(
        policy_summary["policy_risk_band"]
        .isin(
            [
                "High",
                "Critical",
            ]
        )
        .sum()
    )

    overall_passed = bool(
        source_validation["validation_passed"]
        and output_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 7A",
        "component": ("Deterministic PR policy and risk intelligence"),
        "source_dataset": str(SOURCE_DATASET_PATH),
        "output_dataset": str(OUTPUT_DATASET_PATH),
        "evaluated_pr_count": len(policy_summary),
        "rule_count": len(rules),
        "triggered_rule_count": int(long_results["triggered"].astype(bool).sum()),
        "manual_review_count": (manual_review_count),
        "high_or_critical_pr_count": (high_or_critical_count),
        "source_validation": (source_validation),
        "output_validation": (output_validation),
        "artifact_validation": (artifact_validation),
        "risk_scoring_policy": {
            "minimum_score": 0,
            "maximum_score": 100,
            "risk_bands": {
                "Low": "0–19",
                "Moderate": "20–39",
                "High": "40–69",
                "Critical": "70–100",
            },
            "manual_review_rule": (
                "Required when score is at least 40 or a critical rule is triggered."
            ),
            "automatic_merge_decision": False,
            "human_review_required_for_actions": True,
        },
        "rule_engine_properties": {
            "deterministic": True,
            "auditable": True,
            "model_based": False,
            "llm_based": False,
            "causal_claims_allowed": False,
        },
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    print("Stage 7A deterministic PR policy and risk intelligence")
    print("=" * 104)

    print()
    print("Source validation:")

    print(
        json.dumps(
            source_validation,
            indent=2,
        )
    )

    print()
    print("Rule catalogue:")

    print(
        rule_catalogue_dataframe[
            [
                "rule_id",
                "rule_name",
                "category",
                "severity",
                "weight",
            ]
        ].to_string(index=False)
    )

    print()
    print("Rule trigger summary:")

    print(
        rule_summary[
            [
                "rule_id",
                "rule_name",
                "severity",
                "weight",
                "triggered_pr_count",
                "trigger_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("Risk-band summary:")

    print(risk_band_summary.to_string(index=False))

    print()
    print(
        "Manual-review PR count:",
        manual_review_count,
    )

    print(
        "High/Critical PR count:",
        high_or_critical_count,
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
        "Overall Stage 7A verification passed:",
        overall_passed,
    )

    print()
    print("Policy-intelligence dataset:")

    print(OUTPUT_DATASET_PATH)

    print()
    print("Completion report:")

    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
