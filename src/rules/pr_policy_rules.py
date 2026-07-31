"""Deterministic policy and risk rules for pull requests."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PR_IDENTIFIER_COLUMN = "pr_number"

RISK_BAND_ORDER = [
    "Low",
    "Moderate",
    "High",
    "Critical",
]


@dataclass(frozen=True)
class RuleDefinition:
    """One transparent PR policy rule."""

    rule_id: str
    rule_name: str
    category: str
    severity: str
    weight: int
    description: str
    recommendation: str
    evaluator: Callable[
        [pd.Series],
        tuple[bool, str],
    ]


def normalize_boolean_value(
    value: Any,
    default: bool = False,
) -> bool:
    """Normalize one Boolean-like value."""

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(
        value,
        bool | np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        int | float | np.integer | np.floating,
    ):
        return bool(int(value))

    normalized = str(value).strip().lower()

    true_values = {
        "true",
        "yes",
        "y",
        "1",
        "present",
    }

    false_values = {
        "false",
        "no",
        "n",
        "0",
        "none",
        "",
    }

    if normalized in true_values:
        return True

    if normalized in false_values:
        return False

    return default


def get_boolean(
    row: pd.Series,
    column: str,
    default: bool = False,
) -> bool:
    """Read one Boolean feature safely."""

    if column not in row.index:
        return default

    return normalize_boolean_value(
        row[column],
        default=default,
    )


def get_numeric(
    row: pd.Series,
    column: str,
    default: float = 0.0,
) -> float:
    """Read one numeric feature safely."""

    if column not in row.index:
        return float(default)

    value = pd.to_numeric(
        pd.Series(
            [
                row[column],
            ]
        ),
        errors="coerce",
    ).iloc[0]

    if pd.isna(value):
        return float(default)

    numeric_value = float(value)

    if not np.isfinite(numeric_value):
        return float(default)

    return numeric_value


def evaluate_missing_description(
    row: pd.Series,
) -> tuple[bool, str]:
    """Check whether the PR has no description."""

    has_description = get_boolean(
        row,
        "has_description",
    )

    body_length = get_numeric(
        row,
        "body_length",
    )

    triggered = not has_description or body_length <= 0

    evidence = f"has_description={has_description}; body_length={body_length:.0f}"

    return triggered, evidence


def evaluate_weak_description(
    row: pd.Series,
) -> tuple[bool, str]:
    """Check whether the description exists but lacks detail."""

    has_description = get_boolean(
        row,
        "has_description",
    )

    detailed_description = get_boolean(
        row,
        "has_detailed_description",
    )

    body_word_count = get_numeric(
        row,
        "body_word_count",
    )

    triggered = bool(
        has_description and (not detailed_description or body_word_count < 20)
    )

    evidence = (
        f"has_description={has_description}; "
        f"has_detailed_description={detailed_description}; "
        f"body_word_count={body_word_count:.0f}"
    )

    return triggered, evidence


def evaluate_missing_tests(
    row: pd.Series,
) -> tuple[bool, str]:
    """Check for substantive changes without test changes."""

    has_test_changes = get_boolean(
        row,
        "has_test_changes",
    )

    total_changes = get_numeric(
        row,
        "total_changes",
    )

    changed_files = get_numeric(
        row,
        "changed_files",
    )

    security_changes = get_boolean(
        row,
        "has_security_sensitive_changes",
    )

    configuration_changes = get_boolean(
        row,
        "has_configuration_changes",
    )

    substantive_change = bool(
        total_changes >= 50
        or changed_files >= 5
        or security_changes
        or configuration_changes
    )

    triggered = bool(substantive_change and not has_test_changes)

    evidence = (
        f"has_test_changes={has_test_changes}; "
        f"total_changes={total_changes:.0f}; "
        f"changed_files={changed_files:.0f}; "
        f"security_changes={security_changes}; "
        f"configuration_changes={configuration_changes}"
    )

    return triggered, evidence


def evaluate_missing_documentation(
    row: pd.Series,
) -> tuple[bool, str]:
    """Check for substantial changes without documentation updates."""

    has_documentation_changes = get_boolean(
        row,
        "has_documentation_changes",
    )

    total_changes = get_numeric(
        row,
        "total_changes",
    )

    changed_files = get_numeric(
        row,
        "changed_files",
    )

    configuration_changes = get_boolean(
        row,
        "has_configuration_changes",
    )

    documentation_expected = bool(
        total_changes >= 100 or changed_files >= 10 or configuration_changes
    )

    triggered = bool(documentation_expected and not has_documentation_changes)

    evidence = (
        f"has_documentation_changes="
        f"{has_documentation_changes}; "
        f"total_changes={total_changes:.0f}; "
        f"changed_files={changed_files:.0f}; "
        f"configuration_changes={configuration_changes}"
    )

    return triggered, evidence


def evaluate_security_sensitive_changes(
    row: pd.Series,
) -> tuple[bool, str]:
    """Detect security-sensitive changed files."""

    security_flag = get_boolean(
        row,
        "has_security_sensitive_changes",
    )

    security_file_count = get_numeric(
        row,
        "security_sensitive_files_changed",
    )

    triggered = bool(security_flag or security_file_count > 0)

    evidence = (
        f"has_security_sensitive_changes={security_flag}; "
        f"security_sensitive_files_changed="
        f"{security_file_count:.0f}"
    )

    return triggered, evidence


def evaluate_security_without_tests(
    row: pd.Series,
) -> tuple[bool, str]:
    """Detect security-sensitive changes without test updates."""

    security_flag = get_boolean(
        row,
        "has_security_sensitive_changes",
    )

    security_file_count = get_numeric(
        row,
        "security_sensitive_files_changed",
    )

    has_test_changes = get_boolean(
        row,
        "has_test_changes",
    )

    has_security_change = bool(security_flag or security_file_count > 0)

    triggered = bool(has_security_change and not has_test_changes)

    evidence = (
        f"security_change={has_security_change}; "
        f"security_sensitive_files_changed="
        f"{security_file_count:.0f}; "
        f"has_test_changes={has_test_changes}"
    )

    return triggered, evidence


def evaluate_configuration_changes(
    row: pd.Series,
) -> tuple[bool, str]:
    """Detect configuration-related changes."""

    configuration_flag = get_boolean(
        row,
        "has_configuration_changes",
    )

    configuration_file_count = get_numeric(
        row,
        "configuration_files_changed",
    )

    triggered = bool(configuration_flag or configuration_file_count > 0)

    evidence = (
        f"has_configuration_changes={configuration_flag}; "
        f"configuration_files_changed="
        f"{configuration_file_count:.0f}"
    )

    return triggered, evidence


def evaluate_large_pr(
    row: pd.Series,
) -> tuple[bool, str]:
    """Detect a large pull request."""

    total_changes = get_numeric(
        row,
        "total_changes",
    )

    changed_files = get_numeric(
        row,
        "changed_files",
    )

    triggered = bool(
        (total_changes >= 500 or changed_files >= 20)
        and not (total_changes >= 1500 or changed_files >= 50)
    )

    evidence = f"total_changes={total_changes:.0f}; changed_files={changed_files:.0f}"

    return triggered, evidence


def evaluate_very_large_pr(
    row: pd.Series,
) -> tuple[bool, str]:
    """Detect a very large pull request."""

    total_changes = get_numeric(
        row,
        "total_changes",
    )

    changed_files = get_numeric(
        row,
        "changed_files",
    )

    triggered = bool(total_changes >= 1500 or changed_files >= 50)

    evidence = f"total_changes={total_changes:.0f}; changed_files={changed_files:.0f}"

    return triggered, evidence


def evaluate_high_commit_complexity(
    row: pd.Series,
) -> tuple[bool, str]:
    """Detect unusually high commit activity."""

    commit_count = get_numeric(
        row,
        "commit_count",
    )

    triggered = bool(commit_count >= 10)

    evidence = f"commit_count={commit_count:.0f}"

    return triggered, evidence


def evaluate_missing_reviewer(
    row: pd.Series,
) -> tuple[bool, str]:
    """Detect risk-bearing PRs without a requested reviewer."""

    requested_reviewers = get_numeric(
        row,
        "requested_reviewer_count",
    )

    total_changes = get_numeric(
        row,
        "total_changes",
    )

    changed_files = get_numeric(
        row,
        "changed_files",
    )

    security_changes = get_boolean(
        row,
        "has_security_sensitive_changes",
    )

    configuration_changes = get_boolean(
        row,
        "has_configuration_changes",
    )

    reviewer_expected = bool(
        total_changes >= 500
        or changed_files >= 20
        or security_changes
        or configuration_changes
    )

    triggered = bool(reviewer_expected and requested_reviewers <= 0)

    evidence = (
        f"requested_reviewer_count="
        f"{requested_reviewers:.0f}; "
        f"total_changes={total_changes:.0f}; "
        f"changed_files={changed_files:.0f}; "
        f"security_changes={security_changes}; "
        f"configuration_changes={configuration_changes}"
    )

    return triggered, evidence


def evaluate_outlier_pattern(
    row: pd.Series,
) -> tuple[bool, str]:
    """Detect multiple unusual PR characteristics."""

    has_any_outlier = get_boolean(
        row,
        "has_any_iqr_outlier",
    )

    outlier_count = get_numeric(
        row,
        "iqr_outlier_feature_count",
    )

    triggered = bool(outlier_count >= 3 or (has_any_outlier and outlier_count >= 2))

    evidence = (
        f"has_any_iqr_outlier={has_any_outlier}; "
        f"iqr_outlier_feature_count="
        f"{outlier_count:.0f}"
    )

    return triggered, evidence


def build_rule_definitions() -> list[RuleDefinition]:
    """Return the complete Stage 7A policy-rule catalogue."""

    return [
        RuleDefinition(
            rule_id="PR001",
            rule_name="Missing PR description",
            category="Documentation",
            severity="High",
            weight=18,
            description=("The pull request does not provide a usable description."),
            recommendation=(
                "Add a clear description covering the purpose, "
                "scope, expected behaviour and validation performed."
            ),
            evaluator=(evaluate_missing_description),
        ),
        RuleDefinition(
            rule_id="PR002",
            rule_name="Weak PR description",
            category="Documentation",
            severity="Moderate",
            weight=8,
            description=(
                "The description exists but appears too short "
                "or insufficiently detailed."
            ),
            recommendation=(
                "Expand the description with implementation context, "
                "testing evidence and reviewer guidance."
            ),
            evaluator=(evaluate_weak_description),
        ),
        RuleDefinition(
            rule_id="PR003",
            rule_name="Substantive change without tests",
            category="Testing",
            severity="High",
            weight=22,
            description=(
                "A substantive code, configuration or security "
                "change has no detected test-file update."
            ),
            recommendation=(
                "Add or update automated tests, or document why "
                "additional tests are not required."
            ),
            evaluator=(evaluate_missing_tests),
        ),
        RuleDefinition(
            rule_id="PR004",
            rule_name=("Substantial change without documentation"),
            category="Documentation",
            severity="Moderate",
            weight=10,
            description=(
                "A substantial or configuration-related change "
                "has no detected documentation update."
            ),
            recommendation=(
                "Update user, developer or operational documentation "
                "where the change affects expected behaviour."
            ),
            evaluator=(evaluate_missing_documentation),
        ),
        RuleDefinition(
            rule_id="PR005",
            rule_name="Security-sensitive file change",
            category="Security",
            severity="High",
            weight=20,
            description=(
                "The PR changes one or more files classified as security-sensitive."
            ),
            recommendation=(
                "Require focused security review and verify that "
                "secrets, authentication and permission impacts "
                "have been assessed."
            ),
            evaluator=(evaluate_security_sensitive_changes),
        ),
        RuleDefinition(
            rule_id="PR006",
            rule_name=("Security-sensitive change without tests"),
            category="Security",
            severity="Critical",
            weight=28,
            description=(
                "Security-sensitive files changed without a detected test update."
            ),
            recommendation=(
                "Block approval until security-relevant tests or "
                "documented validation evidence are supplied."
            ),
            evaluator=(evaluate_security_without_tests),
        ),
        RuleDefinition(
            rule_id="PR007",
            rule_name="Configuration change",
            category="Operations",
            severity="Moderate",
            weight=10,
            description=("The PR modifies configuration-related files."),
            recommendation=(
                "Confirm environment impact, deployment sequencing, "
                "rollback instructions and configuration validation."
            ),
            evaluator=(evaluate_configuration_changes),
        ),
        RuleDefinition(
            rule_id="PR008",
            rule_name="Large pull request",
            category="Complexity",
            severity="Moderate",
            weight=12,
            description=(
                "The PR contains at least 500 changed lines or 20 changed files."
            ),
            recommendation=(
                "Consider splitting the change or assign additional "
                "review capacity to reduce review risk."
            ),
            evaluator=(evaluate_large_pr),
        ),
        RuleDefinition(
            rule_id="PR009",
            rule_name="Very large pull request",
            category="Complexity",
            severity="High",
            weight=22,
            description=(
                "The PR contains at least 1,500 changed lines or 50 changed files."
            ),
            recommendation=(
                "Split the PR where practical and require a structured "
                "review plan before approval."
            ),
            evaluator=(evaluate_very_large_pr),
        ),
        RuleDefinition(
            rule_id="PR010",
            rule_name="High commit complexity",
            category="Complexity",
            severity="Moderate",
            weight=8,
            description=("The PR contains ten or more commits."),
            recommendation=(
                "Review commit organisation and consider squashing "
                "or restructuring commits for easier review."
            ),
            evaluator=(evaluate_high_commit_complexity),
        ),
        RuleDefinition(
            rule_id="PR011",
            rule_name="Risk-bearing PR without reviewer",
            category="Governance",
            severity="High",
            weight=18,
            description=(
                "A large, security-sensitive or configuration-related "
                "PR has no requested reviewer."
            ),
            recommendation=(
                "Assign an appropriate reviewer before the PR "
                "progresses toward approval."
            ),
            evaluator=(evaluate_missing_reviewer),
        ),
        RuleDefinition(
            rule_id="PR012",
            rule_name="Multiple unusual PR characteristics",
            category="Complexity",
            severity="Moderate",
            weight=10,
            description=(
                "The PR has multiple features outside historical interquartile ranges."
            ),
            recommendation=(
                "Perform additional manual review because several "
                "PR characteristics are unusual for this repository."
            ),
            evaluator=(evaluate_outlier_pattern),
        ),
    ]


def calculate_risk_band(
    risk_score: int,
) -> str:
    """Convert a risk score into an operational risk band."""

    if risk_score >= 70:
        return "Critical"

    if risk_score >= 40:
        return "High"

    if risk_score >= 20:
        return "Moderate"

    return "Low"


def validate_source_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the rule-engine source dataset."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        "has_description",
        "body_length",
        "body_word_count",
        "has_detailed_description",
        "has_test_changes",
        "has_documentation_changes",
        "has_security_sensitive_changes",
        "has_configuration_changes",
        "total_changes",
        "changed_files",
        "commit_count",
        "requested_reviewer_count",
        "has_any_iqr_outlier",
        "iqr_outlier_feature_count",
    }

    missing_required_columns = sorted(required_columns - set(dataframe.columns))

    duplicate_pr_count = (
        int(
            dataframe.duplicated(
                subset=[
                    PR_IDENTIFIER_COLUMN,
                ]
            ).sum()
        )
        if PR_IDENTIFIER_COLUMN in dataframe.columns
        else None
    )

    validation_passed = bool(
        not missing_required_columns and duplicate_pr_count == 0 and len(dataframe) > 0
    )

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "missing_required_columns": (missing_required_columns),
        "duplicate_pr_count": (duplicate_pr_count),
        "validation_passed": (validation_passed),
    }


def evaluate_pr_rules(
    dataframe: pd.DataFrame,
    rules: list[RuleDefinition] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate all deterministic rules for every PR."""

    source_validation = validate_source_dataset(dataframe)

    if not source_validation["validation_passed"]:
        raise ValueError(f"Rule source dataset failed validation: {source_validation}")

    active_rules = rules if rules is not None else build_rule_definitions()

    if not active_rules:
        raise ValueError("No policy rules were supplied.")

    duplicate_rule_ids = len({rule.rule_id for rule in active_rules}) != len(
        active_rules
    )

    if duplicate_rule_ids:
        raise ValueError("Policy-rule identifiers must be unique.")

    long_records: list[dict[str, Any]] = []

    summary_records: list[dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        pr_number = row[PR_IDENTIFIER_COLUMN]

        triggered_rules = []
        recommendations = []
        triggered_categories = []
        raw_score = 0
        critical_rule_count = 0
        high_rule_count = 0

        for rule in active_rules:
            triggered, evidence = rule.evaluator(row)

            if triggered:
                raw_score += rule.weight
                triggered_rules.append(rule.rule_id)
                recommendations.append(rule.recommendation)
                triggered_categories.append(rule.category)

                if rule.severity == "Critical":
                    critical_rule_count += 1

                if rule.severity == "High":
                    high_rule_count += 1

            long_records.append(
                {
                    "pr_number": (pr_number),
                    "rule_id": (rule.rule_id),
                    "rule_name": (rule.rule_name),
                    "category": (rule.category),
                    "severity": (rule.severity),
                    "weight": int(rule.weight),
                    "triggered": bool(triggered),
                    "evidence": (evidence),
                    "description": (rule.description),
                    "recommendation": (rule.recommendation),
                }
            )

        risk_score = min(
            int(raw_score),
            100,
        )

        risk_band = calculate_risk_band(risk_score)

        summary_records.append(
            {
                "pr_number": (pr_number),
                "policy_risk_score": (risk_score),
                "policy_risk_band": (risk_band),
                "triggered_rule_count": len(triggered_rules),
                "critical_rule_count": (critical_rule_count),
                "high_rule_count": (high_rule_count),
                "triggered_rules": (" | ".join(triggered_rules)),
                "triggered_categories": (" | ".join(sorted(set(triggered_categories)))),
                "recommended_actions": (" | ".join(dict.fromkeys(recommendations))),
                "manual_review_required": bool(
                    risk_score >= 40 or critical_rule_count > 0
                ),
            }
        )

    long_table = pd.DataFrame(long_records)

    summary_table = pd.DataFrame(summary_records)

    return (
        summary_table,
        long_table,
    )


def build_rule_summary(
    long_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize rule trigger frequency."""

    required_columns = {
        "rule_id",
        "rule_name",
        "category",
        "severity",
        "weight",
        "triggered",
    }

    missing_columns = sorted(required_columns - set(long_results.columns))

    if missing_columns:
        raise ValueError(f"Long rule results are missing columns: {missing_columns}")

    summary = (
        long_results.groupby(
            [
                "rule_id",
                "rule_name",
                "category",
                "severity",
                "weight",
            ],
            observed=False,
        )
        .agg(
            evaluated_pr_count=(
                "triggered",
                "size",
            ),
            triggered_pr_count=(
                "triggered",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["trigger_rate"] = (
        summary["triggered_pr_count"] / summary["evaluated_pr_count"]
    )

    return summary.sort_values(
        [
            "triggered_pr_count",
            "weight",
            "rule_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)


def build_risk_band_summary(
    summary_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize PR distribution by risk band."""

    required_columns = {
        "policy_risk_band",
        "policy_risk_score",
        "manual_review_required",
    }

    missing_columns = sorted(required_columns - set(summary_results.columns))

    if missing_columns:
        raise ValueError(f"PR summary results are missing columns: {missing_columns}")

    risk_summary = (
        summary_results.groupby(
            "policy_risk_band",
            observed=False,
        )
        .agg(
            pr_count=(
                "policy_risk_band",
                "size",
            ),
            average_risk_score=(
                "policy_risk_score",
                "mean",
            ),
            minimum_risk_score=(
                "policy_risk_score",
                "min",
            ),
            maximum_risk_score=(
                "policy_risk_score",
                "max",
            ),
            manual_review_count=(
                "manual_review_required",
                "sum",
            ),
        )
        .reset_index()
    )

    risk_summary["risk_band_order"] = risk_summary["policy_risk_band"].map(
        {
            risk_band: position
            for position, risk_band in enumerate(
                RISK_BAND_ORDER,
                start=1,
            )
        }
    )

    return (
        risk_summary.sort_values("risk_band_order")
        .drop(
            columns=[
                "risk_band_order",
            ]
        )
        .reset_index(drop=True)
    )


def validate_rule_outputs(
    source_dataframe: pd.DataFrame,
    summary_results: pd.DataFrame,
    long_results: pd.DataFrame,
    rule_count: int,
) -> dict[str, Any]:
    """Validate Stage 7A rule-engine outputs."""

    expected_pr_count = len(source_dataframe)

    expected_long_row_count = expected_pr_count * rule_count

    summary_row_count_valid = bool(len(summary_results) == expected_pr_count)

    long_row_count_valid = bool(len(long_results) == expected_long_row_count)

    unique_pr_count_valid = bool(
        summary_results[PR_IDENTIFIER_COLUMN].nunique() == expected_pr_count
    )

    score_range_valid = bool(
        summary_results["policy_risk_score"]
        .between(
            0,
            100,
            inclusive="both",
        )
        .all()
    )

    risk_bands_valid = bool(
        set(summary_results["policy_risk_band"].unique()).issubset(set(RISK_BAND_ORDER))
    )

    triggered_values_valid = bool(
        long_results["triggered"]
        .isin(
            [
                True,
                False,
            ]
        )
        .all()
    )

    triggered_rule_count_valid = bool(
        (
            long_results.groupby(PR_IDENTIFIER_COLUMN)["triggered"]
            .sum()
            .astype(int)
            .sort_index()
            .to_numpy()
            == summary_results.set_index(PR_IDENTIFIER_COLUMN)["triggered_rule_count"]
            .sort_index()
            .astype(int)
            .to_numpy()
        ).all()
    )

    validation_passed = bool(
        summary_row_count_valid
        and long_row_count_valid
        and unique_pr_count_valid
        and score_range_valid
        and risk_bands_valid
        and triggered_values_valid
        and triggered_rule_count_valid
    )

    return {
        "expected_pr_count": (expected_pr_count),
        "summary_row_count": len(summary_results),
        "summary_row_count_valid": (summary_row_count_valid),
        "rule_count": (rule_count),
        "expected_long_row_count": (expected_long_row_count),
        "actual_long_row_count": len(long_results),
        "long_row_count_valid": (long_row_count_valid),
        "unique_pr_count_valid": (unique_pr_count_valid),
        "score_range_valid": (score_range_valid),
        "risk_bands_valid": (risk_bands_valid),
        "triggered_values_valid": (triggered_values_valid),
        "triggered_rule_count_valid": (triggered_rule_count_valid),
        "validation_passed": (validation_passed),
    }
