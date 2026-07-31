"""Tests for deterministic PR policy rules."""

import pandas as pd

from src.rules.pr_policy_rules import (
    build_risk_band_summary,
    build_rule_definitions,
    build_rule_summary,
    calculate_risk_band,
    evaluate_pr_rules,
    normalize_boolean_value,
    validate_rule_outputs,
    validate_source_dataset,
)


def create_policy_dataset() -> pd.DataFrame:
    """Create representative PR records."""

    return pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
            ],
            "has_description": [
                False,
                True,
                True,
                True,
            ],
            "body_length": [
                0,
                100,
                500,
                250,
            ],
            "body_word_count": [
                0,
                10,
                80,
                40,
            ],
            "has_detailed_description": [
                False,
                False,
                True,
                True,
            ],
            "has_test_changes": [
                False,
                True,
                False,
                True,
            ],
            "has_documentation_changes": [
                False,
                False,
                True,
                True,
            ],
            "has_security_sensitive_changes": [
                False,
                False,
                True,
                False,
            ],
            "security_sensitive_files_changed": [
                0,
                0,
                2,
                0,
            ],
            "has_configuration_changes": [
                False,
                False,
                True,
                False,
            ],
            "configuration_files_changed": [
                0,
                0,
                3,
                0,
            ],
            "total_changes": [
                100,
                30,
                2000,
                20,
            ],
            "changed_files": [
                8,
                2,
                60,
                1,
            ],
            "commit_count": [
                3,
                2,
                14,
                1,
            ],
            "requested_reviewer_count": [
                0,
                1,
                0,
                1,
            ],
            "has_any_iqr_outlier": [
                False,
                False,
                True,
                False,
            ],
            "iqr_outlier_feature_count": [
                0,
                0,
                5,
                0,
            ],
        }
    )


def test_boolean_normalization() -> None:
    """Confirm common Boolean values normalize correctly."""

    assert normalize_boolean_value(True)

    assert normalize_boolean_value("yes")

    assert not normalize_boolean_value("false")

    assert not normalize_boolean_value(None)


def test_risk_band_boundaries() -> None:
    """Confirm risk-score bands."""

    assert calculate_risk_band(0) == "Low"

    assert calculate_risk_band(19) == "Low"

    assert calculate_risk_band(20) == "Moderate"

    assert calculate_risk_band(40) == "High"

    assert calculate_risk_band(70) == "Critical"

    assert calculate_risk_band(100) == "Critical"


def test_source_validation() -> None:
    """Confirm a complete policy dataset passes."""

    result = validate_source_dataset(create_policy_dataset())

    assert result["validation_passed"] is True


def test_rule_catalogue() -> None:
    """Confirm the deterministic catalogue is complete."""

    rules = build_rule_definitions()

    assert len(rules) == 12

    rule_ids = [rule.rule_id for rule in rules]

    assert len(rule_ids) == len(set(rule_ids))


def test_rule_evaluation() -> None:
    """Confirm rules are evaluated for every PR."""

    dataset = create_policy_dataset()

    rules = build_rule_definitions()

    (
        summary,
        long_results,
    ) = evaluate_pr_rules(
        dataframe=dataset,
        rules=rules,
    )

    assert len(summary) == 4

    assert len(long_results) == 48

    assert (
        summary["policy_risk_score"]
        .between(
            0,
            100,
        )
        .all()
    )


def test_high_risk_pr_is_detected() -> None:
    """Confirm the security-heavy PR receives high risk."""

    (
        summary,
        _,
    ) = evaluate_pr_rules(create_policy_dataset())

    high_risk_pr = summary.set_index("pr_number").loc[3]

    assert high_risk_pr["policy_risk_band"] in {
        "High",
        "Critical",
    }

    assert bool(high_risk_pr["manual_review_required"])


def test_rule_and_risk_summaries() -> None:
    """Confirm aggregate reports are produced."""

    (
        summary,
        long_results,
    ) = evaluate_pr_rules(create_policy_dataset())

    rule_summary = build_rule_summary(long_results)

    risk_summary = build_risk_band_summary(summary)

    assert len(rule_summary) == 12

    assert int(risk_summary["pr_count"].sum()) == 4


def test_complete_output_validation() -> None:
    """Confirm complete rule outputs pass validation."""

    dataset = create_policy_dataset()

    rules = build_rule_definitions()

    (
        summary,
        long_results,
    ) = evaluate_pr_rules(
        dataframe=dataset,
        rules=rules,
    )

    validation = validate_rule_outputs(
        source_dataframe=dataset,
        summary_results=summary,
        long_results=long_results,
        rule_count=len(rules),
    )

    assert validation["validation_passed"] is True
