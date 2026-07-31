"""Tests for the unified PR intelligence layer."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.intelligence.unified_pr_intelligence import (
    build_priority_summary,
    build_unified_dataset,
    calculate_review_priority_score,
    classify_review_priority,
    recommend_next_action,
    resolve_configuration_features,
    resolve_configuration_threshold,
    score_binary_model,
    validate_unified_outputs,
)


def create_scoring_data() -> tuple[
    pd.DataFrame,
    StandardScaler,
    LogisticRegression,
    dict,
]:
    """Create a fitted binary model and scoring dataset."""

    dataframe = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
            ],
            "feature_a": [
                0.0,
                1.0,
                2.0,
                3.0,
            ],
            "feature_b": [
                1.0,
                0.0,
                1.0,
                0.0,
            ],
        }
    )

    target = pd.Series(
        [
            0,
            0,
            1,
            1,
        ]
    )

    feature_names = [
        "feature_a",
        "feature_b",
    ]

    preprocessor = StandardScaler()

    transformed_values = preprocessor.fit_transform(dataframe[feature_names])

    transformed_dataframe = pd.DataFrame(
        transformed_values,
        columns=feature_names,
    )

    model = LogisticRegression(
        random_state=42,
    )

    model.fit(
        transformed_dataframe,
        target,
    )

    configuration = {
        "model_name": "test_model",
        "threshold": 0.50,
        "feature_count": 2,
        "features": feature_names,
    }

    return (
        dataframe,
        preprocessor,
        model,
        configuration,
    )


def test_configuration_resolution() -> None:
    """Confirm threshold and features are resolved."""

    configuration = {
        "threshold": 0.60,
        "feature_count": 2,
        "features": [
            "feature_a",
            "feature_b",
        ],
    }

    assert resolve_configuration_threshold(configuration) == 0.60

    assert resolve_configuration_features(configuration) == [
        "feature_a",
        "feature_b",
    ]


def test_binary_model_scoring() -> None:
    """Confirm fitted models score without retraining."""

    (
        dataframe,
        preprocessor,
        model,
        configuration,
    ) = create_scoring_data()

    scores = score_binary_model(
        dataframe=dataframe,
        model=model,
        preprocessor=preprocessor,
        configuration=configuration,
        probability_column=("merge_probability"),
        prediction_column=("merge_prediction"),
        confidence_column=("merge_prediction_confidence"),
        threshold_column=("merge_prediction_threshold"),
    )

    assert len(scores) == 4

    assert (
        scores["merge_probability"]
        .between(
            0,
            1,
        )
        .all()
    )


def test_priority_score_and_classification() -> None:
    """Confirm review-priority scoring."""

    score = calculate_review_priority_score(
        policy_risk_score=60,
        merge_probability=0.30,
        delay_probability=0.80,
    )

    assert 0 <= score <= 100

    priority = classify_review_priority(
        review_priority_score=score,
        policy_risk_band="High",
        manual_review_required=True,
    )

    assert priority == "High"


def test_recommended_actions() -> None:
    """Confirm deterministic operational recommendations."""

    critical_action = recommend_next_action(
        policy_risk_band="Critical",
        manual_review_required=True,
        merge_prediction=1,
        merge_probability=0.90,
        delay_prediction=0,
        delay_probability=0.10,
    )

    delay_action = recommend_next_action(
        policy_risk_band="Low",
        manual_review_required=False,
        merge_prediction=1,
        merge_probability=0.90,
        delay_prediction=1,
        delay_probability=0.85,
    )

    assert "Escalate" in critical_action

    assert "delay" in delay_action.lower()


def create_unified_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create representative unified-layer inputs."""

    core = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
            ],
            "title": [
                "A",
                "B",
                "C",
                "D",
            ],
        }
    )

    merge_scores = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
            ],
            "merge_probability": [
                0.20,
                0.80,
                0.60,
                0.90,
            ],
            "merge_prediction": [
                0,
                1,
                1,
                1,
            ],
            "merge_prediction_confidence": [
                0.80,
                0.80,
                0.60,
                0.90,
            ],
            "merge_prediction_threshold": [
                0.50,
                0.50,
                0.50,
                0.50,
            ],
        }
    )

    delay_scores = pd.DataFrame(
        {
            "pr_number": [
                2,
                4,
            ],
            "delay_probability": [
                0.85,
                0.20,
            ],
            "delay_prediction": [
                1,
                0,
            ],
            "delay_prediction_confidence": [
                0.85,
                0.80,
            ],
            "delay_prediction_threshold": [
                0.75,
                0.75,
            ],
        }
    )

    policy_scores = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
            ],
            "policy_risk_score": [
                80,
                10,
                50,
                0,
            ],
            "policy_risk_band": [
                "Critical",
                "Low",
                "High",
                "Low",
            ],
            "triggered_rule_count": [
                4,
                1,
                2,
                0,
            ],
            "triggered_rules": [
                "PR003 | PR006",
                "PR002",
                "PR003 | PR011",
                "",
            ],
            "triggered_categories": [
                "Security | Testing",
                "Documentation",
                "Governance | Testing",
                "",
            ],
            "recommended_actions": [
                "Escalate",
                "Improve description",
                "Assign reviewer",
                "",
            ],
            "manual_review_required": [
                True,
                False,
                True,
                False,
            ],
        }
    )

    return (
        core,
        merge_scores,
        delay_scores,
        policy_scores,
    )


def test_unified_dataset_creation() -> None:
    """Confirm all intelligence sources combine correctly."""

    (
        core,
        merge_scores,
        delay_scores,
        policy_scores,
    ) = create_unified_inputs()

    unified = build_unified_dataset(
        core_dataframe=core,
        merge_scores=merge_scores,
        delay_scores=delay_scores,
        policy_scores=policy_scores,
    )

    assert len(unified) == 4

    assert int(unified["delay_score_available"].sum()) == 2

    assert {
        "Routine",
        "Moderate",
        "High",
        "Critical",
    }.issuperset(set(unified["review_priority"]))


def test_priority_summary() -> None:
    """Confirm priority aggregation preserves all rows."""

    (
        core,
        merge_scores,
        delay_scores,
        policy_scores,
    ) = create_unified_inputs()

    unified = build_unified_dataset(
        core,
        merge_scores,
        delay_scores,
        policy_scores,
    )

    summary = build_priority_summary(unified)

    assert int(summary["pr_count"].sum()) == 4


def test_unified_output_validation() -> None:
    """Confirm complete unified outputs pass validation."""

    (
        core,
        merge_scores,
        delay_scores,
        policy_scores,
    ) = create_unified_inputs()

    unified = build_unified_dataset(
        core,
        merge_scores,
        delay_scores,
        policy_scores,
    )

    validation = validate_unified_outputs(
        core_dataframe=core,
        unified_dataframe=unified,
        merge_scores=merge_scores,
        delay_scores=delay_scores,
        policy_scores=policy_scores,
    )

    assert validation["validation_passed"] is True
