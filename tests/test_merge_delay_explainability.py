"""Tests for Model 2 explainability and error analysis."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.merge_delay_explainability import (
    build_error_analysis,
    build_probability_band_summary,
    calculate_local_logit_contributions,
    calculate_logistic_coefficient_importance,
    calculate_test_permutation_importance,
    combine_explainability_tables,
    validate_explainability_outputs,
)


def create_explainability_data() -> tuple[
    LogisticRegression,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """Create a fitted model and representative predictions."""

    generator = np.random.default_rng(42)

    row_count = 40

    targets = pd.Series([0, 1] * 20)

    signal = targets.to_numpy() + generator.normal(
        0,
        0.45,
        row_count,
    )

    features = pd.DataFrame(
        {
            "title_length": signal,
            "commit_count": (
                signal
                + generator.normal(
                    0,
                    0.20,
                    row_count,
                )
            ),
            "has_test_changes": (
                generator.integers(
                    0,
                    2,
                    row_count,
                )
            ),
        }
    )

    model = LogisticRegression(
        C=10.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        features,
        targets,
    )

    probabilities = model.predict_proba(features)[:, 1]

    threshold = 0.75

    predicted = (probabilities >= threshold).astype(int)

    confidence = np.where(
        predicted == 1,
        probabilities,
        1 - probabilities,
    )

    outcomes = np.select(
        [
            (targets == 1) & (predicted == 1),
            (targets == 0) & (predicted == 0),
            (targets == 0) & (predicted == 1),
        ],
        [
            "true_positive",
            "true_negative",
            "false_positive",
        ],
        default="false_negative",
    )

    predictions = pd.DataFrame(
        {
            "pr_number": range(
                1,
                row_count + 1,
            ),
            "actual_target": targets,
            "predicted_target": predicted,
            "delay_probability": probabilities,
            "prediction_confidence": confidence,
            "prediction_correct": (predicted == targets.to_numpy()),
            "prediction_outcome": outcomes,
            "selected_threshold": threshold,
        }
    )

    identifiers = pd.Series(
        range(
            1,
            row_count + 1,
        )
    )

    return (
        model,
        features,
        targets,
        identifiers,
        predictions,
    )


def test_coefficient_importance() -> None:
    """Confirm coefficient importance is calculated."""

    (
        model,
        features,
        _,
        _,
        _,
    ) = create_explainability_data()

    result = calculate_logistic_coefficient_importance(
        model=model,
        feature_names=list(features.columns),
    )

    assert len(result) == 3

    assert {
        "coefficient",
        "odds_ratio",
        "effect_direction",
    }.issubset(result.columns)


def test_permutation_importance() -> None:
    """Confirm permutation importance is calculated."""

    (
        model,
        features,
        targets,
        _,
        _,
    ) = create_explainability_data()

    result = calculate_test_permutation_importance(
        model=model,
        features=features,
        targets=targets,
        feature_names=list(features.columns),
        repeats=5,
        random_state=42,
    )

    assert len(result) == 3


def test_combined_explainability() -> None:
    """Confirm explainability tables combine correctly."""

    (
        model,
        features,
        targets,
        _,
        _,
    ) = create_explainability_data()

    coefficient_table = calculate_logistic_coefficient_importance(
        model,
        list(features.columns),
    )

    permutation_table = calculate_test_permutation_importance(
        model=model,
        features=features,
        targets=targets,
        feature_names=list(features.columns),
        repeats=5,
    )

    combined = combine_explainability_tables(
        coefficient_table,
        permutation_table,
    )

    assert len(combined) == 3


def test_local_contributions() -> None:
    """Confirm local logit contributions are generated."""

    (
        model,
        features,
        targets,
        identifiers,
        predictions,
    ) = create_explainability_data()

    result = calculate_local_logit_contributions(
        model=model,
        features=features,
        identifiers=identifiers,
        actual_targets=targets,
        predicted_targets=(predictions["predicted_target"]),
        probabilities=(predictions["delay_probability"]),
        top_feature_count=2,
    )

    assert len(result) == 80

    assert {
        "feature",
        "logit_contribution",
        "contribution_direction",
    }.issubset(result.columns)


def test_error_analysis() -> None:
    """Confirm prediction errors are analysed."""

    (
        model,
        features,
        targets,
        identifiers,
        predictions,
    ) = create_explainability_data()

    local_contributions = calculate_local_logit_contributions(
        model=model,
        features=features,
        identifiers=identifiers,
        actual_targets=targets,
        predicted_targets=(predictions["predicted_target"]),
        probabilities=(predictions["delay_probability"]),
        top_feature_count=2,
    )

    (
        errors,
        feature_summary,
    ) = build_error_analysis(
        predictions=predictions,
        local_contributions=(local_contributions),
    )

    assert len(errors) <= 40
    assert not feature_summary.empty or errors.empty


def test_probability_band_summary() -> None:
    """Confirm probability bands preserve all rows."""

    (
        _,
        _,
        _,
        _,
        predictions,
    ) = create_explainability_data()

    result = build_probability_band_summary(predictions)

    assert int(result["prediction_count"].sum()) == 40


def test_complete_output_validation() -> None:
    """Confirm complete explainability outputs pass."""

    (
        model,
        features,
        targets,
        identifiers,
        predictions,
    ) = create_explainability_data()

    coefficient_table = calculate_logistic_coefficient_importance(
        model,
        list(features.columns),
    )

    permutation_table = calculate_test_permutation_importance(
        model=model,
        features=features,
        targets=targets,
        feature_names=list(features.columns),
        repeats=5,
    )

    combined_table = combine_explainability_tables(
        coefficient_table,
        permutation_table,
    )

    local_contributions = calculate_local_logit_contributions(
        model=model,
        features=features,
        identifiers=identifiers,
        actual_targets=targets,
        predicted_targets=(predictions["predicted_target"]),
        probabilities=(predictions["delay_probability"]),
        top_feature_count=2,
    )

    (
        errors,
        _,
    ) = build_error_analysis(
        predictions,
        local_contributions,
    )

    probability_bands = build_probability_band_summary(predictions)

    validation = validate_explainability_outputs(
        coefficient_table=(coefficient_table),
        permutation_table=(permutation_table),
        combined_table=(combined_table),
        local_contributions=(local_contributions),
        errors=errors,
        probability_bands=(probability_bands),
        expected_feature_count=3,
        expected_test_rows=40,
        top_feature_count=2,
    )

    assert validation["validation_passed"] is True
