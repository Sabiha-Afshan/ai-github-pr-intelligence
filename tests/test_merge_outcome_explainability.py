"""Tests for final merge-outcome explainability."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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


def create_explainability_data() -> tuple[
    RandomForestClassifier,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """Create a fitted model and prediction data."""

    features = pd.DataFrame(
        {
            "title_length": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
            ],
            "commit_count": [
                1,
                1,
                2,
                2,
                4,
                5,
                6,
                8,
            ],
            "has_test_changes": [
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )

    targets = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(
        features,
        targets,
    )

    probabilities = model.predict_proba(features)[:, 1]

    predicted = (probabilities >= 0.5).astype(int)

    predictions = pd.DataFrame(
        {
            "model_name": "test_model",
            "pr_number": range(
                101,
                109,
            ),
            "actual_target": targets,
            "predicted_target": predicted,
            "merge_probability": probabilities,
            "prediction_confidence": (
                np.maximum(
                    probabilities,
                    1 - probabilities,
                )
            ),
            "prediction_correct": (predicted == targets.to_numpy()),
            "selected_threshold": 0.5,
        }
    )

    identifiers = pd.Series(
        range(
            101,
            109,
        )
    )

    return (
        model,
        features,
        targets,
        identifiers,
        predictions,
    )


def test_feature_importance_tables() -> None:
    """Confirm both importance methods produce results."""

    (
        model,
        features,
        targets,
        _,
        _,
    ) = create_explainability_data()

    feature_names = list(features.columns)

    impurity = calculate_random_forest_importance(
        model=model,
        feature_names=feature_names,
    )

    permutation = calculate_test_permutation_importance(
        model=model,
        features=features,
        targets=targets,
        feature_names=feature_names,
        repeats=5,
        random_state=42,
    )

    combined = combine_importance_tables(
        impurity_importance=impurity,
        permutation_importance_table=(permutation),
    )

    assert len(impurity) == 3
    assert len(permutation) == 3
    assert len(combined) == 3


def test_calibration_and_confidence_tables() -> None:
    """Confirm probability summaries preserve all rows."""

    (
        _,
        _,
        _,
        _,
        predictions,
    ) = create_explainability_data()

    calibration = build_probability_calibration_table(
        predictions,
        bin_count=5,
    )

    confidence = build_confidence_band_table(predictions)

    assert int(calibration["prediction_count"].sum()) == len(predictions)

    assert int(confidence["prediction_count"].sum()) == len(predictions)


def test_error_analysis_outputs() -> None:
    """Confirm row-level error outputs are created."""

    (
        _,
        features,
        _,
        identifiers,
        predictions,
    ) = create_explainability_data()

    error_analysis = build_error_analysis_table(
        predictions=predictions,
        test_features=features,
        identifiers=identifiers,
    )

    error_summary = build_error_type_summary(error_analysis)

    error_profile = build_error_feature_profile(
        error_analysis=error_analysis,
        selected_features=list(features.columns),
    )

    assert len(error_analysis) == 8
    assert not error_summary.empty
    assert len(error_profile) == 3


def test_explainability_summary() -> None:
    """Confirm the final summary is calculated."""

    (
        model,
        features,
        targets,
        _,
        predictions,
    ) = create_explainability_data()

    impurity = calculate_random_forest_importance(
        model,
        list(features.columns),
    )

    permutation = calculate_test_permutation_importance(
        model=model,
        features=features,
        targets=targets,
        feature_names=list(features.columns),
        repeats=5,
    )

    combined = combine_importance_tables(
        impurity,
        permutation,
    )

    summary = calculate_explainability_summary(
        predictions,
        combined,
    )

    assert summary["row_count"] == 8

    assert 0 <= summary["roc_auc"] <= 1


def test_explainability_validation() -> None:
    """Confirm complete outputs pass validation."""

    (
        model,
        features,
        targets,
        identifiers,
        predictions,
    ) = create_explainability_data()

    impurity = calculate_random_forest_importance(
        model,
        list(features.columns),
    )

    permutation = calculate_test_permutation_importance(
        model=model,
        features=features,
        targets=targets,
        feature_names=list(features.columns),
        repeats=5,
    )

    combined = combine_importance_tables(
        impurity,
        permutation,
    )

    calibration = build_probability_calibration_table(
        predictions,
        bin_count=5,
    )

    confidence = build_confidence_band_table(predictions)

    error_analysis = build_error_analysis_table(
        predictions=predictions,
        test_features=features,
        identifiers=identifiers,
    )

    validation = validate_explainability_outputs(
        combined_importance=combined,
        calibration_table=calibration,
        confidence_table=confidence,
        error_analysis=error_analysis,
        expected_feature_count=3,
        expected_row_count=8,
    )

    assert validation["validation_passed"] is True
