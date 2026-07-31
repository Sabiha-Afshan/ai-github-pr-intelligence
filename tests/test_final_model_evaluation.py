"""Tests for held-out final model evaluation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.final_model_evaluation import (
    compare_validation_and_test,
    evaluate_locked_model,
    identify_high_confidence_errors,
    validate_final_test_results,
)


def create_fitted_model():
    """Create a small fitted binary classifier."""

    features = pd.DataFrame(
        {
            "feature_a": [
                0,
                0,
                1,
                1,
                2,
                2,
            ],
            "feature_b": [
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )

    targets = pd.Series([0, 0, 0, 1, 1, 1])

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
    )

    model.fit(
        features,
        targets,
    )

    return model


def test_locked_model_evaluation() -> None:
    """Confirm a locked model produces final outputs."""

    model = create_fitted_model()

    features = pd.DataFrame(
        {
            "feature_a": [
                0,
                1,
                2,
                2,
            ],
            "feature_b": [
                0,
                1,
                0,
                1,
            ],
        }
    )

    targets = pd.Series([0, 1, 1, 1])

    identifiers = pd.Series([101, 102, 103, 104])

    (
        metrics,
        predictions,
        classification_table,
    ) = evaluate_locked_model(
        model_name="test_model",
        model=model,
        features=features,
        targets=targets,
        identifiers=identifiers,
        selected_features=[
            "feature_a",
            "feature_b",
        ],
        selected_threshold=0.5,
    )

    assert len(predictions) == 4
    assert not classification_table.empty

    assert 0 <= metrics["roc_auc"] <= 1


def test_validation_test_comparison() -> None:
    """Confirm validation and test metrics are compared."""

    validation_metrics = {
        "accuracy": 0.70,
        "balanced_accuracy": 0.70,
        "precision": 0.70,
        "recall": 0.70,
        "f1": 0.70,
        "roc_auc": 0.75,
        "average_precision": 0.74,
        "log_loss": 0.60,
        "brier_score": 0.21,
    }

    test_metrics = {
        "accuracy": 0.68,
        "balanced_accuracy": 0.68,
        "precision": 0.69,
        "recall": 0.67,
        "f1": 0.68,
        "roc_auc": 0.72,
        "average_precision": 0.71,
        "log_loss": 0.63,
        "brier_score": 0.23,
    }

    result = compare_validation_and_test(
        validation_metrics,
        test_metrics,
    )

    assert len(result) == 9

    roc_row = result.loc[result["metric"] == "roc_auc"].iloc[0]

    assert np.isclose(
        roc_row["test_minus_validation"],
        -0.03,
    )


def test_high_confidence_errors() -> None:
    """Confirm high-confidence errors are extracted."""

    predictions = pd.DataFrame(
        {
            "prediction_correct": [
                True,
                False,
                False,
            ],
            "prediction_confidence": [
                0.90,
                0.80,
                0.60,
            ],
        }
    )

    result = identify_high_confidence_errors(
        predictions,
        minimum_confidence=0.75,
    )

    assert len(result) == 1


def test_final_result_validation() -> None:
    """Confirm valid final outputs pass."""

    metrics = {
        "accuracy": 0.70,
        "balanced_accuracy": 0.70,
        "precision": 0.70,
        "recall": 0.70,
        "f1": 0.70,
        "roc_auc": 0.75,
        "average_precision": 0.74,
        "true_negative": 20,
        "false_positive": 18,
        "false_negative": 5,
        "true_positive": 33,
    }

    predictions = pd.DataFrame(
        {
            "merge_probability": np.linspace(
                0.01,
                0.99,
                76,
            )
        }
    )

    result = validate_final_test_results(
        metrics=metrics,
        predictions=predictions,
        expected_rows=76,
    )

    assert result["validation_passed"] is True
