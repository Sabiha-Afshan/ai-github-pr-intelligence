"""Tests for final Model 2 held-out evaluation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.merge_delay_final_evaluation import (
    build_classification_report_table,
    build_error_table,
    build_validation_test_comparison,
    evaluate_locked_model,
    load_final_test_split,
    validate_final_outputs,
    validate_locked_configuration,
)


def create_final_test_dataframe() -> pd.DataFrame:
    """Create a representative 38-row test dataset."""

    generator = np.random.default_rng(42)

    row_count = 38

    targets = np.array([0, 1] * 19)

    signal = targets + generator.normal(
        0,
        0.40,
        row_count,
    )

    return pd.DataFrame(
        {
            "pr_number": range(
                1001,
                1001 + row_count,
            ),
            "split": "test",
            "merge_delay_target": targets,
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


def create_training_dataframe() -> pd.DataFrame:
    """Create training data for a fitted test model."""

    generator = np.random.default_rng(7)

    row_count = 100

    targets = np.array([0, 1] * 50)

    signal = targets + generator.normal(
        0,
        0.45,
        row_count,
    )

    return pd.DataFrame(
        {
            "title_length": signal,
            "commit_count": (
                signal
                + generator.normal(
                    0,
                    0.25,
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
            "target": targets,
        }
    )


def create_configuration() -> dict:
    """Create a valid locked configuration."""

    return {
        "model_name": "logistic_candidate_test",
        "model_family": "logistic_regression",
        "model_path": "locked_model.joblib",
        "threshold": 0.75,
        "target": "merge_delay_target",
        "feature_count": 3,
        "features": [
            "title_length",
            "commit_count",
            "has_test_changes",
        ],
        "parameters": {
            "C": 10.0,
        },
        "validation_metrics": {
            "accuracy": 0.80,
            "balanced_accuracy": 0.80,
            "precision": 0.80,
            "recall": 0.80,
            "f1": 0.80,
            "roc_auc": 0.85,
            "average_precision": 0.82,
            "log_loss": 0.50,
            "brier_score": 0.16,
        },
        "test_used_for_selection": False,
    }


def test_locked_configuration_validation() -> None:
    """Confirm a complete locked configuration passes."""

    result = validate_locked_configuration(create_configuration())

    assert result["validation_passed"] is True


def test_final_test_split_loading() -> None:
    """Confirm the final test split loads correctly."""

    dataframe = create_final_test_dataframe()

    (
        identifiers,
        targets,
        features,
    ) = load_final_test_split(
        dataframe=dataframe,
        feature_names=[
            "title_length",
            "commit_count",
            "has_test_changes",
        ],
    )

    assert len(identifiers) == 38

    assert len(targets) == 38

    assert features.shape == (
        38,
        3,
    )


def test_locked_model_evaluation() -> None:
    """Confirm the locked model evaluates successfully."""

    training_data = create_training_dataframe()

    model = LogisticRegression(
        C=10.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )

    feature_names = [
        "title_length",
        "commit_count",
        "has_test_changes",
    ]

    model.fit(
        training_data[feature_names],
        training_data["target"],
    )

    (
        identifiers,
        targets,
        features,
    ) = load_final_test_split(
        dataframe=(create_final_test_dataframe()),
        feature_names=feature_names,
    )

    (
        metrics,
        predictions,
    ) = evaluate_locked_model(
        model=model,
        identifiers=identifiers,
        targets=targets,
        features=features,
        threshold=0.75,
        model_name="test_model",
    )

    assert len(predictions) == 38

    assert metrics["threshold"] == 0.75

    assert 0 <= metrics["roc_auc"] <= 1


def test_classification_and_error_tables() -> None:
    """Confirm classification and error outputs are created."""

    training_data = create_training_dataframe()

    feature_names = [
        "title_length",
        "commit_count",
        "has_test_changes",
    ]

    model = LogisticRegression(
        C=10.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        training_data[feature_names],
        training_data["target"],
    )

    (
        identifiers,
        targets,
        features,
    ) = load_final_test_split(
        create_final_test_dataframe(),
        feature_names,
    )

    (
        _,
        predictions,
    ) = evaluate_locked_model(
        model=model,
        identifiers=identifiers,
        targets=targets,
        features=features,
        threshold=0.75,
        model_name="test_model",
    )

    classification_table = build_classification_report_table(predictions)

    error_table = build_error_table(predictions)

    assert not classification_table.empty

    assert len(error_table) <= 38


def test_validation_test_comparison() -> None:
    """Confirm validation and test metrics are compared."""

    validation_metrics = create_configuration()["validation_metrics"]

    test_metrics = {
        "accuracy": 0.78,
        "balanced_accuracy": 0.76,
        "precision": 0.75,
        "recall": 0.73,
        "f1": 0.74,
        "roc_auc": 0.81,
        "average_precision": 0.77,
        "log_loss": 0.56,
        "brier_score": 0.18,
    }

    result = build_validation_test_comparison(
        validation_metrics,
        test_metrics,
    )

    assert len(result) == 9

    assert {
        "metric",
        "validation_value",
        "test_value",
        "test_minus_validation",
    }.issubset(result.columns)


def test_final_output_validation() -> None:
    """Confirm complete final outputs pass validation."""

    training_data = create_training_dataframe()

    feature_names = [
        "title_length",
        "commit_count",
        "has_test_changes",
    ]

    model = LogisticRegression(
        C=10.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        training_data[feature_names],
        training_data["target"],
    )

    (
        identifiers,
        targets,
        features,
    ) = load_final_test_split(
        create_final_test_dataframe(),
        feature_names,
    )

    (
        metrics,
        predictions,
    ) = evaluate_locked_model(
        model=model,
        identifiers=identifiers,
        targets=targets,
        features=features,
        threshold=0.75,
        model_name="test_model",
    )

    classification_table = build_classification_report_table(predictions)

    error_table = build_error_table(predictions)

    validation = validate_final_outputs(
        metrics=metrics,
        predictions=predictions,
        classification_table=(classification_table),
        error_table=error_table,
        expected_threshold=0.75,
    )

    assert validation["validation_passed"] is True
