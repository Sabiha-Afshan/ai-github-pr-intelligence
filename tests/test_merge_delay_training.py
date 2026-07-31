"""Tests for Model 2 baseline training."""

import numpy as np
import pandas as pd

from src.models.merge_delay_evaluation import (
    build_prediction_table,
    build_threshold_table,
    calculate_binary_metrics,
    compare_validation_models,
    select_validation_threshold,
)
from src.models.merge_delay_training import (
    load_preprocessed_split,
    train_logistic_regression,
    train_random_forest,
    validate_split_compatibility,
)


def create_preprocessed_split(
    split_name: str,
    row_count: int,
    starting_pr_number: int,
) -> pd.DataFrame:
    """Create a representative preprocessed split."""

    generator = np.random.default_rng(starting_pr_number)

    target = np.array(
        [
            0,
            1,
        ]
        * (row_count // 2)
    )

    if len(target) < row_count:
        target = np.append(
            target,
            0,
        )

    signal = target + generator.normal(
        0,
        0.45,
        row_count,
    )

    return pd.DataFrame(
        {
            "pr_number": range(
                starting_pr_number,
                starting_pr_number + row_count,
            ),
            "split": split_name,
            "merge_delay_target": target,
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
                target
                ^ generator.integers(
                    0,
                    2,
                    row_count,
                )
            ),
            "deletion_ratio": (
                generator.normal(
                    0,
                    1,
                    row_count,
                )
            ),
        }
    )


def test_binary_metrics() -> None:
    """Confirm binary metrics are calculated."""

    actual = pd.Series(
        [
            0,
            0,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.4,
            0.6,
            0.9,
        ]
    )

    metrics = calculate_binary_metrics(
        actual=actual,
        probabilities=probabilities,
        threshold=0.5,
    )

    assert metrics["accuracy"] == 1.0

    assert metrics["true_positive"] == 2

    assert metrics["true_negative"] == 2


def test_threshold_selection() -> None:
    """Confirm a valid threshold is selected."""

    actual = pd.Series(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
            0.9,
        ]
    )

    threshold_table = build_threshold_table(
        actual=actual,
        probabilities=probabilities,
        minimum_threshold=0.30,
        maximum_threshold=0.70,
        threshold_step=0.10,
    )

    selected = select_validation_threshold(threshold_table)

    assert 0.30 <= selected["threshold"] <= 0.70


def test_prediction_table() -> None:
    """Confirm row-level prediction output is created."""

    identifiers = pd.Series(
        [
            101,
            102,
            103,
        ]
    )

    actual = pd.Series(
        [
            0,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.2,
            0.7,
            0.4,
        ]
    )

    result = build_prediction_table(
        identifiers=identifiers,
        actual=actual,
        probabilities=probabilities,
        threshold=0.5,
        model_name="test_model",
        split_name="validation",
    )

    assert len(result) == 3

    assert {
        "true_negative",
        "true_positive",
        "false_negative",
    }.issubset(set(result["prediction_outcome"]))


def test_split_loading_and_compatibility() -> None:
    """Confirm preprocessed splits are compatible."""

    train_dataframe = create_preprocessed_split(
        split_name="train",
        row_count=80,
        starting_pr_number=1,
    )

    validation_dataframe = create_preprocessed_split(
        split_name="validation",
        row_count=20,
        starting_pr_number=1001,
    )

    training_split = load_preprocessed_split(
        train_dataframe,
        expected_split="train",
    )

    validation_split = load_preprocessed_split(
        validation_dataframe,
        expected_split="validation",
    )

    validation = validate_split_compatibility(
        training_split,
        validation_split,
    )

    assert validation["validation_passed"] is True


def test_logistic_regression_training() -> None:
    """Confirm Logistic Regression trains successfully."""

    training_split = load_preprocessed_split(
        create_preprocessed_split(
            split_name="train",
            row_count=100,
            starting_pr_number=1,
        ),
        expected_split="train",
    )

    validation_split = load_preprocessed_split(
        create_preprocessed_split(
            split_name="validation",
            row_count=30,
            starting_pr_number=1001,
        ),
        expected_split="validation",
    )

    result = train_logistic_regression(
        training_split=training_split,
        validation_split=validation_split,
    )

    assert result.model_name == ("merge_delay_logistic_regression")

    assert 0 <= result.validation_metrics["roc_auc"] <= 1


def test_random_forest_training() -> None:
    """Confirm Random Forest trains successfully."""

    training_split = load_preprocessed_split(
        create_preprocessed_split(
            split_name="train",
            row_count=100,
            starting_pr_number=1,
        ),
        expected_split="train",
    )

    validation_split = load_preprocessed_split(
        create_preprocessed_split(
            split_name="validation",
            row_count=30,
            starting_pr_number=1001,
        ),
        expected_split="validation",
    )

    result = train_random_forest(
        training_split=training_split,
        validation_split=validation_split,
    )

    assert result.model_name == ("merge_delay_random_forest")

    assert len(result.validation_predictions) == 30


def test_model_comparison() -> None:
    """Confirm validation models are ranked."""

    comparison = compare_validation_models(
        [
            {
                "model_name": "model_a",
                "balanced_accuracy": 0.70,
                "f1": 0.68,
                "roc_auc": 0.75,
                "average_precision": 0.72,
            },
            {
                "model_name": "model_b",
                "balanced_accuracy": 0.75,
                "f1": 0.70,
                "roc_auc": 0.78,
                "average_precision": 0.76,
            },
        ]
    )

    assert comparison.iloc[0]["model_name"] == "model_b"

    assert bool(comparison.iloc[0]["nominated_candidate"])
