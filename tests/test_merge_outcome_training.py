"""Tests for corrected merge-outcome training."""

import numpy as np
import pandas as pd

from src.models.merge_outcome_evaluation import (
    build_threshold_table,
    calculate_metrics,
    select_validation_threshold,
)
from src.models.merge_outcome_training import (
    AUTHOR_RANDOM_FOREST_MODEL_NAME,
    NEUTRAL_LOGISTIC_MODEL_NAME,
    NEUTRAL_RANDOM_FOREST_MODEL_NAME,
    build_model_comparison,
    train_merge_outcome_models,
    validate_training_results,
)


def create_training_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create reproducible classification data."""

    generator = np.random.default_rng(42)

    x_train = pd.DataFrame(
        {
            "total_changes": (
                generator.normal(
                    100,
                    25,
                    160,
                )
            ),
            "changed_files": (
                generator.integers(
                    1,
                    12,
                    160,
                )
            ),
            "has_test_changes": (
                generator.integers(
                    0,
                    2,
                    160,
                )
            ),
        }
    )

    y_train = (
        (
            x_train["has_test_changes"]
            + (x_train["changed_files"] < 7).astype(int)
            + generator.normal(
                0,
                0.5,
                160,
            )
        )
        > 1
    ).astype(int)

    x_validation = pd.DataFrame(
        {
            "total_changes": (
                generator.normal(
                    100,
                    25,
                    60,
                )
            ),
            "changed_files": (
                generator.integers(
                    1,
                    12,
                    60,
                )
            ),
            "has_test_changes": (
                generator.integers(
                    0,
                    2,
                    60,
                )
            ),
        }
    )

    y_validation = (
        (
            x_validation["has_test_changes"]
            + (x_validation["changed_files"] < 7).astype(int)
            + generator.normal(
                0,
                0.5,
                60,
            )
        )
        > 1
    ).astype(int)

    validation_ids = pd.Series(
        range(
            1001,
            1061,
        )
    )

    train_author = x_train.copy()
    train_author["author_association"] = np.where(
        y_train == 1,
        "MEMBER",
        "CONTRIBUTOR",
    )

    validation_author = x_validation.copy()

    validation_author["author_association"] = np.where(
        y_validation == 1,
        "MEMBER",
        "CONTRIBUTOR",
    )

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        validation_ids,
        train_author,
        validation_author,
    )


def test_metric_calculation() -> None:
    """Confirm classification metrics are calculated."""

    targets = pd.Series([0, 0, 1, 1])

    probabilities = np.array([0.1, 0.4, 0.7, 0.9])

    metrics = calculate_metrics(
        targets=targets,
        probabilities=probabilities,
        threshold=0.5,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_threshold_selection() -> None:
    """Confirm a validation threshold is selected."""

    targets = pd.Series([0, 0, 1, 1])

    probabilities = np.array([0.1, 0.45, 0.55, 0.9])

    threshold_table = build_threshold_table(
        targets=targets,
        probabilities=probabilities,
    )

    selected = select_validation_threshold(threshold_table)

    assert 0 < selected["threshold"] < 1

    assert 0 <= selected["f1"] <= 1


def test_all_required_models_train() -> None:
    """Confirm the three required models train."""

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        validation_ids,
        train_author,
        validation_author,
    ) = create_training_data()

    results = train_merge_outcome_models(
        neutral_x_train=x_train,
        neutral_y_train=y_train,
        neutral_x_validation=(x_validation),
        neutral_y_validation=(y_validation),
        validation_identifiers=(validation_ids),
        author_x_train=train_author,
        author_x_validation=(validation_author),
    )

    assert set(results) == {
        NEUTRAL_LOGISTIC_MODEL_NAME,
        NEUTRAL_RANDOM_FOREST_MODEL_NAME,
        AUTHOR_RANDOM_FOREST_MODEL_NAME,
    }

    for result in results.values():
        assert 0 < result.selected_threshold < 1

        assert len(result.predictions) == 60


def test_model_comparison_roles() -> None:
    """Confirm the neutral forest is the application candidate."""

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        validation_ids,
        train_author,
        validation_author,
    ) = create_training_data()

    results = train_merge_outcome_models(
        neutral_x_train=x_train,
        neutral_y_train=y_train,
        neutral_x_validation=(x_validation),
        neutral_y_validation=(y_validation),
        validation_identifiers=(validation_ids),
        author_x_train=train_author,
        author_x_validation=(validation_author),
    )

    comparison = build_model_comparison(results)

    eligible = comparison.loc[
        comparison["eligible_for_final_application"],
        "model_name",
    ].tolist()

    assert eligible == [NEUTRAL_RANDOM_FOREST_MODEL_NAME]


def test_training_result_validation() -> None:
    """Confirm complete results pass validation."""

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        validation_ids,
        train_author,
        validation_author,
    ) = create_training_data()

    results = train_merge_outcome_models(
        neutral_x_train=x_train,
        neutral_y_train=y_train,
        neutral_x_validation=(x_validation),
        neutral_y_validation=(y_validation),
        validation_identifiers=(validation_ids),
        author_x_train=train_author,
        author_x_validation=(validation_author),
    )

    validation = validate_training_results(
        results=results,
        expected_validation_rows=60,
    )

    assert validation["validation_passed"] is True
