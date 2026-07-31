"""Tests for the Histogram Gradient Boosting challenger."""

import numpy as np
import pandas as pd

from src.models.merge_outcome_challenger import (
    CHALLENGER_MODEL_NAME,
    build_challenger_parameter_candidates,
    build_temporal_folds,
    compare_challenger_with_random_forest,
    fit_selected_challenger,
    get_selected_cv_candidate,
    run_challenger_cross_validation,
    validate_challenger_outputs,
)


def create_challenger_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Create reproducible chronological classification data."""

    generator = np.random.default_rng(42)

    train_rows = 240
    validation_rows = 80

    x_train = pd.DataFrame(
        {
            "title_length": (
                generator.integers(
                    10,
                    100,
                    train_rows,
                )
            ),
            "body_length": (
                generator.gamma(
                    2,
                    150,
                    train_rows,
                )
            ),
            "log1p_body_length": (
                generator.normal(
                    5,
                    0.8,
                    train_rows,
                )
            ),
            "commit_count": (
                generator.integers(
                    1,
                    15,
                    train_rows,
                )
            ),
            "log1p_commit_count": (
                generator.normal(
                    1.5,
                    0.6,
                    train_rows,
                )
            ),
            "has_test_changes": (
                generator.integers(
                    0,
                    2,
                    train_rows,
                )
            ),
            "has_description": (
                generator.integers(
                    0,
                    2,
                    train_rows,
                )
            ),
            "body_length_iqr_outlier": (
                generator.integers(
                    0,
                    2,
                    train_rows,
                )
            ),
        }
    )

    nonlinear_train_score = (
        1.5 * x_train["has_test_changes"]
        + 1.0 * x_train["has_description"]
        + (
            x_train["commit_count"]
            .between(
                2,
                8,
            )
            .astype(int)
        )
        - (x_train["title_length"] > 80).astype(int)
        + generator.normal(
            0,
            0.8,
            train_rows,
        )
    )

    y_train = (nonlinear_train_score > 1.2).astype(int)

    x_validation = pd.DataFrame(
        {
            "title_length": (
                generator.integers(
                    10,
                    100,
                    validation_rows,
                )
            ),
            "body_length": (
                generator.gamma(
                    2,
                    150,
                    validation_rows,
                )
            ),
            "log1p_body_length": (
                generator.normal(
                    5,
                    0.8,
                    validation_rows,
                )
            ),
            "commit_count": (
                generator.integers(
                    1,
                    15,
                    validation_rows,
                )
            ),
            "log1p_commit_count": (
                generator.normal(
                    1.5,
                    0.6,
                    validation_rows,
                )
            ),
            "has_test_changes": (
                generator.integers(
                    0,
                    2,
                    validation_rows,
                )
            ),
            "has_description": (
                generator.integers(
                    0,
                    2,
                    validation_rows,
                )
            ),
            "body_length_iqr_outlier": (
                generator.integers(
                    0,
                    2,
                    validation_rows,
                )
            ),
        }
    )

    nonlinear_validation_score = (
        1.5 * x_validation["has_test_changes"]
        + 1.0 * x_validation["has_description"]
        + (
            x_validation["commit_count"]
            .between(
                2,
                8,
            )
            .astype(int)
        )
        - (x_validation["title_length"] > 80).astype(int)
        + generator.normal(
            0,
            0.8,
            validation_rows,
        )
    )

    y_validation = (nonlinear_validation_score > 1.2).astype(int)

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
    )


def test_parameter_candidates_are_limited() -> None:
    """Confirm the challenger search remains controlled."""

    candidates = build_challenger_parameter_candidates()

    assert len(candidates) == 6

    assert all("configuration_name" in candidate for candidate in candidates)


def test_temporal_folds_expand() -> None:
    """Confirm temporal folds preserve ordering."""

    folds = build_temporal_folds(
        row_count=240,
        split_count=5,
    )

    assert len(folds) == 5

    for train_indices, validation_indices in folds:
        assert train_indices.max() < validation_indices.min()


def test_challenger_cross_validation() -> None:
    """Confirm the controlled challenger search runs."""

    (
        x_train,
        y_train,
        _,
        _,
    ) = create_challenger_data()

    results, comparison = run_challenger_cross_validation(
        x_train=x_train,
        y_train=y_train,
    )

    assert len(results) == 18
    assert len(comparison) == 18

    assert int(comparison["selected_candidate"].sum()) == 1


def test_selected_challenger_fit() -> None:
    """Confirm the selected challenger fits and predicts."""

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
    ) = create_challenger_data()

    results, comparison = run_challenger_cross_validation(
        x_train=x_train,
        y_train=y_train,
    )

    selected_candidate = get_selected_cv_candidate(
        results=results,
        comparison=comparison,
    )

    challenger = fit_selected_challenger(
        selected_candidate=(selected_candidate),
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    assert challenger.model_name == CHALLENGER_MODEL_NAME

    assert 0 < challenger.selected_threshold < 1

    assert len(challenger.validation_probabilities) == len(x_validation)


def test_model_selection_rule() -> None:
    """Confirm a meaningful challenger improvement is selected."""

    random_forest_metrics = {
        "roc_auc": 0.73,
        "balanced_accuracy": 0.71,
        "f1": 0.75,
    }

    challenger_metrics = {
        "roc_auc": 0.78,
        "balanced_accuracy": 0.74,
        "f1": 0.76,
    }

    comparison = compare_challenger_with_random_forest(
        challenger_metrics=(challenger_metrics),
        random_forest_metrics=(random_forest_metrics),
        minimum_roc_auc_improvement=0.02,
    )

    assert comparison["challenger_selected"] is True

    assert comparison["selected_final_model"] == CHALLENGER_MODEL_NAME


def test_challenger_output_validation() -> None:
    """Confirm complete challenger outputs pass validation."""

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
    ) = create_challenger_data()

    results, comparison = run_challenger_cross_validation(
        x_train=x_train,
        y_train=y_train,
    )

    selected_candidate = get_selected_cv_candidate(
        results=results,
        comparison=comparison,
    )

    challenger = fit_selected_challenger(
        selected_candidate=(selected_candidate),
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    validation = validate_challenger_outputs(
        cv_results=results,
        cv_comparison=comparison,
        challenger_result=challenger,
        expected_validation_rows=(len(x_validation)),
    )

    assert validation["validation_passed"] is True
