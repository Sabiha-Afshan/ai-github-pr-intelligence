"""Tests for contributor-neutral Random Forest tuning."""

import numpy as np
import pandas as pd

from src.models.merge_outcome_tuning import (
    build_feature_sets,
    build_parameter_candidates,
    get_selected_candidate,
    is_outlier_feature,
    tune_neutral_random_forest,
    validate_tuning_results,
)


def create_tuning_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Create reproducible tuning data."""

    generator = np.random.default_rng(42)

    train_rows = 180
    validation_rows = 70

    x_train = pd.DataFrame(
        {
            "additions": generator.gamma(
                2,
                25,
                train_rows,
            ),
            "log1p_additions": (
                generator.normal(
                    3,
                    0.7,
                    train_rows,
                )
            ),
            "changed_files": (
                generator.integers(
                    1,
                    15,
                    train_rows,
                )
            ),
            "log1p_changed_files": (
                generator.normal(
                    1.5,
                    0.5,
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
            "additions_iqr_outlier": (
                generator.integers(
                    0,
                    2,
                    train_rows,
                )
            ),
            "iqr_outlier_feature_count": (
                generator.integers(
                    0,
                    4,
                    train_rows,
                )
            ),
        }
    )

    train_score = (
        1.5 * x_train["has_test_changes"]
        + 0.8 * x_train["has_description"]
        - 0.08 * x_train["changed_files"]
        + generator.normal(
            0,
            0.7,
            train_rows,
        )
    )

    y_train = (train_score > 0.6).astype(int)

    x_validation = pd.DataFrame(
        {
            "additions": generator.gamma(
                2,
                25,
                validation_rows,
            ),
            "log1p_additions": (
                generator.normal(
                    3,
                    0.7,
                    validation_rows,
                )
            ),
            "changed_files": (
                generator.integers(
                    1,
                    15,
                    validation_rows,
                )
            ),
            "log1p_changed_files": (
                generator.normal(
                    1.5,
                    0.5,
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
            "additions_iqr_outlier": (
                generator.integers(
                    0,
                    2,
                    validation_rows,
                )
            ),
            "iqr_outlier_feature_count": (
                generator.integers(
                    0,
                    4,
                    validation_rows,
                )
            ),
        }
    )

    validation_score = (
        1.5 * x_validation["has_test_changes"]
        + 0.8 * x_validation["has_description"]
        - 0.08 * x_validation["changed_files"]
        + generator.normal(
            0,
            0.7,
            validation_rows,
        )
    )

    y_validation = (validation_score > 0.6).astype(int)

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
    )


def test_outlier_feature_detection() -> None:
    """Confirm IQR-derived features are detected."""

    assert is_outlier_feature("additions_iqr_outlier")

    assert is_outlier_feature("iqr_outlier_feature_count")

    assert not is_outlier_feature("additions")


def test_controlled_feature_sets() -> None:
    """Confirm feature-set variants are created."""

    features = [
        "additions",
        "log1p_additions",
        "has_test_changes",
        "additions_iqr_outlier",
    ]

    result = build_feature_sets(features)

    assert set(result) == {
        "full_approved",
        "core_raw",
        "core_log",
    }

    assert "additions_iqr_outlier" not in result["core_raw"]

    assert "log1p_additions" not in result["core_raw"]

    assert "additions" not in result["core_log"]


def test_parameter_candidates_exist() -> None:
    """Confirm the search is limited and reproducible."""

    candidates = build_parameter_candidates()

    assert len(candidates) == 8

    assert all("configuration_name" in candidate for candidate in candidates)


def test_complete_tuning_process() -> None:
    """Confirm tuning produces one selected candidate."""

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
    ) = create_tuning_data()

    results, comparison = tune_neutral_random_forest(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    assert len(results) == 24
    assert len(comparison) == 24

    assert int(comparison["selected_candidate"].sum()) == 1

    selected = get_selected_candidate(
        results=results,
        comparison=comparison,
    )

    assert 0 < selected.selected_threshold < 1


def test_tuning_validation() -> None:
    """Confirm tuning outputs pass validation."""

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
    ) = create_tuning_data()

    results, comparison = tune_neutral_random_forest(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    validation = validate_tuning_results(
        results=results,
        comparison=comparison,
        expected_validation_rows=(len(x_validation)),
    )

    assert validation["validation_passed"] is True
