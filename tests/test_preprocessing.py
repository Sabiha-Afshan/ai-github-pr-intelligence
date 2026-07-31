"""Tests for model preprocessing."""

import numpy as np
import pandas as pd

from src.models.preprocessing import (
    EXPECTED_SPLIT_COUNTS,
    FeatureTypeGroups,
    build_preprocessing_pipeline,
    create_chronological_splits,
    get_model_feature_columns,
    identify_feature_types,
    is_binary_training_feature,
    transform_feature_dataframe,
)


def create_full_split_dataframe() -> pd.DataFrame:
    """Create a dataset with production split counts."""

    records = []

    pr_number = 1

    for split_name, row_count in EXPECTED_SPLIT_COUNTS.items():
        for row_index in range(row_count):
            records.append(
                {
                    "pr_number": pr_number,
                    "was_merged": (row_index % 2),
                    "split": split_name,
                    "binary_feature": (row_index % 2),
                    "continuous_feature": (float(row_index)),
                }
            )

            pr_number += 1

    return pd.DataFrame(records)


def test_get_model_feature_columns() -> None:
    """Confirm metadata columns are excluded."""

    dataframe = create_full_split_dataframe()

    result = get_model_feature_columns(dataframe)

    assert result == [
        "binary_feature",
        "continuous_feature",
    ]


def test_create_chronological_splits() -> None:
    """Confirm chronological splits retain expected counts."""

    dataframe = create_full_split_dataframe()

    splits = create_chronological_splits(dataframe)

    assert len(splits.train) == 440

    assert len(splits.validation) == 84

    assert len(splits.test) == 76


def test_binary_feature_detection() -> None:
    """Confirm binary features are recognized."""

    binary_series = pd.Series(
        [
            0,
            1,
            1,
            np.nan,
        ]
    )

    continuous_series = pd.Series(
        [
            0,
            1,
            2,
            3,
        ]
    )

    assert is_binary_training_feature(binary_series) is True

    assert is_binary_training_feature(continuous_series) is False


def test_identify_feature_types() -> None:
    """Confirm feature types use training data."""

    dataframe = pd.DataFrame(
        {
            "binary_feature": [
                0,
                1,
                0,
                1,
            ],
            "continuous_feature": [
                10,
                20,
                30,
                40,
            ],
        }
    )

    result = identify_feature_types(
        training_dataframe=dataframe,
        feature_columns=[
            "binary_feature",
            "continuous_feature",
        ],
    )

    assert result.binary_features == ["binary_feature"]

    assert result.continuous_features == ["continuous_feature"]


def test_pipeline_imputes_and_scales() -> None:
    """Confirm missing values are removed after transformation."""

    training_dataframe = pd.DataFrame(
        {
            "continuous_feature": [
                1.0,
                2.0,
                np.nan,
                4.0,
            ],
            "binary_feature": [
                0.0,
                1.0,
                np.nan,
                1.0,
            ],
        }
    )

    groups = FeatureTypeGroups(
        binary_features=["binary_feature"],
        continuous_features=["continuous_feature"],
    )

    preprocessor = build_preprocessing_pipeline(groups)

    preprocessor.fit(training_dataframe)

    transformed = transform_feature_dataframe(
        preprocessor=preprocessor,
        dataframe=(training_dataframe),
        feature_columns=[
            "continuous_feature",
            "binary_feature",
        ],
    )

    assert transformed.shape == (
        4,
        2,
    )

    assert int(transformed.isna().sum().sum()) == 0

    assert int(np.isinf(transformed.to_numpy()).sum()) == 0


def test_validation_is_not_used_for_fit() -> None:
    """Confirm fitted statistics come only from training."""

    training_dataframe = pd.DataFrame(
        {
            "continuous_feature": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    validation_dataframe = pd.DataFrame(
        {
            "continuous_feature": [
                1000.0,
            ],
        }
    )

    groups = FeatureTypeGroups(
        binary_features=[],
        continuous_features=["continuous_feature"],
    )

    preprocessor = build_preprocessing_pipeline(groups)

    preprocessor.fit(training_dataframe)

    continuous_pipeline = preprocessor.named_transformers_["continuous"]

    scaler = continuous_pipeline.named_steps["standard_scaler"]

    assert float(scaler.mean_[0]) == 2.0

    transformed_validation = transform_feature_dataframe(
        preprocessor=preprocessor,
        dataframe=validation_dataframe,
        feature_columns=["continuous_feature"],
    )

    assert (
        transformed_validation.iloc[
            0,
            0,
        ]
        > 100
    )
