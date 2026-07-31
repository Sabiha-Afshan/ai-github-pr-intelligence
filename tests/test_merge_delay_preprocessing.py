"""Tests for Model 2 train-only preprocessing."""

import numpy as np
import pandas as pd

from src.models.merge_delay_preprocessing import (
    build_output_split,
    build_preprocessor,
    get_feature_names,
    identify_feature_types,
    normalize_split_values,
    normalize_target,
    prepare_model2_data,
    validate_source_dataset,
)


def create_model2_dataset() -> pd.DataFrame:
    """Create a representative approved Model 2 dataset."""

    generator = np.random.default_rng(42)

    row_count = 120

    split_values = ["train"] * 80 + ["validation"] * 20 + ["test"] * 20

    target_values = [0, 1] * 40 + [0, 1] * 10 + [0, 1] * 10

    dataframe = pd.DataFrame(
        {
            "pr_number": range(
                1,
                row_count + 1,
            ),
            "split": split_values,
            "merge_delay_target": (target_values),
            "has_test_changes": (
                generator.integers(
                    0,
                    2,
                    row_count,
                )
            ),
            "has_description": (
                generator.integers(
                    0,
                    2,
                    row_count,
                )
            ),
            "title_length": (
                generator.normal(
                    40,
                    10,
                    row_count,
                )
            ),
            "commit_count": (
                generator.integers(
                    1,
                    15,
                    row_count,
                ).astype(float)
            ),
        }
    )

    dataframe.loc[
        3,
        "title_length",
    ] = np.nan

    dataframe.loc[
        85,
        "commit_count",
    ] = np.nan

    dataframe.loc[
        105,
        "has_description",
    ] = np.nan

    return dataframe


def test_split_normalization() -> None:
    """Confirm split aliases normalize correctly."""

    result = normalize_split_values(
        pd.Series(
            [
                "training",
                "validation",
                "testing",
            ]
        )
    )

    assert result.tolist() == [
        "train",
        "validation",
        "test",
    ]


def test_target_normalization() -> None:
    """Confirm target values normalize correctly."""

    result = normalize_target(
        pd.Series(
            [
                0,
                1,
                0,
                1,
            ]
        )
    )

    assert result.tolist() == [
        0,
        1,
        0,
        1,
    ]


def test_source_validation() -> None:
    """Confirm a valid approved dataset passes."""

    result = validate_source_dataset(create_model2_dataset())

    assert result["validation_passed"] is True


def test_feature_name_selection() -> None:
    """Confirm metadata and target columns are excluded."""

    result = get_feature_names(create_model2_dataset())

    assert set(result) == {
        "has_test_changes",
        "has_description",
        "title_length",
        "commit_count",
    }


def test_feature_type_detection() -> None:
    """Confirm binary and continuous features are identified."""

    dataframe = create_model2_dataset()

    training_features = dataframe.loc[
        dataframe["split"] == "train",
        [
            "has_test_changes",
            "has_description",
            "title_length",
            "commit_count",
        ],
    ].reset_index(drop=True)

    result = identify_feature_types(training_features)

    assert set(result.binary_features) == {
        "has_test_changes",
        "has_description",
    }

    assert set(result.continuous_features) == {
        "title_length",
        "commit_count",
    }


def test_preprocessor_handles_missing_values() -> None:
    """Confirm preprocessing imputes and scales."""

    dataframe = create_model2_dataset()

    features = dataframe[
        [
            "has_test_changes",
            "has_description",
            "title_length",
            "commit_count",
        ]
    ]

    feature_types = identify_feature_types(
        features.loc[dataframe["split"] == "train"].reset_index(drop=True)
    )

    preprocessor = build_preprocessor(
        binary_features=(feature_types.binary_features),
        continuous_features=(feature_types.continuous_features),
    )

    train_features = features.loc[dataframe["split"] == "train"]

    transformed = preprocessor.fit_transform(train_features)

    transformed_array = np.asarray(
        transformed,
        dtype=float,
    )

    assert not np.isnan(transformed_array).any()

    assert np.isfinite(transformed_array).all()


def test_complete_model2_preparation() -> None:
    """Confirm all splits are prepared correctly."""

    prepared = prepare_model2_data(create_model2_dataset())

    assert len(prepared.x_train) == 80

    assert len(prepared.x_validation) == 20

    assert len(prepared.x_test) == 20

    assert list(prepared.x_train.columns) == list(prepared.x_validation.columns)

    assert list(prepared.x_train.columns) == list(prepared.x_test.columns)

    assert not prepared.x_train.isna().any().any()
    assert not prepared.x_validation.isna().any().any()
    assert not prepared.x_test.isna().any().any()


def test_output_split_creation() -> None:
    """Confirm transformed split metadata is restored."""

    prepared = prepare_model2_data(create_model2_dataset())

    output = build_output_split(
        identifiers=(prepared.train_identifiers),
        targets=prepared.y_train,
        features=prepared.x_train,
        split_name="train",
    )

    assert output.columns[0] == "pr_number"

    assert output.columns[1] == "split"

    assert output.columns[2] == "merge_delay_target"

    assert set(output["split"]) == {"train"}


def test_prepared_outputs_are_valid() -> None:
    """Confirm prepared outputs contain finite values and both classes."""

    prepared = prepare_model2_data(create_model2_dataset())

    for features in (
        prepared.x_train,
        prepared.x_validation,
        prepared.x_test,
    ):
        values = features.to_numpy(dtype=float)

        assert np.isfinite(values).all()

    for targets in (
        prepared.y_train,
        prepared.y_validation,
        prepared.y_test,
    ):
        assert set(targets.unique()) == {
            0,
            1,
        }
