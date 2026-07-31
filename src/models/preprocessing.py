"""Preprocessing utilities for merge-outcome modelling."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.feature_schema import (
    PR_IDENTIFIER_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)


EXPECTED_SPLIT_COUNTS = {
    "train": 440,
    "validation": 84,
    "test": 76,
}


@dataclass(frozen=True)
class ModelDataSplits:
    """Container for chronological model datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FeatureTypeGroups:
    """Container for binary and continuous feature names."""

    binary_features: list[str]
    continuous_features: list[str]


@dataclass(frozen=True)
class TransformedModelSplits:
    """Container for transformed feature matrices and targets."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    train_identifiers: pd.Series
    validation_identifiers: pd.Series
    test_identifiers: pd.Series


def get_model_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return feature columns from the approved modelling dataset."""

    excluded_columns = {
        PR_IDENTIFIER_COLUMN,
        TARGET_COLUMN,
        SPLIT_COLUMN,
    }

    return [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]


def validate_approved_modelling_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the approved dataset before preprocessing."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        TARGET_COLUMN,
        SPLIT_COLUMN,
    }

    missing_required_columns = sorted(
        required_columns - set(dataframe.columns)
    )

    feature_columns = get_model_feature_columns(
        dataframe
    )

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]

    duplicate_pr_count = (
        int(
            dataframe.duplicated(
                subset=[PR_IDENTIFIER_COLUMN]
            ).sum()
        )
        if PR_IDENTIFIER_COLUMN in dataframe.columns
        else None
    )

    infinite_value_count = 0

    if feature_columns:
        numeric_values = (
            dataframe[feature_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(
                dtype=float
            )
        )

        infinite_value_count = int(
            np.isinf(
                numeric_values
            ).sum()
        )

    validation_passed = (
        len(dataframe) == 600
        and not missing_required_columns
        and len(feature_columns) == 61
        and not non_numeric_features
        and duplicate_pr_count == 0
        and infinite_value_count == 0
    )

    return {
        "row_count": len(
            dataframe
        ),
        "column_count": len(
            dataframe.columns
        ),
        "feature_count": len(
            feature_columns
        ),
        "missing_required_columns": (
            missing_required_columns
        ),
        "non_numeric_features": (
            non_numeric_features
        ),
        "duplicate_pr_count": (
            duplicate_pr_count
        ),
        "infinite_value_count": (
            infinite_value_count
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def create_chronological_splits(
    dataframe: pd.DataFrame,
) -> ModelDataSplits:
    """Separate the approved dataset by chronological split."""

    if SPLIT_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Dataset is missing {SPLIT_COLUMN!r}."
        )

    normalized_splits = (
        dataframe[SPLIT_COLUMN]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    split_counts = (
        normalized_splits.value_counts()
        .to_dict()
    )

    for split_name, expected_count in (
        EXPECTED_SPLIT_COUNTS.items()
    ):
        actual_count = int(
            split_counts.get(
                split_name,
                0,
            )
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Expected {expected_count} rows for "
                f"{split_name!r}, but found "
                f"{actual_count}."
            )

    train = (
        dataframe.loc[
            normalized_splits == "train"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    validation = (
        dataframe.loc[
            normalized_splits
            == "validation"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test = (
        dataframe.loc[
            normalized_splits == "test"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return ModelDataSplits(
        train=train,
        validation=validation,
        test=test,
    )


def is_binary_training_feature(
    series: pd.Series,
) -> bool:
    """Determine whether a training feature is binary."""

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    unique_values = set(
        numeric_series.dropna().unique()
    )

    return (
        len(unique_values) > 0
        and unique_values.issubset(
            {
                0,
                1,
                0.0,
                1.0,
            }
        )
    )


def identify_feature_types(
    training_dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> FeatureTypeGroups:
    """Identify binary and continuous features from training data only."""

    missing_features = sorted(
        set(feature_columns)
        - set(training_dataframe.columns)
    )

    if missing_features:
        raise ValueError(
            "Training dataset is missing features: "
            f"{missing_features}"
        )

    binary_features = [
        feature
        for feature in feature_columns
        if is_binary_training_feature(
            training_dataframe[feature]
        )
    ]

    continuous_features = [
        feature
        for feature in feature_columns
        if feature not in binary_features
    ]

    return FeatureTypeGroups(
        binary_features=binary_features,
        continuous_features=continuous_features,
    )


def build_preprocessing_pipeline(
    feature_groups: FeatureTypeGroups,
) -> ColumnTransformer:
    """Build the preprocessing transformer."""

    transformers = []

    if feature_groups.continuous_features:
        continuous_pipeline = Pipeline(
            steps=[
                (
                    "median_imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "standard_scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "continuous",
                continuous_pipeline,
                feature_groups.continuous_features,
            )
        )

    if feature_groups.binary_features:
        binary_pipeline = Pipeline(
            steps=[
                (
                    "most_frequent_imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "binary",
                binary_pipeline,
                feature_groups.binary_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No model features were supplied "
            "to the preprocessing pipeline."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """Return output feature names from a fitted transformer."""

    try:
        feature_names = (
            preprocessor.get_feature_names_out()
        )
    except AttributeError as error:
        raise RuntimeError(
            "The preprocessing pipeline must be "
            "fitted before feature names are requested."
        ) from error

    return [
        str(feature_name)
        for feature_name in feature_names
    ]


def transform_feature_dataframe(
    preprocessor: ColumnTransformer,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transform one dataset using a fitted preprocessor."""

    missing_features = sorted(
        set(feature_columns)
        - set(dataframe.columns)
    )

    if missing_features:
        raise ValueError(
            "Dataset is missing model features: "
            f"{missing_features}"
        )

    transformed_values = (
        preprocessor.transform(
            dataframe[feature_columns]
        )
    )

    transformed_feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )

    return pd.DataFrame(
        transformed_values,
        columns=transformed_feature_names,
        index=dataframe.index,
    )


def fit_and_transform_model_splits(
    splits: ModelDataSplits,
    feature_columns: list[str],
) -> tuple[
    ColumnTransformer,
    FeatureTypeGroups,
    TransformedModelSplits,
]:
    """Fit preprocessing on train and transform every split."""

    feature_groups = identify_feature_types(
        training_dataframe=splits.train,
        feature_columns=feature_columns,
    )

    preprocessor = build_preprocessing_pipeline(
        feature_groups
    )

    preprocessor.fit(
        splits.train[feature_columns]
    )

    x_train = transform_feature_dataframe(
        preprocessor=preprocessor,
        dataframe=splits.train,
        feature_columns=feature_columns,
    )

    x_validation = transform_feature_dataframe(
        preprocessor=preprocessor,
        dataframe=splits.validation,
        feature_columns=feature_columns,
    )

    x_test = transform_feature_dataframe(
        preprocessor=preprocessor,
        dataframe=splits.test,
        feature_columns=feature_columns,
    )

    transformed_splits = TransformedModelSplits(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=pd.to_numeric(
            splits.train[
                TARGET_COLUMN
            ],
            errors="raise",
        ).astype(int),
        y_validation=pd.to_numeric(
            splits.validation[
                TARGET_COLUMN
            ],
            errors="raise",
        ).astype(int),
        y_test=pd.to_numeric(
            splits.test[
                TARGET_COLUMN
            ],
            errors="raise",
        ).astype(int),
        train_identifiers=splits.train[
            PR_IDENTIFIER_COLUMN
        ].copy(),
        validation_identifiers=(
            splits.validation[
                PR_IDENTIFIER_COLUMN
            ].copy()
        ),
        test_identifiers=splits.test[
            PR_IDENTIFIER_COLUMN
        ].copy(),
    )

    return (
        preprocessor,
        feature_groups,
        transformed_splits,
    )


def validate_transformed_splits(
    transformed: TransformedModelSplits,
    expected_feature_count: int,
) -> dict[str, Any]:
    """Validate transformed model datasets."""

    matrices = {
        "train": transformed.x_train,
        "validation": transformed.x_validation,
        "test": transformed.x_test,
    }

    expected_rows = EXPECTED_SPLIT_COUNTS

    split_results = {}

    for split_name, matrix in matrices.items():
        numeric_values = matrix.to_numpy(
            dtype=float
        )

        missing_count = int(
            np.isnan(
                numeric_values
            ).sum()
        )

        infinite_count = int(
            np.isinf(
                numeric_values
            ).sum()
        )

        split_results[
            split_name
        ] = {
            "row_count": len(
                matrix
            ),
            "feature_count": len(
                matrix.columns
            ),
            "missing_value_count": (
                missing_count
            ),
            "infinite_value_count": (
                infinite_count
            ),
            "validation_passed": (
                len(matrix)
                == expected_rows[
                    split_name
                ]
                and len(
                    matrix.columns
                )
                == expected_feature_count
                and missing_count == 0
                and infinite_count == 0
            ),
        }

    consistent_columns = (
        list(
            transformed.x_train.columns
        )
        == list(
            transformed.x_validation.columns
        )
        == list(
            transformed.x_test.columns
        )
    )

    target_results = {
        "train_target_count": len(
            transformed.y_train
        ),
        "validation_target_count": len(
            transformed.y_validation
        ),
        "test_target_count": len(
            transformed.y_test
        ),
        "train_target_values": sorted(
            transformed.y_train.unique().tolist()
        ),
        "validation_target_values": sorted(
            transformed.y_validation.unique().tolist()
        ),
        "test_target_values": sorted(
            transformed.y_test.unique().tolist()
        ),
    }

    targets_valid = (
        target_results[
            "train_target_count"
        ]
        == EXPECTED_SPLIT_COUNTS[
            "train"
        ]
        and target_results[
            "validation_target_count"
        ]
        == EXPECTED_SPLIT_COUNTS[
            "validation"
        ]
        and target_results[
            "test_target_count"
        ]
        == EXPECTED_SPLIT_COUNTS[
            "test"
        ]
        and target_results[
            "train_target_values"
        ]
        == [0, 1]
        and target_results[
            "validation_target_values"
        ]
        == [0, 1]
        and target_results[
            "test_target_values"
        ]
        == [0, 1]
    )

    overall_passed = (
        all(
            result[
                "validation_passed"
            ]
            for result in split_results.values()
        )
        and consistent_columns
        and targets_valid
    )

    return {
        "split_results": (
            split_results
        ),
        "consistent_feature_columns": (
            consistent_columns
        ),
        "target_results": (
            target_results
        ),
        "targets_valid": (
            targets_valid
        ),
        "validation_passed": (
            overall_passed
        ),
    }


def build_transformed_output(
    identifiers: pd.Series,
    targets: pd.Series,
    features: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Build a saved transformed split dataset."""

    metadata = pd.DataFrame(
        {
            PR_IDENTIFIER_COLUMN: (
                identifiers.reset_index(
                    drop=True
                )
            ),
            TARGET_COLUMN: (
                targets.reset_index(
                    drop=True
                )
            ),
            SPLIT_COLUMN: (
                split_name
            ),
        }
    )

    return pd.concat(
        [
            metadata,
            features.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )