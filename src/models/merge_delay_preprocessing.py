"""Train-only preprocessing for Model 2 merge-delay prediction."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PR_IDENTIFIER_COLUMN = "pr_number"
SPLIT_COLUMN = "split"
TARGET_COLUMN = "merge_delay_target"

ALLOWED_SPLITS = {
    "train",
    "validation",
    "test",
}


@dataclass(frozen=True)
class FeatureTypeResult:
    """Classification of Model 2 features."""

    binary_features: list[str]
    continuous_features: list[str]


@dataclass(frozen=True)
class PreparedModel2Data:
    """All transformed Model 2 data and metadata."""

    preprocessor: ColumnTransformer
    feature_names: list[str]
    binary_features: list[str]
    continuous_features: list[str]

    train_identifiers: pd.Series
    validation_identifiers: pd.Series
    test_identifiers: pd.Series

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame


def normalize_split_values(
    values: pd.Series,
) -> pd.Series:
    """Normalize split names."""

    normalized = (
        values.astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "train": "train",
        "training": "train",
        "validation": "validation",
        "validate": "validation",
        "valid": "validation",
        "val": "validation",
        "test": "test",
        "testing": "test",
    }

    result = normalized.map(mapping)

    if result.isna().any():
        unknown_values = sorted(
            normalized.loc[
                result.isna()
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unexpected split values: "
            f"{unknown_values}"
        )

    return result.astype(str)


def normalize_target(
    values: pd.Series,
) -> pd.Series:
    """Normalize the binary merge-delay target."""

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    if numeric_values.isna().any():
        missing_count = int(
            numeric_values.isna().sum()
        )

        raise ValueError(
            f"{TARGET_COLUMN} contains "
            f"{missing_count} invalid values."
        )

    integer_values = numeric_values.astype(int)

    invalid_values = sorted(
        set(integer_values.unique())
        - {0, 1}
    )

    if invalid_values:
        raise ValueError(
            f"{TARGET_COLUMN} contains invalid values: "
            f"{invalid_values}"
        )

    return integer_values


def get_feature_names(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return Model 2 feature columns."""

    excluded_columns = {
        PR_IDENTIFIER_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
    }

    feature_names = [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]

    if not feature_names:
        raise ValueError(
            "No Model 2 feature columns were found."
        )

    return feature_names


def validate_source_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the approved Model 2 dataset."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
    }

    missing_required_columns = sorted(
        required_columns
        - set(dataframe.columns)
    )

    duplicate_pr_count = (
        int(
            dataframe.duplicated(
                subset=[
                    PR_IDENTIFIER_COLUMN
                ]
            ).sum()
        )
        if PR_IDENTIFIER_COLUMN
        in dataframe.columns
        else None
    )

    feature_names = (
        get_feature_names(
            dataframe
        )
        if not missing_required_columns
        else []
    )

    non_numeric_features = []

    infinite_value_count = 0

    for feature in feature_names:
        numeric_values = pd.to_numeric(
            dataframe[
                feature
            ],
            errors="coerce",
        )

        source_non_missing_count = int(
            dataframe[
                feature
            ].notna().sum()
        )

        numeric_non_missing_count = int(
            numeric_values.notna().sum()
        )

        if (
            source_non_missing_count
            != numeric_non_missing_count
        ):
            non_numeric_features.append(
                feature
            )

        finite_values = (
            numeric_values.dropna()
        )

        infinite_value_count += int(
            (
                ~np.isfinite(
                    finite_values
                )
            ).sum()
        )

    validation_passed = (
        not missing_required_columns
        and duplicate_pr_count == 0
        and len(feature_names) > 0
        and not non_numeric_features
        and infinite_value_count == 0
    )

    return {
        "row_count": len(dataframe),
        "column_count": len(
            dataframe.columns
        ),
        "feature_count": len(
            feature_names
        ),
        "missing_required_columns": (
            missing_required_columns
        ),
        "duplicate_pr_count": (
            duplicate_pr_count
        ),
        "non_numeric_features": (
            non_numeric_features
        ),
        "infinite_value_count": (
            infinite_value_count
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def identify_feature_types(
    training_features: pd.DataFrame,
) -> FeatureTypeResult:
    """Separate binary and continuous features using training data only."""

    binary_features = []

    continuous_features = []

    for feature in training_features.columns:
        numeric_values = pd.to_numeric(
            training_features[
                feature
            ],
            errors="coerce",
        )

        unique_values = set(
            numeric_values
            .dropna()
            .unique()
            .tolist()
        )

        is_binary = (
            len(unique_values) > 0
            and unique_values.issubset(
                {0, 1, False, True}
            )
        )

        if is_binary:
            binary_features.append(
                feature
            )
        else:
            continuous_features.append(
                feature
            )

    if (
        len(binary_features)
        + len(continuous_features)
        != len(training_features.columns)
    ):
        raise ValueError(
            "Feature classification did not cover all columns."
        )

    return FeatureTypeResult(
        binary_features=(
            binary_features
        ),
        continuous_features=(
            continuous_features
        ),
    )


def build_preprocessor(
    binary_features: list[str],
    continuous_features: list[str],
) -> ColumnTransformer:
    """Build the train-only Model 2 preprocessing pipeline."""

    transformers = []

    if binary_features:
        binary_pipeline = Pipeline(
            steps=[
                (
                    "most_frequent_imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "binary",
                binary_pipeline,
                binary_features,
            )
        )

    if continuous_features:
        continuous_pipeline = Pipeline(
            steps=[
                (
                    "median_imputer",
                    SimpleImputer(
                        strategy="median"
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
                continuous_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No preprocessing transformers were created."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )


def create_split_data(
    dataframe: pd.DataFrame,
    split_name: str,
    feature_names: list[str],
) -> tuple[
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """Extract one chronological split."""

    if split_name not in ALLOWED_SPLITS:
        raise ValueError(
            f"Unexpected split name: {split_name}"
        )

    split_rows = dataframe.loc[
        dataframe[
            SPLIT_COLUMN
        ]
        == split_name
    ].copy()

    if split_rows.empty:
        raise ValueError(
            f"Split {split_name!r} contains no rows."
        )

    identifiers = (
        split_rows[
            PR_IDENTIFIER_COLUMN
        ]
        .reset_index(drop=True)
    )

    targets = (
        split_rows[
            TARGET_COLUMN
        ]
        .astype(int)
        .reset_index(drop=True)
    )

    features = (
        split_rows[
            feature_names
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .reset_index(drop=True)
    )

    return (
        identifiers,
        targets,
        features,
    )


def transform_to_dataframe(
    preprocessor: ColumnTransformer,
    features: pd.DataFrame,
    output_feature_names: list[str],
) -> pd.DataFrame:
    """Transform one feature split into a DataFrame."""

    transformed = preprocessor.transform(
        features
    )

    transformed_array = np.asarray(
        transformed,
        dtype=float,
    )

    if transformed_array.shape[1] != len(
        output_feature_names
    ):
        raise ValueError(
            "Transformed feature count does not match "
            "the expected output feature count."
        )

    return pd.DataFrame(
        transformed_array,
        columns=output_feature_names,
    )


def prepare_model2_data(
    dataframe: pd.DataFrame,
) -> PreparedModel2Data:
    """Fit train-only preprocessing and transform all splits."""

    source_validation = (
        validate_source_dataset(
            dataframe
        )
    )

    if not source_validation[
        "validation_passed"
    ]:
        raise ValueError(
            "Model 2 source dataset failed validation: "
            f"{source_validation}"
        )

    working = dataframe.copy()

    working[
        SPLIT_COLUMN
    ] = normalize_split_values(
        working[
            SPLIT_COLUMN
        ]
    )

    working[
        TARGET_COLUMN
    ] = normalize_target(
        working[
            TARGET_COLUMN
        ]
    )

    feature_names = get_feature_names(
        working
    )

    (
        train_identifiers,
        y_train,
        raw_x_train,
    ) = create_split_data(
        dataframe=working,
        split_name="train",
        feature_names=feature_names,
    )

    (
        validation_identifiers,
        y_validation,
        raw_x_validation,
    ) = create_split_data(
        dataframe=working,
        split_name="validation",
        feature_names=feature_names,
    )

    (
        test_identifiers,
        y_test,
        raw_x_test,
    ) = create_split_data(
        dataframe=working,
        split_name="test",
        feature_names=feature_names,
    )

    feature_types = identify_feature_types(
        raw_x_train
    )

    preprocessor = build_preprocessor(
        binary_features=(
            feature_types.binary_features
        ),
        continuous_features=(
            feature_types.continuous_features
        ),
    )

    preprocessor.fit(
        raw_x_train
    )

    output_feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    x_train = transform_to_dataframe(
        preprocessor=preprocessor,
        features=raw_x_train,
        output_feature_names=(
            output_feature_names
        ),
    )

    x_validation = transform_to_dataframe(
        preprocessor=preprocessor,
        features=raw_x_validation,
        output_feature_names=(
            output_feature_names
        ),
    )

    x_test = transform_to_dataframe(
        preprocessor=preprocessor,
        features=raw_x_test,
        output_feature_names=(
            output_feature_names
        ),
    )

    return PreparedModel2Data(
        preprocessor=preprocessor,
        feature_names=(
            output_feature_names
        ),
        binary_features=(
            feature_types.binary_features
        ),
        continuous_features=(
            feature_types.continuous_features
        ),
        train_identifiers=(
            train_identifiers
        ),
        validation_identifiers=(
            validation_identifiers
        ),
        test_identifiers=(
            test_identifiers
        ),
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
    )


def build_output_split(
    identifiers: pd.Series,
    targets: pd.Series,
    features: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Build a saved preprocessed split."""

    if not (
        len(identifiers)
        == len(targets)
        == len(features)
    ):
        raise ValueError(
            "Identifiers, targets and features are misaligned."
        )

    output = features.copy()

    output.insert(
        0,
        TARGET_COLUMN,
        targets.reset_index(
            drop=True
        ),
    )

    output.insert(
        0,
        SPLIT_COLUMN,
        split_name,
    )

    output.insert(
        0,
        PR_IDENTIFIER_COLUMN,
        identifiers.reset_index(
            drop=True
        ),
    )

    return output


def validate_transformed_split(
    dataframe: pd.DataFrame,
    expected_split: str,
    expected_rows: int,
    expected_features: int,
) -> dict[str, Any]:
    """Validate one transformed split."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
    }

    missing_required_columns = sorted(
        required_columns
        - set(dataframe.columns)
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column
        not in required_columns
    ]

    split_values = (
        sorted(
            dataframe[
                SPLIT_COLUMN
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        if SPLIT_COLUMN
        in dataframe.columns
        else []
    )

    target_values = (
        sorted(
            dataframe[
                TARGET_COLUMN
            ]
            .astype(int)
            .unique()
            .tolist()
        )
        if TARGET_COLUMN
        in dataframe.columns
        else []
    )

    missing_value_count = int(
        dataframe[
            feature_columns
        ].isna().sum().sum()
    )

    infinite_value_count = int(
        (
            ~np.isfinite(
                dataframe[
                    feature_columns
                ].to_numpy(
                    dtype=float
                )
            )
        ).sum()
    )

    validation_passed = (
        not missing_required_columns
        and len(dataframe)
        == expected_rows
        and len(feature_columns)
        == expected_features
        and split_values
        == [expected_split]
        and set(target_values).issubset(
            {0, 1}
        )
        and missing_value_count == 0
        and infinite_value_count == 0
    )

    return {
        "row_count": len(dataframe),
        "expected_row_count": (
            expected_rows
        ),
        "feature_count": len(
            feature_columns
        ),
        "expected_feature_count": (
            expected_features
        ),
        "missing_required_columns": (
            missing_required_columns
        ),
        "split_values": (
            split_values
        ),
        "target_values": (
            target_values
        ),
        "missing_value_count": (
            missing_value_count
        ),
        "infinite_value_count": (
            infinite_value_count
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def validate_prepared_data(
    prepared: PreparedModel2Data,
) -> dict[str, Any]:
    """Validate the complete Stage 6C output."""

    train_output = build_output_split(
        identifiers=(
            prepared.train_identifiers
        ),
        targets=prepared.y_train,
        features=prepared.x_train,
        split_name="train",
    )

    validation_output = build_output_split(
        identifiers=(
            prepared.validation_identifiers
        ),
        targets=(
            prepared.y_validation
        ),
        features=prepared.x_validation,
        split_name="validation",
    )

    test_output = build_output_split(
        identifiers=(
            prepared.test_identifiers
        ),
        targets=prepared.y_test,
        features=prepared.x_test,
        split_name="test",
    )

    expected_feature_count = len(
        prepared.feature_names
    )

    split_results = {
        "train": validate_transformed_split(
            dataframe=train_output,
            expected_split="train",
            expected_rows=220,
            expected_features=(
                expected_feature_count
            ),
        ),
        "validation": (
            validate_transformed_split(
                dataframe=validation_output,
                expected_split="validation",
                expected_rows=42,
                expected_features=(
                    expected_feature_count
                ),
            )
        ),
        "test": validate_transformed_split(
            dataframe=test_output,
            expected_split="test",
            expected_rows=38,
            expected_features=(
                expected_feature_count
            ),
        ),
    }

    feature_columns_consistent = (
        list(
            prepared.x_train.columns
        )
        == list(
            prepared.x_validation.columns
        )
        == list(
            prepared.x_test.columns
        )
    )

    targets_valid = all(
        set(
            target.astype(int)
            .unique()
            .tolist()
        ).issubset(
            {0, 1}
        )
        for target in (
            prepared.y_train,
            prepared.y_validation,
            prepared.y_test,
        )
    )

    both_classes_present = all(
        set(
            target.astype(int)
            .unique()
            .tolist()
        )
        == {0, 1}
        for target in (
            prepared.y_train,
            prepared.y_validation,
            prepared.y_test,
        )
    )

    validation_passed = (
        all(
            result[
                "validation_passed"
            ]
            for result
            in split_results.values()
        )
        and feature_columns_consistent
        and targets_valid
        and both_classes_present
    )

    return {
        "split_results": (
            split_results
        ),
        "feature_columns_consistent": (
            feature_columns_consistent
        ),
        "targets_valid": (
            targets_valid
        ),
        "both_classes_present": (
            both_classes_present
        ),
        "validation_passed": (
            validation_passed
        ),
    }