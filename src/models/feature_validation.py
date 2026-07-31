"""Validate and prepare machine-learning features."""

from typing import Any

import numpy as np
import pandas as pd

from src.data.processed_dataset_validation import (
    normalize_target,
)
from src.models.feature_schema import (
    BOOLEAN_TEXT_VALUES,
    NON_MODEL_COLUMNS,
    PR_IDENTIFIER_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    identify_leakage_columns,
    normalize_column_name,
)


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "validate": "validation",
    "val": "validation",
    "test": "test",
    "testing": "test",
}


def normalize_split_series(
    series: pd.Series,
) -> pd.Series:
    """Normalize split names."""

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(SPLIT_ALIASES)
        .astype("string")
    )


def identify_split_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Identify the split-assignment column."""

    candidates = (
        "split",
        "dataset_split",
        "model_split",
        "split_assignment",
        "time_split",
        "set",
    )

    normalized_mapping = {
        normalize_column_name(column): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in normalized_mapping:
            return normalized_mapping[candidate]

    return None


def merge_features_with_splits(
    feature_dataframe: pd.DataFrame,
    split_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Attach chronological split assignments to feature rows."""

    if (
        PR_IDENTIFIER_COLUMN
        not in feature_dataframe.columns
    ):
        raise ValueError(
            "Feature dataset is missing pr_number."
        )

    if (
        PR_IDENTIFIER_COLUMN
        not in split_dataframe.columns
    ):
        raise ValueError(
            "Split dataset is missing pr_number."
        )

    split_column = identify_split_column(
        split_dataframe
    )

    if split_column is None:
        raise ValueError(
            "No recognized split column was found."
        )

    features = feature_dataframe.copy()

    splits = split_dataframe[
        [
            PR_IDENTIFIER_COLUMN,
            split_column,
        ]
    ].copy()

    features[
        PR_IDENTIFIER_COLUMN
    ] = pd.to_numeric(
        features[
            PR_IDENTIFIER_COLUMN
        ],
        errors="coerce",
    ).astype("Int64")

    splits[
        PR_IDENTIFIER_COLUMN
    ] = pd.to_numeric(
        splits[
            PR_IDENTIFIER_COLUMN
        ],
        errors="coerce",
    ).astype("Int64")

    splits[
        SPLIT_COLUMN
    ] = normalize_split_series(
        splits[split_column]
    )

    splits = splits[
        [
            PR_IDENTIFIER_COLUMN,
            SPLIT_COLUMN,
        ]
    ]

    merged = features.merge(
        splits,
        on=PR_IDENTIFIER_COLUMN,
        how="left",
        validate="one_to_one",
    )

    return merged


def convert_boolean_like_series(
    series: pd.Series,
) -> pd.Series | None:
    """Convert a boolean-like column to numeric values."""

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype("Int64")

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    non_missing = normalized.dropna()

    if non_missing.empty:
        return None

    unique_values = set(
        non_missing.unique()
    )

    if unique_values.issubset(
        BOOLEAN_TEXT_VALUES
    ):
        return normalized.map(
            BOOLEAN_TEXT_VALUES
        ).astype("Int64")

    return None


def convert_feature_series(
    series: pd.Series,
) -> pd.Series | None:
    """Convert a candidate feature to numeric form."""

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    boolean_conversion = (
        convert_boolean_like_series(
            series
        )
    )

    if boolean_conversion is not None:
        return boolean_conversion

    numeric_conversion = pd.to_numeric(
        series,
        errors="coerce",
    )

    original_non_missing = int(
        series.notna().sum()
    )

    converted_non_missing = int(
        numeric_conversion.notna().sum()
    )

    if original_non_missing == 0:
        return None

    conversion_rate = (
        converted_non_missing
        / original_non_missing
    )

    if conversion_rate >= 0.95:
        return numeric_conversion

    return None


def build_numeric_feature_matrix(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build a numeric feature matrix and feature audit."""

    feature_columns: dict[
        str,
        pd.Series,
    ] = {}

    audit_records: list[
        dict[str, Any]
    ] = []

    normalized_exclusions = {
        normalize_column_name(column)
        for column in NON_MODEL_COLUMNS
    }

    for column in dataframe.columns:
        normalized_column = (
            normalize_column_name(
                column
            )
        )

        if (
            normalized_column
            in normalized_exclusions
        ):
            audit_records.append(
                {
                    "column": column,
                    "status": "excluded",
                    "reason": (
                        "Identifier, target, timestamp, "
                        "split or leakage column"
                    ),
                    "source_dtype": str(
                        dataframe[column].dtype
                    ),
                    "missing_count": int(
                        dataframe[
                            column
                        ].isna().sum()
                    ),
                    "unique_count": int(
                        dataframe[
                            column
                        ].nunique(
                            dropna=True
                        )
                    ),
                }
            )

            continue

        converted = convert_feature_series(
            dataframe[column]
        )

        if converted is None:
            audit_records.append(
                {
                    "column": column,
                    "status": "excluded",
                    "reason": (
                        "Not safely convertible "
                        "to a numeric feature"
                    ),
                    "source_dtype": str(
                        dataframe[column].dtype
                    ),
                    "missing_count": int(
                        dataframe[
                            column
                        ].isna().sum()
                    ),
                    "unique_count": int(
                        dataframe[
                            column
                        ].nunique(
                            dropna=True
                        )
                    ),
                }
            )

            continue

        numeric_feature = (
            pd.to_numeric(
                converted,
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .astype(float)
        )

        unique_count = int(
            numeric_feature.nunique(
                dropna=True
            )
        )

        if unique_count <= 1:
            audit_records.append(
                {
                    "column": column,
                    "status": "excluded",
                    "reason": (
                        "Constant or empty feature"
                    ),
                    "source_dtype": str(
                        dataframe[column].dtype
                    ),
                    "missing_count": int(
                        numeric_feature.isna().sum()
                    ),
                    "unique_count": unique_count,
                }
            )

            continue

        feature_columns[
            column
        ] = numeric_feature

        audit_records.append(
            {
                "column": column,
                "status": "selected",
                "reason": (
                    "Numeric model feature"
                ),
                "source_dtype": str(
                    dataframe[column].dtype
                ),
                "missing_count": int(
                    numeric_feature.isna().sum()
                ),
                "unique_count": unique_count,
            }
        )

    feature_matrix = pd.DataFrame(
        feature_columns,
        index=dataframe.index,
    )

    feature_audit = pd.DataFrame(
        audit_records
    )

    return (
        feature_matrix,
        feature_audit,
    )


def validate_target(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the merge-outcome target."""

    if TARGET_COLUMN not in dataframe.columns:
        return {
            "target_column": TARGET_COLUMN,
            "target_present": False,
            "validation_passed": False,
            "reason": (
                "Target column is missing."
            ),
        }

    normalized_target = normalize_target(
        dataframe[TARGET_COLUMN]
    )

    distribution = {
        str(key): int(value)
        for key, value in (
            normalized_target.value_counts(
                dropna=False
            )
            .sort_index()
            .to_dict()
        ).items()
    }

    missing_count = int(
        normalized_target.isna().sum()
    )

    validation_passed = (
        missing_count == 0
        and distribution.get(
            "0",
            0,
        )
        == 300
        and distribution.get(
            "1",
            0,
        )
        == 300
    )

    return {
        "target_column": TARGET_COLUMN,
        "target_present": True,
        "missing_target_count": (
            missing_count
        ),
        "target_distribution": (
            distribution
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def validate_splits(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate chronological split assignments."""

    if SPLIT_COLUMN not in dataframe.columns:
        return {
            "split_column": SPLIT_COLUMN,
            "validation_passed": False,
            "reason": (
                "Split column is missing."
            ),
        }

    normalized_split = (
        normalize_split_series(
            dataframe[SPLIT_COLUMN]
        )
    )

    distribution = {
        str(key): int(value)
        for key, value in (
            normalized_split.value_counts(
                dropna=False
            ).to_dict()
        ).items()
    }

    missing_count = int(
        normalized_split.isna().sum()
    )

    expected_distribution = {
        "train": 440,
        "validation": 84,
        "test": 76,
    }

    validation_passed = (
        missing_count == 0
        and distribution
        == expected_distribution
    )

    return {
        "split_column": SPLIT_COLUMN,
        "missing_split_count": (
            missing_count
        ),
        "split_distribution": (
            distribution
        ),
        "expected_distribution": (
            expected_distribution
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def validate_feature_matrix(
    feature_matrix: pd.DataFrame,
    source_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the selected numeric feature matrix."""

    leakage_columns = (
        identify_leakage_columns(
            list(feature_matrix.columns)
        )
    )

    infinite_count = int(
        np.isinf(
            feature_matrix.to_numpy(
                dtype=float
            )
        ).sum()
    )

    missing_count = int(
        feature_matrix.isna().sum().sum()
    )

    columns_with_missing = int(
        (
            feature_matrix.isna().sum()
            > 0
        ).sum()
    )

    duplicated_feature_names = (
        feature_matrix.columns[
            feature_matrix.columns.duplicated()
        ].tolist()
    )

    target_in_features = (
        TARGET_COLUMN
        in feature_matrix.columns
    )

    row_alignment_passed = (
        len(feature_matrix)
        == len(source_dataframe)
    )

    validation_passed = (
        not feature_matrix.empty
        and row_alignment_passed
        and not leakage_columns
        and not target_in_features
        and infinite_count == 0
        and not duplicated_feature_names
    )

    return {
        "row_count": len(
            feature_matrix
        ),
        "feature_count": len(
            feature_matrix.columns
        ),
        "feature_names": list(
            feature_matrix.columns
        ),
        "total_missing_feature_values": (
            missing_count
        ),
        "columns_with_missing_values": (
            columns_with_missing
        ),
        "infinite_value_count": (
            infinite_count
        ),
        "duplicated_feature_names": (
            duplicated_feature_names
        ),
        "target_in_features": (
            target_in_features
        ),
        "identified_leakage_columns": (
            leakage_columns
        ),
        "row_alignment_passed": (
            row_alignment_passed
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def calculate_feature_missing_summary(
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate feature-level missing-value statistics."""

    records = []

    for column in feature_matrix.columns:
        missing_count = int(
            feature_matrix[
                column
            ].isna().sum()
        )

        records.append(
            {
                "feature": column,
                "missing_count": (
                    missing_count
                ),
                "missing_percent": (
                    round(
                        missing_count
                        / len(feature_matrix)
                        * 100,
                        2,
                    )
                    if len(feature_matrix)
                    > 0
                    else 0.0
                ),
                "unique_count": int(
                    feature_matrix[
                        column
                    ].nunique(
                        dropna=True
                    )
                ),
                "minimum": (
                    float(
                        feature_matrix[
                            column
                        ].min()
                    )
                    if feature_matrix[
                        column
                    ].notna().any()
                    else None
                ),
                "maximum": (
                    float(
                        feature_matrix[
                            column
                        ].max()
                    )
                    if feature_matrix[
                        column
                    ].notna().any()
                    else None
                ),
            }
        )

    return pd.DataFrame(
        records
    ).sort_values(
        [
            "missing_percent",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def build_modelling_dataset(
    feature_dataframe: pd.DataFrame,
    split_dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build the complete modelling table and feature audit."""

    merged_dataframe = (
        merge_features_with_splits(
            feature_dataframe,
            split_dataframe,
        )
    )

    feature_matrix, feature_audit = (
        build_numeric_feature_matrix(
            merged_dataframe
        )
    )

    modelling_dataframe = (
        pd.concat(
            [
                merged_dataframe[
                    [
                        PR_IDENTIFIER_COLUMN,
                        TARGET_COLUMN,
                        SPLIT_COLUMN,
                    ]
                ].reset_index(
                    drop=True
                ),
                feature_matrix.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )
    )

    return (
        modelling_dataframe,
        feature_matrix,
        feature_audit,
    )