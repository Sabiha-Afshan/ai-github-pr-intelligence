"""Compare detailed and validated PR datasets."""

from typing import Any

import pandas as pd


def compare_dataset_populations(
    source_dataframe: pd.DataFrame,
    validated_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Compare PR membership before and after validation."""

    source_prs = set(
        pd.to_numeric(
            source_dataframe["pr_number"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    validated_prs = set(
        pd.to_numeric(
            validated_dataframe["pr_number"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    missing_after_validation = (
        source_prs - validated_prs
    )

    introduced_after_validation = (
        validated_prs - source_prs
    )

    return {
        "source_row_count": len(
            source_dataframe
        ),
        "validated_row_count": len(
            validated_dataframe
        ),
        "source_unique_pr_count": len(
            source_prs
        ),
        "validated_unique_pr_count": len(
            validated_prs
        ),
        "matched_pr_count": len(
            source_prs & validated_prs
        ),
        "missing_after_validation_count": len(
            missing_after_validation
        ),
        "introduced_after_validation_count": len(
            introduced_after_validation
        ),
        "missing_after_validation": sorted(
            missing_after_validation
        ),
        "introduced_after_validation": sorted(
            introduced_after_validation
        ),
        "exact_population_alignment": (
            source_prs == validated_prs
        ),
    }


def compare_shared_columns(
    source_dataframe: pd.DataFrame,
    validated_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compare missing values across shared columns."""

    shared_columns = sorted(
        set(source_dataframe.columns)
        & set(validated_dataframe.columns)
    )

    records = []

    for column in shared_columns:
        source_missing = int(
            source_dataframe[column]
            .isna()
            .sum()
        )

        validated_missing = int(
            validated_dataframe[column]
            .isna()
            .sum()
        )

        records.append(
            {
                "column": column,
                "source_missing_count": (
                    source_missing
                ),
                "validated_missing_count": (
                    validated_missing
                ),
                "missing_count_change": (
                    validated_missing
                    - source_missing
                ),
            }
        )

    return pd.DataFrame(records)