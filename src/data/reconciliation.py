"""Reconcile population, summary and detailed PR datasets."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.discovery import load_dataset
from src.data.population_validation import (
    standardize_pr_number_column,
    validate_population,
)


def prepare_dataset_for_merge(
    dataframe: pd.DataFrame,
    suffix: str,
) -> pd.DataFrame:
    """Prepare a PR dataset for safe merging."""

    normalized_dataframe = (
        standardize_pr_number_column(
            dataframe
        )
    )

    normalized_dataframe = (
        normalized_dataframe.drop_duplicates(
            subset=["pr_number"],
            keep="last",
        )
    )

    protected_columns = {
        "pr_number",
    }

    rename_mapping = {
        column: f"{column}{suffix}"
        for column in normalized_dataframe.columns
        if column not in protected_columns
    }

    return normalized_dataframe.rename(
        columns=rename_mapping
    )


def calculate_dataset_coverage(
    population_pr_numbers: set[int],
    dataset_pr_numbers: set[int],
    dataset_name: str,
) -> dict[str, Any]:
    """Calculate coverage against the population."""

    covered = (
        population_pr_numbers
        & dataset_pr_numbers
    )

    missing = (
        population_pr_numbers
        - dataset_pr_numbers
    )

    outside_population = (
        dataset_pr_numbers
        - population_pr_numbers
    )

    population_count = len(
        population_pr_numbers
    )

    coverage_rate = (
        len(covered) / population_count
        if population_count
        else 0.0
    )

    return {
        "dataset": dataset_name,
        "population_count": population_count,
        "dataset_unique_pr_count": len(
            dataset_pr_numbers
        ),
        "covered_population_count": len(
            covered
        ),
        "missing_population_count": len(
            missing
        ),
        "outside_population_count": len(
            outside_population
        ),
        "coverage_rate": coverage_rate,
        "missing_pr_numbers": sorted(
            missing
        ),
        "outside_population_pr_numbers": (
            sorted(outside_population)
        ),
    }


def reconcile_pr_datasets(
    population_path: Path,
    summary_path: Path,
    detailed_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, list[int]],
    dict[str, Any],
]:
    """Create the canonical PR population."""

    population_dataframe = (
        standardize_pr_number_column(
            load_dataset(population_path)
        )
    )

    summary_dataframe = (
        standardize_pr_number_column(
            load_dataset(summary_path)
        )
    )

    detailed_dataframe = (
        standardize_pr_number_column(
            load_dataset(detailed_path)
        )
    )

    population_dataframe = (
        population_dataframe.drop_duplicates(
            subset=["pr_number"],
            keep="last",
        )
    )

    summary_dataframe = (
        summary_dataframe.drop_duplicates(
            subset=["pr_number"],
            keep="last",
        )
    )

    detailed_dataframe = (
        detailed_dataframe.drop_duplicates(
            subset=["pr_number"],
            keep="last",
        )
    )

    summary_prepared = (
        prepare_dataset_for_merge(
            summary_dataframe,
            suffix="_summary",
        )
    )

    detailed_prepared = (
        prepare_dataset_for_merge(
            detailed_dataframe,
            suffix="_detail",
        )
    )

    canonical_dataframe = (
        population_dataframe.merge(
            summary_prepared,
            on="pr_number",
            how="left",
        )
    )

    canonical_dataframe = (
        canonical_dataframe.merge(
            detailed_prepared,
            on="pr_number",
            how="left",
        )
    )

    population_pr_numbers = set(
        population_dataframe[
            "pr_number"
        ].astype(int)
    )

    summary_pr_numbers = set(
        summary_dataframe[
            "pr_number"
        ].astype(int)
    )

    detailed_pr_numbers = set(
        detailed_dataframe[
            "pr_number"
        ].astype(int)
    )

    summary_coverage = (
        calculate_dataset_coverage(
            population_pr_numbers,
            summary_pr_numbers,
            "PR summaries",
        )
    )

    detailed_coverage = (
        calculate_dataset_coverage(
            population_pr_numbers,
            detailed_pr_numbers,
            "Detailed PR records",
        )
    )

    coverage_records = [
        summary_coverage,
        detailed_coverage,
    ]

    coverage_dataframe = pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "missing_pr_numbers",
                    "outside_population_pr_numbers",
                }
            }
            for record in coverage_records
        ]
    )

    missing_lists = {
        "missing_summary_pr_numbers": (
            summary_coverage[
                "missing_pr_numbers"
            ]
        ),
        "missing_detailed_pr_numbers": (
            detailed_coverage[
                "missing_pr_numbers"
            ]
        ),
        "summary_prs_outside_population": (
            summary_coverage[
                "outside_population_pr_numbers"
            ]
        ),
        "detailed_prs_outside_population": (
            detailed_coverage[
                "outside_population_pr_numbers"
            ]
        ),
    }

    population_validation = (
        validate_population(
            population_dataframe,
            expected_records=600,
        )
    )

    reconciliation_summary = {
        "population_validation": (
            population_validation
        ),
        "canonical_row_count": len(
            canonical_dataframe
        ),
        "canonical_unique_pr_count": int(
            canonical_dataframe[
                "pr_number"
            ].nunique()
        ),
        "summary_coverage_count": (
            summary_coverage[
                "covered_population_count"
            ]
        ),
        "detailed_coverage_count": (
            detailed_coverage[
                "covered_population_count"
            ]
        ),
    }

    return (
        canonical_dataframe,
        coverage_dataframe,
        missing_lists,
        reconciliation_summary,
    )