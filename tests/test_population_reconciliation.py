"""Tests for PR population reconciliation."""

import pandas as pd

from src.data.reconciliation import (
    calculate_dataset_coverage,
    prepare_dataset_for_merge,
)


def test_dataset_coverage_is_calculated() -> None:
    """Confirm population coverage calculation."""

    population = {
        1,
        2,
        3,
        4,
    }

    dataset = {
        2,
        3,
        5,
    }

    result = calculate_dataset_coverage(
        population_pr_numbers=population,
        dataset_pr_numbers=dataset,
        dataset_name="Test dataset",
    )

    assert result["covered_population_count"] == 2

    assert result["missing_population_count"] == 2

    assert result["outside_population_count"] == 1

    assert result["coverage_rate"] == 0.5

    assert result["missing_pr_numbers"] == [
        1,
        4,
    ]


def test_merge_preparation_adds_suffix() -> None:
    """Confirm non-key columns are renamed."""

    dataframe = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ],
            "title": [
                "First",
                "Second",
            ],
            "state": [
                "closed",
                "closed",
            ],
        }
    )

    result = prepare_dataset_for_merge(
        dataframe,
        suffix="_summary",
    )

    assert "pr_number" in result.columns
    assert "title_summary" in result.columns
    assert "state_summary" in result.columns
