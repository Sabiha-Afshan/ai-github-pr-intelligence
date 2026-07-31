"""Tests for merge-delay target construction."""

import numpy as np
import pandas as pd

from src.models.merge_delay_target import (
    PRIMARY_DELAY_TARGET,
    add_delay_targets,
    attach_split_assignments,
    audit_target_leakage_columns,
    build_merge_duration_summary,
    build_split_target_summary,
    build_threshold_summary,
    create_merged_population,
    normalize_boolean_target,
    validate_delay_dataset,
    validate_source_dataset,
)


def create_source_data() -> pd.DataFrame:
    """Create a representative PR dataset."""

    return pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "was_merged": [
                True,
                True,
                False,
                True,
                True,
                True,
            ],
            "merge_hours": [
                10.0,
                49.0,
                np.nan,
                72.0,
                168.5,
                30.0,
            ],
            "created_at": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-02-01",
                    "2021-03-01",
                    "2022-01-01",
                    "2023-01-01",
                    "2024-01-01",
                ],
                utc=True,
            ),
            "title_length": [
                10,
                20,
                30,
                40,
                50,
                60,
            ],
        }
    )


def create_split_assignments() -> pd.DataFrame:
    """Create chronological split assignments using supported aliases."""

    return pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "split": [
                "training",
                "training",
                "training",
                "validation",
                "test",
                "test",
            ],
        }
    )


def test_boolean_normalization() -> None:
    """Confirm Boolean outcomes normalize to zero and one."""

    values = pd.Series(
        [
            True,
            False,
            True,
        ]
    )

    result = normalize_boolean_target(
        values,
        "was_merged",
    )

    assert result.tolist() == [
        1,
        0,
        1,
    ]


def test_source_validation() -> None:
    """Confirm a valid source dataset passes."""

    result = validate_source_dataset(create_source_data())

    assert result["validation_passed"] is True


def test_merged_population() -> None:
    """Confirm only merged rows are retained."""

    result = create_merged_population(create_source_data())

    assert len(result) == 5

    assert result["was_merged"].eq(1).all()

    assert result["merge_hours"].notna().all()


def test_delay_targets() -> None:
    """Confirm all delay thresholds are calculated."""

    merged = create_merged_population(create_source_data())

    result = add_delay_targets(merged)

    target_by_pr = result.set_index("pr_number")[PRIMARY_DELAY_TARGET].to_dict()

    assert target_by_pr[1] == 0
    assert target_by_pr[2] == 1
    assert target_by_pr[4] == 1
    assert target_by_pr[5] == 1
    assert target_by_pr[6] == 0


def test_threshold_summary() -> None:
    """Confirm threshold distributions are summarized."""

    delay_data = add_delay_targets(create_merged_population(create_source_data()))

    summary = build_threshold_summary(delay_data)

    assert len(summary) == 4

    primary = summary.loc[summary["target_column"] == PRIMARY_DELAY_TARGET].iloc[0]

    assert primary["threshold_hours"] == 48.0


def test_split_attachment_and_summary() -> None:
    """Confirm split assignments attach correctly."""

    delay_data = add_delay_targets(create_merged_population(create_source_data()))

    result = attach_split_assignments(
        delay_dataframe=delay_data,
        split_assignments=(create_split_assignments()),
    )

    summary = build_split_target_summary(result)

    assert set(result["split"]) == {
        "train",
        "validation",
        "test",
    }

    assert int(summary["row_count"].sum()) == 5


def test_duration_summary() -> None:
    """Confirm merge-duration statistics are produced."""

    merged = create_merged_population(create_source_data())

    summary = build_merge_duration_summary(merged)

    assert summary["row_count"] == 5

    assert summary["maximum_hours"] == 168.5


def test_leakage_audit() -> None:
    """Confirm target leakage columns are blocked."""

    delay_data = add_delay_targets(create_merged_population(create_source_data()))

    result = audit_target_leakage_columns(delay_data)

    merge_hours_row = result.loc[result["column"] == "merge_hours"].iloc[0]

    assert merge_hours_row["allowed_as_model_feature"] is False or not bool(
        merge_hours_row["allowed_as_model_feature"]
    )


def test_complete_delay_dataset_validation() -> None:
    """Confirm a completed Model 2 dataset passes."""

    delay_data = add_delay_targets(create_merged_population(create_source_data()))

    delay_data = attach_split_assignments(
        delay_dataframe=delay_data,
        split_assignments=(create_split_assignments()),
    )

    validation = validate_delay_dataset(delay_data)

    assert validation["validation_passed"] is True
