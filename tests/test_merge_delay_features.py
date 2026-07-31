"""Tests for Model 2 feature approval."""

import numpy as np
import pandas as pd

from src.models.merge_delay_features import (
    build_feature_group_summary,
    build_feature_review,
    build_modelling_dataset,
    get_approved_feature_names,
    normalize_binary_target,
    normalize_split_values,
    validate_delay_population_source,
    validate_modelling_dataset,
)


def create_delay_population() -> pd.DataFrame:
    """Create a representative merge-delay population."""

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
            "merge_delay_target": [
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            "merge_hours": [
                4.0,
                60.0,
                20.0,
                72.0,
                10.0,
                100.0,
            ],
            "was_merged": [
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "author_association": [
                "MEMBER",
                "CONTRIBUTOR",
                "MEMBER",
                "NONE",
                "MEMBER",
                "CONTRIBUTOR",
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
            "created_year": [
                2021,
                2021,
                2021,
                2022,
                2023,
                2024,
            ],
            "title_length": [
                10,
                20,
                30,
                40,
                50,
                60,
            ],
            "commit_count": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "has_test_changes": [
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            "deletion_ratio": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
            ],
            "file_records_returned": [
                2,
                3,
                4,
                5,
                6,
                7,
            ],
            "constant_feature": [
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "infinite_feature": [
                1.0,
                2.0,
                np.inf,
                4.0,
                5.0,
                6.0,
            ],
            "resolution_hours_iqr_outlier": [
                False,
                True,
                False,
                True,
                False,
                True,
            ],
        }
    )


def test_split_normalization() -> None:
    """Confirm supported split aliases are normalized."""

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
    """Confirm delay target is normalized."""

    result = normalize_binary_target(
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
    """Confirm the Stage 6A population passes."""

    result = validate_delay_population_source(create_delay_population())

    assert result["validation_passed"] is True


def test_feature_review_blocks_leakage() -> None:
    """Confirm target and identity leakage are blocked."""

    review = build_feature_review(create_delay_population())

    decisions = review.set_index("column")["approved_as_feature"].to_dict()

    assert not bool(decisions["merge_hours"])

    assert not bool(decisions["was_merged"])

    assert not bool(decisions["author_association"])

    assert not bool(decisions["created_year"])

    assert not bool(decisions["file_records_returned"])

    assert not bool(decisions["resolution_hours_iqr_outlier"])


def test_valid_numeric_features_are_approved() -> None:
    """Confirm contributor-neutral numeric features are approved."""

    review = build_feature_review(create_delay_population())

    approved = set(get_approved_feature_names(review))

    assert {
        "title_length",
        "commit_count",
        "has_test_changes",
        "deletion_ratio",
    }.issubset(approved)

    assert "constant_feature" not in approved

    assert "infinite_feature" not in approved


def test_modelling_dataset_is_leakage_safe() -> None:
    """Confirm only approved features enter the model dataset."""

    population = create_delay_population()

    review = build_feature_review(population)

    approved = get_approved_feature_names(review)

    modelling_dataset = build_modelling_dataset(
        delay_population=population,
        approved_features=approved,
    )

    assert "merge_hours" not in modelling_dataset.columns

    assert "author_association" not in modelling_dataset.columns

    assert set(modelling_dataset["split"]) == {
        "train",
        "validation",
        "test",
    }


def test_feature_group_summary() -> None:
    """Confirm feature decisions are summarized."""

    review = build_feature_review(create_delay_population())

    summary = build_feature_group_summary(review)

    assert not summary.empty

    assert {
        "approved",
        "blocked",
        "metadata",
    }.issubset(set(summary["decision"]))


def test_complete_modelling_dataset_validation() -> None:
    """Confirm a complete approved dataset passes validation."""

    population = create_delay_population()

    review = build_feature_review(population)

    approved = get_approved_feature_names(review)

    modelling_dataset = build_modelling_dataset(
        delay_population=population,
        approved_features=approved,
    )

    validation = validate_modelling_dataset(
        dataframe=modelling_dataset,
        approved_features=approved,
    )

    assert validation["validation_passed"] is True
