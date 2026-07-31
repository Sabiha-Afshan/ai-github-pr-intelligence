"""Tests for manual model-feature approval."""

import pandas as pd

from src.models.feature_approval import (
    build_feature_group_summary,
    determine_exclusion_reason,
    get_approved_features,
    review_candidate_features,
    validate_approved_features,
)


def test_known_leakage_is_excluded() -> None:
    """Confirm known leakage features are blocked."""

    blocked_features = [
        "merge_hours",
        "resolution_hours",
        "merge_target",
        "resolution_hours_iqr_outlier",
        "merge_hours_iqr_outlier",
    ]

    for feature in blocked_features:
        assert determine_exclusion_reason(feature) is not None


def test_chronological_periods_are_excluded() -> None:
    """Confirm broad historical period fields are blocked."""

    blocked_features = [
        "created_year",
        "created_month",
        "created_quarter",
    ]

    for feature in blocked_features:
        assert determine_exclusion_reason(feature) is not None


def test_safe_features_are_approved() -> None:
    """Confirm pre-outcome features remain approved."""

    candidates = [
        "total_changes",
        "changed_files",
        "created_hour_utc",
        "merge_hours",
        "resolution_hours_iqr_outlier",
    ]

    review = review_candidate_features(candidates)

    approved_features = get_approved_features(review)

    assert "total_changes" in approved_features
    assert "changed_files" in approved_features
    assert "created_hour_utc" in approved_features
    assert "merge_hours" not in approved_features

    assert "resolution_hours_iqr_outlier" not in approved_features


def test_approved_feature_validation() -> None:
    """Confirm clean approved feature sets pass."""

    result = validate_approved_features(
        [
            "total_changes",
            "changed_files",
        ]
    )

    assert result["validation_passed"] is True

    assert result["blocked_features_present"] == []

    assert result["suspicious_derived_features"] == []


def test_blocked_feature_validation_fails() -> None:
    """Confirm blocked features fail final validation."""

    result = validate_approved_features(
        [
            "total_changes",
            "resolution_hours_iqr_outlier",
        ]
    )

    assert result["validation_passed"] is False

    assert result["blocked_features_present"] == ["resolution_hours_iqr_outlier"]


def test_feature_group_summary() -> None:
    """Confirm approved features are grouped."""

    review = pd.DataFrame(
        {
            "feature": [
                "total_changes",
                "changed_files",
                "merge_hours",
            ],
            "approved": [
                True,
                True,
                False,
            ],
            "decision": [
                "approved",
                "approved",
                "excluded",
            ],
            "feature_group": [
                "Code change",
                "Code change",
                "Excluded",
            ],
            "reason": [
                "Approved pre-outcome feature",
                "Approved pre-outcome feature",
                "Direct post-outcome leakage",
            ],
        }
    )

    result = build_feature_group_summary(review)

    assert len(result) == 1

    assert (
        result.loc[
            0,
            "feature_group",
        ]
        == "Code change"
    )

    assert (
        int(
            result.loc[
                0,
                "feature_count",
            ]
        )
        == 2
    )
