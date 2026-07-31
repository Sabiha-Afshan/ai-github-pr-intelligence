"""Tests for controlled Model 2 candidate selection."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.merge_delay_selection import (
    MergeDelayCandidate,
    build_all_candidates,
    build_candidate_comparison,
    build_family_summary,
    build_hist_gradient_boosting_candidates,
    build_logistic_candidates,
    build_random_forest_candidates,
    evaluate_candidate,
    select_locked_candidate,
)
from src.models.merge_delay_training import (
    load_preprocessed_split,
)


def create_split(
    split_name: str,
    row_count: int,
    starting_pr: int,
) -> pd.DataFrame:
    """Create a synthetic preprocessed split."""

    generator = np.random.default_rng(starting_pr)

    targets = np.array([0, 1] * (row_count // 2))

    if len(targets) < row_count:
        targets = np.append(
            targets,
            0,
        )

    signal = targets + generator.normal(
        0,
        0.50,
        row_count,
    )

    return pd.DataFrame(
        {
            "pr_number": range(
                starting_pr,
                starting_pr + row_count,
            ),
            "split": split_name,
            "merge_delay_target": targets,
            "title_length": signal,
            "commit_count": (
                signal
                + generator.normal(
                    0,
                    0.30,
                    row_count,
                )
            ),
            "deletion_ratio": (
                generator.normal(
                    0,
                    1,
                    row_count,
                )
            ),
            "has_test_changes": (
                generator.integers(
                    0,
                    2,
                    row_count,
                )
            ),
        }
    )


def test_candidate_counts() -> None:
    """Confirm the controlled search size."""

    assert len(build_logistic_candidates()) == 7

    assert len(build_random_forest_candidates()) == 6

    assert len(build_hist_gradient_boosting_candidates()) == 6

    assert len(build_all_candidates()) == 19


def test_candidate_ids_are_unique() -> None:
    """Confirm every candidate has a unique identifier."""

    candidates = build_all_candidates()

    candidate_ids = [candidate.candidate_id for candidate in candidates]

    assert len(candidate_ids) == len(set(candidate_ids))


def test_single_candidate_evaluation() -> None:
    """Confirm one candidate can be trained and evaluated."""

    training_split = load_preprocessed_split(
        create_split(
            "train",
            100,
            1,
        ),
        expected_split="train",
    )

    validation_split = load_preprocessed_split(
        create_split(
            "validation",
            30,
            1001,
        ),
        expected_split="validation",
    )

    candidate = MergeDelayCandidate(
        candidate_id="test_logistic",
        model_family="logistic_regression",
        model=LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        ),
        parameters={
            "class_weight": "balanced",
            "solver": "liblinear",
        },
    )

    result = evaluate_candidate(
        candidate=candidate,
        training_split=training_split,
        validation_split=validation_split,
    )

    assert result.candidate.candidate_id == ("test_logistic")

    assert 0 <= result.selected_threshold <= 1

    assert len(result.validation_predictions) == 30


def test_candidate_ranking_and_selection() -> None:
    """Confirm evaluated candidates are ranked and locked."""

    training_split = load_preprocessed_split(
        create_split(
            "train",
            100,
            1,
        ),
        expected_split="train",
    )

    validation_split = load_preprocessed_split(
        create_split(
            "validation",
            30,
            1001,
        ),
        expected_split="validation",
    )

    candidates = [
        MergeDelayCandidate(
            candidate_id="candidate_a",
            model_family="logistic_regression",
            model=LogisticRegression(
                C=0.10,
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            ),
            parameters={
                "C": 0.10,
            },
        ),
        MergeDelayCandidate(
            candidate_id="candidate_b",
            model_family="logistic_regression",
            model=LogisticRegression(
                C=1.00,
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            ),
            parameters={
                "C": 1.00,
            },
        ),
    ]

    evaluated = [
        evaluate_candidate(
            candidate=candidate,
            training_split=training_split,
            validation_split=validation_split,
        )
        for candidate in candidates
    ]

    comparison = build_candidate_comparison(evaluated)

    locked = select_locked_candidate(
        evaluated_candidates=evaluated,
        comparison=comparison,
    )

    assert len(comparison) == 2

    assert comparison["selected_for_locking"].sum() == 1

    assert locked.candidate.candidate_id in {
        "candidate_a",
        "candidate_b",
    }


def test_family_summary() -> None:
    """Confirm one strongest row is retained per family."""

    comparison = pd.DataFrame(
        {
            "validation_rank": [
                1,
                2,
                3,
            ],
            "candidate_id": [
                "logistic_1",
                "forest_1",
                "logistic_2",
            ],
            "model_family": [
                "logistic_regression",
                "random_forest",
                "logistic_regression",
            ],
        }
    )

    summary = build_family_summary(comparison)

    assert len(summary) == 2

    assert set(summary["candidate_id"]) == {
        "logistic_1",
        "forest_1",
    }
