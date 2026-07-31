"""Controlled candidate selection for Model 2 merge-delay prediction."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from src.models.merge_delay_evaluation import (
    build_prediction_table,
    build_threshold_table,
    calculate_binary_metrics,
    select_validation_threshold,
)
from src.models.merge_delay_training import Model2Split


@dataclass(frozen=True)
class MergeDelayCandidate:
    """One controlled Model 2 candidate configuration."""

    candidate_id: str
    model_family: str
    model: Any
    parameters: dict[str, Any]
    use_balanced_sample_weight: bool = False


@dataclass(frozen=True)
class EvaluatedMergeDelayCandidate:
    """One trained candidate and its validation evidence."""

    candidate: MergeDelayCandidate
    trained_model: Any
    selected_threshold: float
    training_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    threshold_table: pd.DataFrame
    validation_predictions: pd.DataFrame


def build_logistic_candidates() -> list[MergeDelayCandidate]:
    """Build a small Logistic Regression regularization search."""

    regularization_values = [
        0.01,
        0.03,
        0.10,
        0.30,
        1.00,
        3.00,
        10.00,
    ]

    candidates = []

    for position, regularization_strength in enumerate(
        regularization_values,
        start=1,
    ):
        parameters = {
            "C": regularization_strength,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 3000,
            "random_state": 42,
        }

        candidates.append(
            MergeDelayCandidate(
                candidate_id=(
                    f"logistic_candidate_{position:02d}"
                ),
                model_family="logistic_regression",
                model=LogisticRegression(
                    **parameters
                ),
                parameters=parameters,
            )
        )

    return candidates


def build_random_forest_candidates() -> list[MergeDelayCandidate]:
    """Build a deliberately restricted Random Forest search."""

    configurations = [
        {
            "n_estimators": 600,
            "max_depth": 6,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "max_samples": 0.80,
        },
        {
            "n_estimators": 800,
            "max_depth": 8,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "max_samples": 0.85,
        },
        {
            "n_estimators": 1000,
            "max_depth": 10,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "max_samples": 0.85,
        },
        {
            "n_estimators": 800,
            "max_depth": 12,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "max_features": 0.40,
            "max_samples": 0.85,
        },
        {
            "n_estimators": 1000,
            "max_depth": 8,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": 0.50,
            "max_samples": 0.90,
        },
        {
            "n_estimators": 1200,
            "max_depth": None,
            "min_samples_split": 12,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "max_samples": 0.80,
        },
    ]

    candidates = []

    for position, configuration in enumerate(
        configurations,
        start=1,
    ):
        parameters = {
            **configuration,
            "class_weight": "balanced_subsample",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        }

        candidates.append(
            MergeDelayCandidate(
                candidate_id=(
                    f"random_forest_candidate_{position:02d}"
                ),
                model_family="random_forest",
                model=RandomForestClassifier(
                    **parameters
                ),
                parameters=parameters,
            )
        )

    return candidates


def build_hist_gradient_boosting_candidates(
) -> list[MergeDelayCandidate]:
    """Build one controlled HistGradientBoosting challenger family."""

    configurations = [
        {
            "learning_rate": 0.03,
            "max_iter": 250,
            "max_leaf_nodes": 15,
            "max_depth": 4,
            "min_samples_leaf": 20,
            "l2_regularization": 2.0,
        },
        {
            "learning_rate": 0.03,
            "max_iter": 350,
            "max_leaf_nodes": 31,
            "max_depth": 5,
            "min_samples_leaf": 20,
            "l2_regularization": 3.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 15,
            "max_depth": 5,
            "min_samples_leaf": 20,
            "l2_regularization": 3.0,
        },
        {
            "learning_rate": 0.04,
            "max_iter": 350,
            "max_leaf_nodes": 31,
            "max_depth": 6,
            "min_samples_leaf": 25,
            "l2_regularization": 4.0,
        },
        {
            "learning_rate": 0.06,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "max_depth": 5,
            "min_samples_leaf": 25,
            "l2_regularization": 5.0,
        },
        {
            "learning_rate": 0.03,
            "max_iter": 450,
            "max_leaf_nodes": 15,
            "max_depth": 6,
            "min_samples_leaf": 30,
            "l2_regularization": 5.0,
        },
    ]

    candidates = []

    for position, configuration in enumerate(
        configurations,
        start=1,
    ):
        parameters = {
            **configuration,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 30,
            "random_state": 42,
        }

        candidates.append(
            MergeDelayCandidate(
                candidate_id=(
                    "hist_gradient_boosting_"
                    f"candidate_{position:02d}"
                ),
                model_family=(
                    "hist_gradient_boosting"
                ),
                model=HistGradientBoostingClassifier(
                    **parameters
                ),
                parameters=parameters,
                use_balanced_sample_weight=True,
            )
        )

    return candidates


def build_all_candidates() -> list[MergeDelayCandidate]:
    """Return the complete controlled candidate set."""

    return [
        *build_logistic_candidates(),
        *build_random_forest_candidates(),
        *build_hist_gradient_boosting_candidates(),
    ]


def evaluate_candidate(
    candidate: MergeDelayCandidate,
    training_split: Model2Split,
    validation_split: Model2Split,
) -> EvaluatedMergeDelayCandidate:
    """Train and evaluate one candidate without using the test set."""

    fit_arguments: dict[str, Any] = {}

    if candidate.use_balanced_sample_weight:
        fit_arguments[
            "sample_weight"
        ] = compute_sample_weight(
            class_weight="balanced",
            y=training_split.targets,
        )

    candidate.model.fit(
        training_split.features,
        training_split.targets,
        **fit_arguments,
    )

    training_probabilities = (
        candidate.model.predict_proba(
            training_split.features
        )[:, 1]
    )

    validation_probabilities = (
        candidate.model.predict_proba(
            validation_split.features
        )[:, 1]
    )

    threshold_table = build_threshold_table(
        actual=validation_split.targets,
        probabilities=validation_probabilities,
        minimum_threshold=0.20,
        maximum_threshold=0.80,
        threshold_step=0.025,
    )

    selected_threshold_metrics = (
        select_validation_threshold(
            threshold_table
        )
    )

    selected_threshold = float(
        selected_threshold_metrics[
            "threshold"
        ]
    )

    training_metrics = (
        calculate_binary_metrics(
            actual=training_split.targets,
            probabilities=training_probabilities,
            threshold=selected_threshold,
        )
    )

    validation_metrics = (
        calculate_binary_metrics(
            actual=validation_split.targets,
            probabilities=validation_probabilities,
            threshold=selected_threshold,
        )
    )

    for metrics in (
        training_metrics,
        validation_metrics,
    ):
        metrics[
            "candidate_id"
        ] = candidate.candidate_id

        metrics[
            "model_family"
        ] = candidate.model_family

    validation_predictions = (
        build_prediction_table(
            identifiers=(
                validation_split.identifiers
            ),
            actual=(
                validation_split.targets
            ),
            probabilities=(
                validation_probabilities
            ),
            threshold=selected_threshold,
            model_name=(
                candidate.candidate_id
            ),
            split_name="validation",
        )
    )

    return EvaluatedMergeDelayCandidate(
        candidate=candidate,
        trained_model=candidate.model,
        selected_threshold=(
            selected_threshold
        ),
        training_metrics=(
            training_metrics
        ),
        validation_metrics=(
            validation_metrics
        ),
        threshold_table=(
            threshold_table
        ),
        validation_predictions=(
            validation_predictions
        ),
    )


def build_candidate_comparison(
    evaluated_candidates: list[
        EvaluatedMergeDelayCandidate
    ],
) -> pd.DataFrame:
    """Build and rank the complete validation comparison."""

    if not evaluated_candidates:
        raise ValueError(
            "No evaluated candidates were supplied."
        )

    records = []

    for evaluated in evaluated_candidates:
        validation_metrics = dict(
            evaluated.validation_metrics
        )

        training_metrics = (
            evaluated.training_metrics
        )

        validation_metrics[
            "training_accuracy"
        ] = training_metrics[
            "accuracy"
        ]

        validation_metrics[
            "training_balanced_accuracy"
        ] = training_metrics[
            "balanced_accuracy"
        ]

        validation_metrics[
            "training_roc_auc"
        ] = training_metrics[
            "roc_auc"
        ]

        validation_metrics[
            "roc_auc_gap"
        ] = (
            training_metrics[
                "roc_auc"
            ]
            - validation_metrics[
                "roc_auc"
            ]
        )

        validation_metrics[
            "balanced_accuracy_gap"
        ] = (
            training_metrics[
                "balanced_accuracy"
            ]
            - validation_metrics[
                "balanced_accuracy"
            ]
        )

        validation_metrics[
            "overfitting_warning"
        ] = (
            validation_metrics[
                "roc_auc_gap"
            ]
            > 0.20
        )

        validation_metrics[
            "parameters"
        ] = str(
            evaluated.candidate.parameters
        )

        records.append(
            validation_metrics
        )

    comparison = pd.DataFrame(
        records
    )

    comparison[
        "absolute_distance_from_half"
    ] = (
        comparison[
            "threshold"
        ]
        - 0.50
    ).abs()

    comparison = comparison.sort_values(
        [
            "balanced_accuracy",
            "f1",
            "recall",
            "roc_auc",
            "average_precision",
            "roc_auc_gap",
            "absolute_distance_from_half",
            "candidate_id",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "validation_rank",
        range(
            1,
            len(comparison) + 1,
        ),
    )

    comparison[
        "selected_for_locking"
    ] = (
        comparison[
            "validation_rank"
        ]
        == 1
    )

    return comparison


def select_locked_candidate(
    evaluated_candidates: list[
        EvaluatedMergeDelayCandidate
    ],
    comparison: pd.DataFrame,
) -> EvaluatedMergeDelayCandidate:
    """Return the candidate selected entirely from validation evidence."""

    selected_rows = comparison.loc[
        comparison[
            "selected_for_locking"
        ].astype(bool)
    ]

    if len(selected_rows) != 1:
        raise ValueError(
            "Exactly one candidate must be selected for locking."
        )

    selected_candidate_id = str(
        selected_rows.iloc[0][
            "candidate_id"
        ]
    )

    matching_candidates = [
        candidate
        for candidate in evaluated_candidates
        if (
            candidate.candidate.candidate_id
            == selected_candidate_id
        )
    ]

    if len(matching_candidates) != 1:
        raise ValueError(
            "Selected candidate could not be resolved uniquely."
        )

    return matching_candidates[0]


def build_family_summary(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Return the strongest candidate from each model family."""

    required_columns = {
        "model_family",
        "validation_rank",
    }

    missing_columns = sorted(
        required_columns
        - set(comparison.columns)
    )

    if missing_columns:
        raise ValueError(
            "Comparison is missing columns: "
            f"{missing_columns}"
        )

    family_summary = (
        comparison.sort_values(
            "validation_rank"
        )
        .groupby(
            "model_family",
            observed=False,
            as_index=False,
        )
        .first()
        .sort_values(
            "validation_rank"
        )
        .reset_index(drop=True)
    )

    return family_summary