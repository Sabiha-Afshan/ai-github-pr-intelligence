"""Controlled tuning utilities for the contributor-neutral Random Forest."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.models.merge_outcome_evaluation import (
    build_threshold_table,
    calculate_metrics,
    create_prediction_table,
    get_positive_probability,
    select_validation_threshold,
)


RANDOM_STATE = 42

FINAL_MODEL_NAME = (
    "contributor_neutral_random_forest_tuned"
)


RAW_LOG_EQUIVALENTS = {
    "body_length": "log1p_body_length",
    "body_word_count": "log1p_body_word_count",
    "additions": "log1p_additions",
    "deletions": "log1p_deletions",
    "total_changes": "log1p_total_changes",
    "changed_files": "log1p_changed_files",
    "commit_count": "log1p_commit_count",
}


@dataclass(frozen=True)
class TuningCandidateResult:
    """Results for one feature-set and parameter candidate."""

    candidate_id: str
    feature_set_name: str
    feature_names: list[str]
    parameters: dict[str, Any]
    model: RandomForestClassifier
    validation_probabilities: np.ndarray
    selected_threshold: float
    validation_metrics: dict[str, Any]
    training_metrics: dict[str, Any]
    threshold_table: pd.DataFrame
    training_seconds: float


def is_outlier_feature(
    feature_name: str,
) -> bool:
    """Return whether a feature is an IQR-derived indicator."""

    return (
        feature_name.endswith(
            "_iqr_outlier"
        )
        or feature_name
        in {
            "iqr_outlier_feature_count",
            "has_any_iqr_outlier",
        }
    )


def build_feature_sets(
    available_features: list[str],
) -> dict[str, list[str]]:
    """Create controlled feature groups for tuning."""

    available_set = set(
        available_features
    )

    full_features = list(
        available_features
    )

    core_raw_features = [
        feature
        for feature in available_features
        if not is_outlier_feature(
            feature
        )
        and not feature.startswith(
            "log1p_"
        )
    ]

    raw_features_to_remove = {
        raw_feature
        for raw_feature, log_feature
        in RAW_LOG_EQUIVALENTS.items()
        if (
            raw_feature
            in available_set
            and log_feature
            in available_set
        )
    }

    core_log_features = [
        feature
        for feature in available_features
        if not is_outlier_feature(
            feature
        )
        and feature
        not in raw_features_to_remove
    ]

    feature_sets = {
        "full_approved": (
            full_features
        ),
        "core_raw": (
            core_raw_features
        ),
        "core_log": (
            core_log_features
        ),
    }

    for (
        feature_set_name,
        feature_names,
    ) in feature_sets.items():
        if not feature_names:
            raise ValueError(
                f"Feature set {feature_set_name!r} "
                "contains no features."
            )

        if len(
            feature_names
        ) != len(
            set(feature_names)
        ):
            raise ValueError(
                f"Feature set {feature_set_name!r} "
                "contains duplicates."
            )

    return feature_sets


def build_parameter_candidates() -> list[
    dict[str, Any]
]:
    """Return a limited, analytically justified parameter search."""

    return [
        {
            "configuration_name": (
                "shallow_regularized"
            ),
            "n_estimators": 750,
            "max_depth": 6,
            "min_samples_split": 12,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "max_samples": 0.80,
        },
        {
            "configuration_name": (
                "medium_regularized"
            ),
            "n_estimators": 750,
            "max_depth": 10,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "max_samples": 0.85,
        },
        {
            "configuration_name": (
                "medium_wider_features"
            ),
            "n_estimators": 750,
            "max_depth": 10,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "max_features": 0.50,
            "class_weight": "balanced",
            "max_samples": 0.85,
        },
        {
            "configuration_name": (
                "deep_regularized"
            ),
            "n_estimators": 1000,
            "max_depth": 16,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "max_samples": 0.90,
        },
        {
            "configuration_name": (
                "unlimited_leaf_regularized"
            ),
            "n_estimators": 1000,
            "max_depth": None,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "max_samples": 0.90,
        },
        {
            "configuration_name": (
                "balanced_subsample"
            ),
            "n_estimators": 1000,
            "max_depth": 12,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "class_weight": (
                "balanced_subsample"
            ),
            "max_samples": 0.85,
        },
        {
            "configuration_name": (
                "conservative_leaf"
            ),
            "n_estimators": 1000,
            "max_depth": 10,
            "min_samples_split": 16,
            "min_samples_leaf": 8,
            "max_features": 0.50,
            "class_weight": "balanced",
            "max_samples": 0.90,
        },
        {
            "configuration_name": (
                "wide_feature_sampling"
            ),
            "n_estimators": 1000,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "max_features": 0.75,
            "class_weight": "balanced",
            "max_samples": 0.85,
        },
    ]


def build_random_forest(
    parameters: dict[str, Any],
) -> RandomForestClassifier:
    """Build one Random Forest candidate."""

    model_parameters = {
        key: value
        for key, value
        in parameters.items()
        if key
        != "configuration_name"
    }

    return RandomForestClassifier(
        bootstrap=True,
        criterion="gini",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **model_parameters,
    )


def train_tuning_candidate(
    candidate_id: str,
    feature_set_name: str,
    feature_names: list[str],
    parameters: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> TuningCandidateResult:
    """Train and validate one controlled tuning candidate."""

    missing_train_features = sorted(
        set(feature_names)
        - set(x_train.columns)
    )

    missing_validation_features = sorted(
        set(feature_names)
        - set(x_validation.columns)
    )

    if missing_train_features:
        raise ValueError(
            "Training data is missing features: "
            f"{missing_train_features}"
        )

    if missing_validation_features:
        raise ValueError(
            "Validation data is missing features: "
            f"{missing_validation_features}"
        )

    model = build_random_forest(
        parameters
    )

    training_started = perf_counter()

    model.fit(
        x_train[feature_names],
        y_train,
    )

    training_seconds = (
        perf_counter()
        - training_started
    )

    validation_probabilities = (
        get_positive_probability(
            model,
            x_validation[
                feature_names
            ],
        )
    )

    threshold_table = (
        build_threshold_table(
            targets=y_validation,
            probabilities=(
                validation_probabilities
            ),
            minimum_threshold=0.20,
            maximum_threshold=0.80,
            threshold_step=0.025,
        )
    )

    selected_metrics = (
        select_validation_threshold(
            threshold_table
        )
    )

    selected_threshold = float(
        selected_metrics[
            "threshold"
        ]
    )

    validation_metrics = dict(
        selected_metrics
    )

    validation_metrics.update(
        {
            "candidate_id": candidate_id,
            "feature_set_name": (
                feature_set_name
            ),
            "configuration_name": (
                parameters[
                    "configuration_name"
                ]
            ),
            "feature_count": len(
                feature_names
            ),
            "training_seconds": float(
                training_seconds
            ),
        }
    )

    training_probabilities = (
        get_positive_probability(
            model,
            x_train[
                feature_names
            ],
        )
    )

    training_metrics = calculate_metrics(
        targets=y_train,
        probabilities=(
            training_probabilities
        ),
        threshold=selected_threshold,
    )

    return TuningCandidateResult(
        candidate_id=candidate_id,
        feature_set_name=(
            feature_set_name
        ),
        feature_names=feature_names,
        parameters=parameters,
        model=model,
        validation_probabilities=(
            validation_probabilities
        ),
        selected_threshold=(
            selected_threshold
        ),
        validation_metrics=(
            validation_metrics
        ),
        training_metrics=(
            training_metrics
        ),
        threshold_table=(
            threshold_table
        ),
        training_seconds=float(
            training_seconds
        ),
    )


def tune_neutral_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    dict[str, TuningCandidateResult],
    pd.DataFrame,
]:
    """Train the controlled candidate collection."""

    if list(
        x_train.columns
    ) != list(
        x_validation.columns
    ):
        raise ValueError(
            "Training and validation columns differ."
        )

    feature_sets = build_feature_sets(
        list(x_train.columns)
    )

    parameter_candidates = (
        build_parameter_candidates()
    )

    results: dict[
        str,
        TuningCandidateResult,
    ] = {}

    comparison_records = []

    for (
        feature_set_name,
        feature_names,
    ) in feature_sets.items():
        for (
            candidate_position,
            parameters,
        ) in enumerate(
            parameter_candidates,
            start=1,
        ):
            candidate_id = (
                f"{feature_set_name}"
                f"__candidate_"
                f"{candidate_position:02d}"
            )

            result = train_tuning_candidate(
                candidate_id=candidate_id,
                feature_set_name=(
                    feature_set_name
                ),
                feature_names=(
                    feature_names
                ),
                parameters=parameters,
                x_train=x_train,
                y_train=y_train,
                x_validation=(
                    x_validation
                ),
                y_validation=(
                    y_validation
                ),
            )

            results[
                candidate_id
            ] = result

            validation_record = dict(
                result.validation_metrics
            )

            validation_record.update(
                {
                    "train_accuracy": (
                        result.training_metrics[
                            "accuracy"
                        ]
                    ),
                    "train_f1": (
                        result.training_metrics[
                            "f1"
                        ]
                    ),
                    "train_roc_auc": (
                        result.training_metrics[
                            "roc_auc"
                        ]
                    ),
                    "accuracy_gap": (
                        result.training_metrics[
                            "accuracy"
                        ]
                        - validation_record[
                            "accuracy"
                        ]
                    ),
                    "roc_auc_gap": (
                        result.training_metrics[
                            "roc_auc"
                        ]
                        - validation_record[
                            "roc_auc"
                        ]
                    ),
                }
            )

            comparison_records.append(
                validation_record
            )

    comparison = pd.DataFrame(
        comparison_records
    )

    comparison = comparison.sort_values(
        [
            "roc_auc",
            "balanced_accuracy",
            "f1",
            "roc_auc_gap",
            "feature_count",
        ],
        ascending=[
            False,
            False,
            False,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    comparison.insert(
        0,
        "validation_rank",
        range(
            1,
            len(comparison) + 1,
        ),
    )

    comparison[
        "selected_candidate"
    ] = (
        comparison[
            "validation_rank"
        ]
        == 1
    )

    return (
        results,
        comparison,
    )


def get_selected_candidate(
    results: dict[
        str,
        TuningCandidateResult,
    ],
    comparison: pd.DataFrame,
) -> TuningCandidateResult:
    """Return the highest-ranked validation candidate."""

    selected_rows = comparison[
        comparison[
            "selected_candidate"
        ]
    ]

    if len(selected_rows) != 1:
        raise ValueError(
            "Exactly one tuning candidate must be selected."
        )

    candidate_id = str(
        selected_rows.iloc[0][
            "candidate_id"
        ]
    )

    if candidate_id not in results:
        raise KeyError(
            "Selected candidate is missing "
            "from fitted results."
        )

    return results[
        candidate_id
    ]


def calculate_permutation_importance(
    selected_result: TuningCandidateResult,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    repeats: int = 30,
) -> pd.DataFrame:
    """Calculate validation permutation importance."""

    importance = permutation_importance(
        estimator=selected_result.model,
        X=x_validation[
            selected_result.feature_names
        ],
        y=y_validation,
        scoring="roc_auc",
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_table = pd.DataFrame(
        {
            "feature": (
                selected_result.feature_names
            ),
            "importance_mean": (
                importance.importances_mean
            ),
            "importance_std": (
                importance.importances_std
            ),
            "positive_importance": (
                importance.importances_mean
                > 0
            ),
        }
    )

    return importance_table.sort_values(
        [
            "importance_mean",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def create_selected_predictions(
    selected_result: TuningCandidateResult,
    validation_identifiers: pd.Series,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """Create row-level predictions for the selected candidate."""

    return create_prediction_table(
        identifiers=(
            validation_identifiers
        ),
        targets=y_validation,
        probabilities=(
            selected_result
            .validation_probabilities
        ),
        threshold=(
            selected_result
            .selected_threshold
        ),
        model_name=FINAL_MODEL_NAME,
    )


def validate_tuning_results(
    results: dict[
        str,
        TuningCandidateResult,
    ],
    comparison: pd.DataFrame,
    expected_validation_rows: int,
) -> dict[str, Any]:
    """Validate the controlled tuning outputs."""

    expected_candidate_count = (
        len(
            build_feature_sets(
                list(
                    next(
                        iter(
                            results.values()
                        )
                    ).model.feature_names_in_
                )
            )
        )
        * len(
            build_parameter_candidates()
        )
        if results
        else 0
    )

    selected_count = int(
        comparison[
            "selected_candidate"
        ].sum()
    )

    metrics_valid = bool(
        comparison[
            [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
            ]
        ]
        .apply(
            lambda column: (
                column.between(
                    0,
                    1,
                    inclusive="both",
                ).all()
            )
        )
        .all()
    )

    probability_counts_valid = all(
        len(
            result
            .validation_probabilities
        )
        == expected_validation_rows
        for result in results.values()
    )

    thresholds_valid = all(
        0
        < result.selected_threshold
        < 1
        for result in results.values()
    )

    validation_passed = (
        len(results)
        == len(comparison)
        and selected_count == 1
        and metrics_valid
        and probability_counts_valid
        and thresholds_valid
    )

    return {
        "candidate_count": len(
            results
        ),
        "comparison_row_count": len(
            comparison
        ),
        "expected_candidate_count": (
            expected_candidate_count
        ),
        "selected_candidate_count": (
            selected_count
        ),
        "metrics_valid": (
            metrics_valid
        ),
        "probability_counts_valid": (
            probability_counts_valid
        ),
        "thresholds_valid": (
            thresholds_valid
        ),
        "validation_passed": (
            validation_passed
        ),
    }