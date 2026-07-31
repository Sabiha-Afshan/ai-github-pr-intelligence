"""Histogram Gradient Boosting challenger for merge-outcome prediction."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.models.merge_outcome_evaluation import (
    build_threshold_table,
    calculate_metrics,
    create_prediction_table,
    get_positive_probability,
    select_validation_threshold,
)
from src.models.merge_outcome_tuning import build_feature_sets


RANDOM_STATE = 42

CHALLENGER_MODEL_NAME = (
    "contributor_neutral_hist_gradient_boosting"
)


@dataclass(frozen=True)
class CrossValidationCandidate:
    """Cross-validation result for one challenger configuration."""

    candidate_id: str
    feature_set_name: str
    feature_names: list[str]
    parameters: dict[str, Any]
    fold_scores: list[float]
    mean_cv_roc_auc: float
    std_cv_roc_auc: float
    minimum_cv_roc_auc: float
    valid_fold_count: int
    total_fold_count: int
    training_seconds: float


@dataclass(frozen=True)
class ChallengerResult:
    """Fitted challenger and its validation results."""

    model_name: str
    selected_candidate_id: str
    feature_set_name: str
    feature_names: list[str]
    parameters: dict[str, Any]
    model: HistGradientBoostingClassifier
    selected_threshold: float
    validation_probabilities: np.ndarray
    validation_metrics: dict[str, Any]
    training_metrics: dict[str, Any]
    threshold_table: pd.DataFrame
    training_seconds: float


def build_challenger_parameter_candidates() -> list[dict[str, Any]]:
    """Return a small, controlled Gradient Boosting search space."""

    return [
        {
            "configuration_name": "shallow_conservative",
            "learning_rate": 0.04,
            "max_iter": 250,
            "max_leaf_nodes": 15,
            "max_depth": 4,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        {
            "configuration_name": "shallow_regularized",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "max_depth": 5,
            "min_samples_leaf": 15,
            "l2_regularization": 2.0,
        },
        {
            "configuration_name": "medium_balanced",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "max_depth": 6,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        {
            "configuration_name": "medium_regularized",
            "learning_rate": 0.04,
            "max_iter": 350,
            "max_leaf_nodes": 31,
            "max_depth": 6,
            "min_samples_leaf": 25,
            "l2_regularization": 3.0,
        },
        {
            "configuration_name": "small_leaf_model",
            "learning_rate": 0.03,
            "max_iter": 400,
            "max_leaf_nodes": 12,
            "max_depth": 4,
            "min_samples_leaf": 12,
            "l2_regularization": 2.0,
        },
        {
            "configuration_name": "strong_regularization",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 20,
            "max_depth": 5,
            "min_samples_leaf": 30,
            "l2_regularization": 5.0,
        },
    ]


def build_challenger_model(
    parameters: dict[str, Any],
) -> HistGradientBoostingClassifier:
    """Build one Histogram Gradient Boosting classifier."""

    model_parameters = {
        key: value
        for key, value in parameters.items()
        if key != "configuration_name"
    }

    return HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=RANDOM_STATE,
        **model_parameters,
    )


def build_temporal_folds(
    row_count: int,
    split_count: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window temporal folds."""

    if row_count < 100:
        raise ValueError(
            "At least 100 chronologically ordered rows are required."
        )

    splitter = TimeSeriesSplit(
        n_splits=split_count,
    )

    placeholder = np.arange(
        row_count
    )

    return list(
        splitter.split(
            placeholder
        )
    )


def evaluate_cross_validation_candidate(
    candidate_id: str,
    feature_set_name: str,
    feature_names: list[str],
    parameters: dict[str, Any],
    features: pd.DataFrame,
    targets: pd.Series,
    temporal_folds: list[
        tuple[np.ndarray, np.ndarray]
    ],
) -> CrossValidationCandidate:
    """Evaluate one candidate using temporal cross-validation."""

    missing_features = sorted(
        set(feature_names)
        - set(features.columns)
    )

    if missing_features:
        raise ValueError(
            "Training data is missing challenger features: "
            f"{missing_features}"
        )

    selected_features = features[
        feature_names
    ].reset_index(drop=True)

    selected_targets = targets.reset_index(
        drop=True
    ).astype(int)

    fold_scores: list[float] = []

    training_started = perf_counter()

    for train_indices, validation_indices in temporal_folds:
        fold_train_targets = selected_targets.iloc[
            train_indices
        ]

        fold_validation_targets = selected_targets.iloc[
            validation_indices
        ]

        if (
            fold_train_targets.nunique()
            < 2
            or fold_validation_targets.nunique()
            < 2
        ):
            continue

        fold_model = build_challenger_model(
            parameters
        )

        fold_model.fit(
            selected_features.iloc[
                train_indices
            ],
            fold_train_targets,
        )

        fold_probabilities = (
            get_positive_probability(
                fold_model,
                selected_features.iloc[
                    validation_indices
                ],
            )
        )

        fold_score = roc_auc_score(
            fold_validation_targets,
            fold_probabilities,
        )

        fold_scores.append(
            float(fold_score)
        )

    training_seconds = (
        perf_counter()
        - training_started
    )

    if len(fold_scores) < 3:
        raise ValueError(
            f"{candidate_id} produced fewer than three valid "
            "temporal cross-validation folds."
        )

    return CrossValidationCandidate(
        candidate_id=candidate_id,
        feature_set_name=feature_set_name,
        feature_names=feature_names,
        parameters=parameters,
        fold_scores=fold_scores,
        mean_cv_roc_auc=float(
            np.mean(fold_scores)
        ),
        std_cv_roc_auc=float(
            np.std(
                fold_scores,
                ddof=0,
            )
        ),
        minimum_cv_roc_auc=float(
            np.min(fold_scores)
        ),
        valid_fold_count=len(
            fold_scores
        ),
        total_fold_count=len(
            temporal_folds
        ),
        training_seconds=float(
            training_seconds
        ),
    )


def run_challenger_cross_validation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    dict[str, CrossValidationCandidate],
    pd.DataFrame,
]:
    """Run controlled temporal cross-validation."""

    if len(x_train) != len(y_train):
        raise ValueError(
            "Training features and targets are misaligned."
        )

    feature_sets = build_feature_sets(
        list(x_train.columns)
    )

    parameter_candidates = (
        build_challenger_parameter_candidates()
    )

    temporal_folds = build_temporal_folds(
        row_count=len(x_train),
        split_count=5,
    )

    results: dict[
        str,
        CrossValidationCandidate,
    ] = {}

    records: list[dict[str, Any]] = []

    for feature_set_name, feature_names in feature_sets.items():
        for position, parameters in enumerate(
            parameter_candidates,
            start=1,
        ):
            candidate_id = (
                f"{feature_set_name}"
                f"__hist_candidate_{position:02d}"
            )

            result = (
                evaluate_cross_validation_candidate(
                    candidate_id=candidate_id,
                    feature_set_name=feature_set_name,
                    feature_names=feature_names,
                    parameters=parameters,
                    features=x_train,
                    targets=y_train,
                    temporal_folds=temporal_folds,
                )
            )

            results[
                candidate_id
            ] = result

            records.append(
                {
                    "candidate_id": candidate_id,
                    "feature_set_name": feature_set_name,
                    "configuration_name": parameters[
                        "configuration_name"
                    ],
                    "feature_count": len(
                        feature_names
                    ),
                    "mean_cv_roc_auc": (
                        result.mean_cv_roc_auc
                    ),
                    "std_cv_roc_auc": (
                        result.std_cv_roc_auc
                    ),
                    "minimum_cv_roc_auc": (
                        result.minimum_cv_roc_auc
                    ),
                    "valid_fold_count": (
                        result.valid_fold_count
                    ),
                    "total_fold_count": (
                        result.total_fold_count
                    ),
                    "training_seconds": (
                        result.training_seconds
                    ),
                    "fold_scores": json_safe_fold_scores(
                        result.fold_scores
                    ),
                }
            )

    comparison = pd.DataFrame(
        records
    )

    comparison = comparison.sort_values(
        [
            "mean_cv_roc_auc",
            "std_cv_roc_auc",
            "minimum_cv_roc_auc",
            "feature_count",
        ],
        ascending=[
            False,
            True,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    comparison.insert(
        0,
        "cv_rank",
        range(
            1,
            len(comparison) + 1,
        ),
    )

    comparison[
        "selected_candidate"
    ] = (
        comparison["cv_rank"]
        == 1
    )

    return (
        results,
        comparison,
    )


def json_safe_fold_scores(
    fold_scores: list[float],
) -> str:
    """Convert fold scores into a stable text representation."""

    return ", ".join(
        f"{score:.6f}"
        for score in fold_scores
    )


def get_selected_cv_candidate(
    results: dict[
        str,
        CrossValidationCandidate,
    ],
    comparison: pd.DataFrame,
) -> CrossValidationCandidate:
    """Return the top-ranked temporal CV candidate."""

    selected_rows = comparison.loc[
        comparison[
            "selected_candidate"
        ]
    ]

    if len(selected_rows) != 1:
        raise ValueError(
            "Exactly one challenger candidate must be selected."
        )

    candidate_id = str(
        selected_rows.iloc[0][
            "candidate_id"
        ]
    )

    if candidate_id not in results:
        raise KeyError(
            "Selected challenger is missing from the fitted results."
        )

    return results[
        candidate_id
    ]


def fit_selected_challenger(
    selected_candidate: CrossValidationCandidate,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> ChallengerResult:
    """Fit the selected challenger and evaluate on validation data."""

    feature_names = (
        selected_candidate.feature_names
    )

    model = build_challenger_model(
        selected_candidate.parameters
    )

    training_started = perf_counter()

    model.fit(
        x_train[
            feature_names
        ],
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
            "model_name": CHALLENGER_MODEL_NAME,
            "selected_candidate_id": (
                selected_candidate
                .candidate_id
            ),
            "feature_set_name": (
                selected_candidate
                .feature_set_name
            ),
            "feature_count": len(
                feature_names
            ),
            "mean_temporal_cv_roc_auc": (
                selected_candidate
                .mean_cv_roc_auc
            ),
            "std_temporal_cv_roc_auc": (
                selected_candidate
                .std_cv_roc_auc
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
        probabilities=training_probabilities,
        threshold=selected_threshold,
    )

    return ChallengerResult(
        model_name=CHALLENGER_MODEL_NAME,
        selected_candidate_id=(
            selected_candidate
            .candidate_id
        ),
        feature_set_name=(
            selected_candidate
            .feature_set_name
        ),
        feature_names=feature_names,
        parameters=(
            selected_candidate.parameters
        ),
        model=model,
        selected_threshold=(
            selected_threshold
        ),
        validation_probabilities=(
            validation_probabilities
        ),
        validation_metrics=(
            validation_metrics
        ),
        training_metrics=(
            training_metrics
        ),
        threshold_table=threshold_table,
        training_seconds=float(
            training_seconds
        ),
    )


def create_challenger_predictions(
    result: ChallengerResult,
    validation_identifiers: pd.Series,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """Create row-level challenger validation predictions."""

    return create_prediction_table(
        identifiers=validation_identifiers,
        targets=y_validation,
        probabilities=(
            result.validation_probabilities
        ),
        threshold=(
            result.selected_threshold
        ),
        model_name=(
            result.model_name
        ),
    )


def compare_challenger_with_random_forest(
    challenger_metrics: dict[str, Any],
    random_forest_metrics: dict[str, Any],
    minimum_roc_auc_improvement: float = 0.02,
) -> dict[str, Any]:
    """Determine whether the challenger should replace Random Forest."""

    challenger_roc_auc = float(
        challenger_metrics[
            "roc_auc"
        ]
    )

    forest_roc_auc = float(
        random_forest_metrics[
            "roc_auc"
        ]
    )

    challenger_balanced_accuracy = float(
        challenger_metrics[
            "balanced_accuracy"
        ]
    )

    forest_balanced_accuracy = float(
        random_forest_metrics[
            "balanced_accuracy"
        ]
    )

    challenger_f1 = float(
        challenger_metrics[
            "f1"
        ]
    )

    forest_f1 = float(
        random_forest_metrics[
            "f1"
        ]
    )

    roc_auc_change = (
        challenger_roc_auc
        - forest_roc_auc
    )

    balanced_accuracy_change = (
        challenger_balanced_accuracy
        - forest_balanced_accuracy
    )

    f1_change = (
        challenger_f1
        - forest_f1
    )

    meaningful_roc_auc_improvement = (
        roc_auc_change
        >= minimum_roc_auc_improvement
    )

    strong_secondary_improvement = (
        balanced_accuracy_change
        >= 0.02
        and f1_change >= 0
        and roc_auc_change >= 0
    )

    challenger_selected = (
        meaningful_roc_auc_improvement
        or strong_secondary_improvement
    )

    selected_model_name = (
        CHALLENGER_MODEL_NAME
        if challenger_selected
        else (
            "contributor_neutral_"
            "random_forest_tuned"
        )
    )

    return {
        "random_forest_roc_auc": (
            forest_roc_auc
        ),
        "challenger_roc_auc": (
            challenger_roc_auc
        ),
        "roc_auc_change": float(
            roc_auc_change
        ),
        "random_forest_balanced_accuracy": (
            forest_balanced_accuracy
        ),
        "challenger_balanced_accuracy": (
            challenger_balanced_accuracy
        ),
        "balanced_accuracy_change": float(
            balanced_accuracy_change
        ),
        "random_forest_f1": (
            forest_f1
        ),
        "challenger_f1": (
            challenger_f1
        ),
        "f1_change": float(
            f1_change
        ),
        "minimum_required_roc_auc_improvement": (
            minimum_roc_auc_improvement
        ),
        "meaningful_roc_auc_improvement": (
            meaningful_roc_auc_improvement
        ),
        "strong_secondary_improvement": (
            strong_secondary_improvement
        ),
        "challenger_selected": (
            challenger_selected
        ),
        "selected_final_model": (
            selected_model_name
        ),
    }


def validate_challenger_outputs(
    cv_results: dict[
        str,
        CrossValidationCandidate,
    ],
    cv_comparison: pd.DataFrame,
    challenger_result: ChallengerResult,
    expected_validation_rows: int,
) -> dict[str, Any]:
    """Validate the complete challenger experiment."""

    expected_candidate_count = (
        3
        * len(
            build_challenger_parameter_candidates()
        )
    )

    actual_candidate_count = len(
        cv_results
    )

    selected_candidate_count = int(
        cv_comparison[
            "selected_candidate"
        ].sum()
    )

    probabilities = (
        challenger_result
        .validation_probabilities
    )

    probabilities_valid = bool(
        np.isfinite(
            probabilities
        ).all()
        and (
            (
                probabilities >= 0
            )
            & (
                probabilities <= 1
            )
        ).all()
    )

    metrics_valid = all(
        0
        <= float(
            challenger_result
            .validation_metrics[
                metric_name
            ]
        )
        <= 1
        for metric_name in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        )
    )

    threshold_valid = (
        0
        < challenger_result
        .selected_threshold
        < 1
    )

    validation_passed = (
        actual_candidate_count
        == expected_candidate_count
        and actual_candidate_count
        == len(
            cv_comparison
        )
        and selected_candidate_count
        == 1
        and len(
            probabilities
        )
        == expected_validation_rows
        and probabilities_valid
        and metrics_valid
        and threshold_valid
    )

    return {
        "candidate_count": (
            actual_candidate_count
        ),
        "comparison_row_count": len(
            cv_comparison
        ),
        "expected_candidate_count": (
            expected_candidate_count
        ),
        "selected_candidate_count": (
            selected_candidate_count
        ),
        "validation_probability_count": len(
            probabilities
        ),
        "expected_validation_rows": (
            expected_validation_rows
        ),
        "probabilities_valid": (
            probabilities_valid
        ),
        "metrics_valid": (
            metrics_valid
        ),
        "threshold_valid": (
            threshold_valid
        ),
        "validation_passed": (
            validation_passed
        ),
    }