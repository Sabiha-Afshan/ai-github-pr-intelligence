"""Evaluation utilities for Model 2 merge-delay prediction."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_metrics(
    actual: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate binary classification metrics at one threshold."""

    actual_array = np.asarray(
        actual,
        dtype=int,
    )

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if actual_array.shape[0] != probability_array.shape[0]:
        raise ValueError(
            "Actual targets and probabilities have different lengths."
        )

    if not np.isfinite(
        probability_array
    ).all():
        raise ValueError(
            "Probabilities contain missing or infinite values."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between zero and one."
        )

    predictions = (
        probability_array
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    clipped_probabilities = np.clip(
        probability_array,
        1e-15,
        1 - 1e-15,
    )

    return {
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy_score(
                actual_array,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                actual_array,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                actual_array,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual_array,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                actual_array,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                actual_array,
                probability_array,
            )
        ),
        "average_precision": float(
            average_precision_score(
                actual_array,
                probability_array,
            )
        ),
        "log_loss": float(
            log_loss(
                actual_array,
                clipped_probabilities,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                actual_array,
                probability_array,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_positive_count": int(
            predictions.sum()
        ),
        "predicted_negative_count": int(
            len(predictions)
            - predictions.sum()
        ),
    }


def build_threshold_table(
    actual: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    minimum_threshold: float = 0.20,
    maximum_threshold: float = 0.80,
    threshold_step: float = 0.025,
) -> pd.DataFrame:
    """Evaluate a controlled range of validation thresholds."""

    if threshold_step <= 0:
        raise ValueError(
            "Threshold step must be positive."
        )

    thresholds = np.arange(
        minimum_threshold,
        maximum_threshold
        + threshold_step / 2,
        threshold_step,
    )

    records = [
        calculate_binary_metrics(
            actual=actual,
            probabilities=probabilities,
            threshold=float(
                round(
                    threshold,
                    6,
                )
            ),
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(
        records
    )


def select_validation_threshold(
    threshold_table: pd.DataFrame,
) -> dict[str, Any]:
    """
    Select the validation threshold.

    Primary criterion:
        balanced accuracy

    Tie-breakers:
        F1 score
        recall
        threshold closest to 0.50
        lower threshold
    """

    required_columns = {
        "threshold",
        "balanced_accuracy",
        "f1",
        "recall",
    }

    missing_columns = sorted(
        required_columns
        - set(
            threshold_table.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Threshold table is missing columns: "
            f"{missing_columns}"
        )

    working = threshold_table.copy()

    working[
        "distance_from_half"
    ] = (
        working[
            "threshold"
        ]
        - 0.50
    ).abs()

    selected = (
        working.sort_values(
            [
                "balanced_accuracy",
                "f1",
                "recall",
                "distance_from_half",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],
        )
        .iloc[0]
        .drop(
            labels=[
                "distance_from_half"
            ]
        )
        .to_dict()
    )

    return {
        key: (
            value.item()
            if hasattr(
                value,
                "item",
            )
            else value
        )
        for key, value
        in selected.items()
    }


def build_prediction_table(
    identifiers: pd.Series,
    actual: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Create row-level prediction results."""

    identifier_values = (
        identifiers.reset_index(
            drop=True
        )
    )

    actual_values = (
        actual.astype(int)
        .reset_index(
            drop=True
        )
    )

    probability_values = np.asarray(
        probabilities,
        dtype=float,
    )

    if not (
        len(identifier_values)
        == len(actual_values)
        == len(probability_values)
    ):
        raise ValueError(
            "Prediction inputs are not aligned."
        )

    predicted_values = (
        probability_values
        >= threshold
    ).astype(int)

    confidence_values = np.where(
        predicted_values == 1,
        probability_values,
        1 - probability_values,
    )

    prediction_table = pd.DataFrame(
        {
            "model_name": model_name,
            "split": split_name,
            "pr_number": identifier_values,
            "actual_target": actual_values,
            "predicted_target": predicted_values,
            "delay_probability": probability_values,
            "prediction_confidence": confidence_values,
            "selected_threshold": threshold,
        }
    )

    prediction_table[
        "prediction_correct"
    ] = (
        prediction_table[
            "actual_target"
        ]
        == prediction_table[
            "predicted_target"
        ]
    )

    prediction_table[
        "prediction_outcome"
    ] = np.select(
        [
            (
                prediction_table[
                    "actual_target"
                ]
                == 1
            )
            & (
                prediction_table[
                    "predicted_target"
                ]
                == 1
            ),
            (
                prediction_table[
                    "actual_target"
                ]
                == 0
            )
            & (
                prediction_table[
                    "predicted_target"
                ]
                == 0
            ),
            (
                prediction_table[
                    "actual_target"
                ]
                == 0
            )
            & (
                prediction_table[
                    "predicted_target"
                ]
                == 1
            ),
        ],
        [
            "true_positive",
            "true_negative",
            "false_positive",
        ],
        default="false_negative",
    )

    return prediction_table


def compare_validation_models(
    metric_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a ranked validation comparison table."""

    if not metric_records:
        raise ValueError(
            "No model metrics were supplied."
        )

    comparison = pd.DataFrame(
        metric_records
    )

    required_columns = {
        "model_name",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "average_precision",
    }

    missing_columns = sorted(
        required_columns
        - set(
            comparison.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Model comparison is missing columns: "
            f"{missing_columns}"
        )

    comparison = comparison.sort_values(
        [
            "balanced_accuracy",
            "f1",
            "roc_auc",
            "average_precision",
            "model_name",
        ],
        ascending=[
            False,
            False,
            False,
            False,
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
        "nominated_candidate"
    ] = (
        comparison[
            "validation_rank"
        ]
        == 1
    )

    return comparison