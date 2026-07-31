"""Evaluation utilities for merge-outcome classifiers."""

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


def get_positive_probability(
    model: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return predicted probabilities for the merged class."""

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The fitted model does not provide predict_proba()."
        )

    probabilities = model.predict_proba(features)

    if probabilities.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional probability matrix."
        )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The fitted model does not contain target class 1."
        )

    positive_index = classes.index(1)

    return np.asarray(
        probabilities[:, positive_index],
        dtype=float,
    )


def calculate_metrics(
    targets: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate binary classification metrics."""

    if not 0 < threshold < 1:
        raise ValueError(
            "Threshold must be between zero and one."
        )

    actual = np.asarray(
        targets,
        dtype=int,
    )

    predicted_probability = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(actual) != len(predicted_probability):
        raise ValueError(
            "Targets and probabilities have different row counts."
        )

    if set(np.unique(actual)) != {0, 1}:
        raise ValueError(
            "Evaluation targets must contain classes 0 and 1."
        )

    predicted = (
        predicted_probability >= threshold
    ).astype(int)

    clipped_probability = np.clip(
        predicted_probability,
        1e-15,
        1 - 1e-15,
    )

    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                actual,
                predicted,
            )
        ),
        "precision": float(
            precision_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                actual,
                predicted_probability,
            )
        ),
        "average_precision": float(
            average_precision_score(
                actual,
                predicted_probability,
            )
        ),
        "log_loss": float(
            log_loss(
                actual,
                clipped_probability,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                actual,
                predicted_probability,
            )
        ),
        "true_negative": int(
            true_negative
        ),
        "false_positive": int(
            false_positive
        ),
        "false_negative": int(
            false_negative
        ),
        "true_positive": int(
            true_positive
        ),
    }


def build_threshold_table(
    targets: pd.Series,
    probabilities: np.ndarray,
    minimum_threshold: float = 0.25,
    maximum_threshold: float = 0.75,
    threshold_step: float = 0.05,
) -> pd.DataFrame:
    """Evaluate candidate classification thresholds."""

    thresholds = np.arange(
        minimum_threshold,
        maximum_threshold + threshold_step / 2,
        threshold_step,
    )

    records = [
        calculate_metrics(
            targets=targets,
            probabilities=probabilities,
            threshold=float(
                round(threshold, 4)
            ),
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(records)


def select_validation_threshold(
    threshold_table: pd.DataFrame,
) -> dict[str, Any]:
    """Select a threshold using validation F1 and balanced accuracy."""

    required_columns = {
        "threshold",
        "f1",
        "balanced_accuracy",
        "precision",
        "recall",
    }

    missing_columns = (
        required_columns
        - set(threshold_table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Threshold table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    ranked = threshold_table.sort_values(
        [
            "f1",
            "balanced_accuracy",
            "recall",
            "precision",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return ranked.iloc[0].to_dict()


def create_prediction_table(
    identifiers: pd.Series,
    targets: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    """Create row-level validation predictions."""

    actual = (
        targets.reset_index(drop=True)
        .astype(int)
    )

    probability = np.asarray(
        probabilities,
        dtype=float,
    )

    predicted = (
        probability >= threshold
    ).astype(int)

    confidence = np.maximum(
        probability,
        1 - probability,
    )

    return pd.DataFrame(
        {
            "model_name": model_name,
            "pr_number": (
                identifiers.reset_index(
                    drop=True
                )
            ),
            "actual_target": actual,
            "predicted_target": predicted,
            "merge_probability": probability,
            "prediction_confidence": confidence,
            "prediction_correct": (
                predicted
                == actual.to_numpy()
            ),
            "selected_threshold": threshold,
        }
    )