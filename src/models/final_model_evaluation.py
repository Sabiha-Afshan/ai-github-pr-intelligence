"""Final held-out test evaluation utilities."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.models.merge_outcome_evaluation import (
    calculate_metrics,
    create_prediction_table,
    get_positive_probability,
)


def evaluate_locked_model(
    model_name: str,
    model: Any,
    features: pd.DataFrame,
    targets: pd.Series,
    identifiers: pd.Series,
    selected_features: list[str],
    selected_threshold: float,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate a locked model once on held-out test data."""

    if not selected_features:
        raise ValueError(
            "The selected feature list is empty."
        )

    missing_features = sorted(
        set(selected_features)
        - set(features.columns)
    )

    if missing_features:
        raise ValueError(
            "Test data is missing selected features: "
            f"{missing_features}"
        )

    if len(features) != len(targets):
        raise ValueError(
            "Test features and targets are misaligned."
        )

    if len(features) != len(identifiers):
        raise ValueError(
            "Test features and identifiers are misaligned."
        )

    model_features = features[
        selected_features
    ].copy()

    probabilities = get_positive_probability(
        model,
        model_features,
    )

    metrics = calculate_metrics(
        targets=targets,
        probabilities=probabilities,
        threshold=selected_threshold,
    )

    metrics.update(
        {
            "model_name": model_name,
            "evaluation_split": "test",
            "evaluation_rows": len(
                targets
            ),
            "feature_count": len(
                selected_features
            ),
            "threshold_source": (
                "Stage 5E validation selection"
            ),
        }
    )

    predictions = create_prediction_table(
        identifiers=identifiers,
        targets=targets,
        probabilities=probabilities,
        threshold=selected_threshold,
        model_name=model_name,
    )

    predicted_labels = (
        probabilities
        >= selected_threshold
    ).astype(int)

    report_dictionary = classification_report(
        y_true=targets,
        y_pred=predicted_labels,
        labels=[0, 1],
        target_names=[
            "Closed without merge",
            "Merged",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_records = []

    for label, values in (
        report_dictionary.items()
    ):
        if isinstance(
            values,
            dict,
        ):
            report_records.append(
                {
                    "class_or_average": label,
                    "precision": values.get(
                        "precision"
                    ),
                    "recall": values.get(
                        "recall"
                    ),
                    "f1_score": values.get(
                        "f1-score"
                    ),
                    "support": values.get(
                        "support"
                    ),
                }
            )
        else:
            report_records.append(
                {
                    "class_or_average": label,
                    "precision": None,
                    "recall": None,
                    "f1_score": values,
                    "support": len(
                        targets
                    ),
                }
            )

    classification_table = pd.DataFrame(
        report_records
    )

    return (
        metrics,
        predictions,
        classification_table,
    )


def compare_validation_and_test(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Compare locked validation and test performance."""

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier_score",
    ]

    records = []

    for metric_name in metric_names:
        validation_value = float(
            validation_metrics[
                metric_name
            ]
        )

        test_value = float(
            test_metrics[
                metric_name
            ]
        )

        records.append(
            {
                "metric": metric_name,
                "validation_value": (
                    validation_value
                ),
                "test_value": test_value,
                "test_minus_validation": (
                    test_value
                    - validation_value
                ),
                "absolute_difference": abs(
                    test_value
                    - validation_value
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def identify_high_confidence_errors(
    predictions: pd.DataFrame,
    minimum_confidence: float = 0.75,
) -> pd.DataFrame:
    """Identify incorrect high-confidence predictions."""

    required_columns = {
        "prediction_correct",
        "prediction_confidence",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        predictions.loc[
            (~predictions[
                "prediction_correct"
            ])
            & (
                predictions[
                    "prediction_confidence"
                ]
                >= minimum_confidence
            )
        ]
        .sort_values(
            "prediction_confidence",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def validate_final_test_results(
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    expected_rows: int,
) -> dict[str, Any]:
    """Validate final test-evaluation outputs."""

    probability_range_valid = bool(
        predictions[
            "merge_probability"
        ].between(
            0,
            1,
            inclusive="both",
        ).all()
    )

    metric_range_valid = all(
        0
        <= float(
            metrics[
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

    confusion_total = sum(
        int(
            metrics[
                metric_name
            ]
        )
        for metric_name in (
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        )
    )

    validation_passed = (
        len(predictions)
        == expected_rows
        and probability_range_valid
        and metric_range_valid
        and confusion_total
        == expected_rows
    )

    return {
        "expected_test_rows": expected_rows,
        "actual_test_rows": len(
            predictions
        ),
        "probability_range_valid": (
            probability_range_valid
        ),
        "metric_range_valid": (
            metric_range_valid
        ),
        "confusion_matrix_total": (
            confusion_total
        ),
        "validation_passed": (
            validation_passed
        ),
    }