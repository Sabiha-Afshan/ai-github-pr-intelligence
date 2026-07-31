"""Final held-out evaluation utilities for Model 2."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.models.merge_delay_evaluation import (
    build_prediction_table,
    calculate_binary_metrics,
)

PR_IDENTIFIER_COLUMN = "pr_number"
SPLIT_COLUMN = "split"
TARGET_COLUMN = "merge_delay_target"


def validate_locked_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Validate the locked Model 2 configuration."""

    required_fields = {
        "model_name",
        "model_family",
        "model_path",
        "threshold",
        "target",
        "feature_count",
        "features",
        "parameters",
        "validation_metrics",
        "test_used_for_selection",
    }

    missing_fields = sorted(required_fields - set(configuration))

    threshold = configuration.get("threshold")

    features = configuration.get(
        "features",
        [],
    )

    feature_count = configuration.get("feature_count")

    test_used_for_selection = configuration.get("test_used_for_selection")

    threshold_valid = bool(
        isinstance(
            threshold,
            int | float,
        )
        and 0 <= float(threshold) <= 1
    )

    features_valid = bool(
        isinstance(features, list)
        and len(features) > 0
        and len(features) == feature_count
        and len(features) == len(set(features))
    )

    validation_passed = bool(
        not missing_fields
        and threshold_valid
        and features_valid
        and test_used_for_selection is False
    )

    return {
        "missing_fields": (missing_fields),
        "threshold": threshold,
        "threshold_valid": (threshold_valid),
        "feature_count": (feature_count),
        "actual_feature_count": len(features),
        "features_valid": (features_valid),
        "test_used_for_selection": (test_used_for_selection),
        "validation_passed": (validation_passed),
    }


def load_final_test_split(
    dataframe: pd.DataFrame,
    feature_names: list[str],
) -> tuple[
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """Load and validate the final chronological test split."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        *feature_names,
    }

    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        raise ValueError(f"Test dataset is missing columns: {missing_columns}")

    expected_columns = {
        PR_IDENTIFIER_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        *feature_names,
    }

    unexpected_columns = sorted(set(dataframe.columns) - expected_columns)

    if unexpected_columns:
        raise ValueError(
            f"Test dataset contains unexpected columns: {unexpected_columns}"
        )

    split_values = (
        dataframe[SPLIT_COLUMN].astype(str).str.strip().str.lower().unique().tolist()
    )

    if split_values != ["test"]:
        raise ValueError(
            f"Final dataset must contain only test rows. Found: {split_values}"
        )

    identifiers = dataframe[PR_IDENTIFIER_COLUMN].reset_index(drop=True)

    targets = (
        pd.to_numeric(
            dataframe[TARGET_COLUMN],
            errors="raise",
        )
        .astype(int)
        .reset_index(drop=True)
    )

    target_values = set(targets.unique())

    if target_values != {0, 1}:
        raise ValueError("The final test split must contain both classes.")

    features = (
        dataframe[feature_names]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .astype(float)
        .reset_index(drop=True)
    )

    feature_values = features.to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError("Test features contain missing or infinite values.")

    if len(dataframe) != 38:
        raise ValueError(f"Expected 38 final test rows, but found {len(dataframe)}.")

    return (
        identifiers,
        targets,
        features,
    )


def evaluate_locked_model(
    model: Any,
    identifiers: pd.Series,
    targets: pd.Series,
    features: pd.DataFrame,
    threshold: float,
    model_name: str,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    """Evaluate the locked model without modifying it."""

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError("Locked model does not support predict_proba.")

    model_feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if model_feature_names is not None:
        if list(model_feature_names) != list(features.columns):
            raise ValueError(
                "Locked model feature order does not match the final test dataset."
            )

    probabilities = model.predict_proba(features)[:, 1]

    metrics = calculate_binary_metrics(
        actual=targets,
        probabilities=probabilities,
        threshold=threshold,
    )

    metrics["model_name"] = model_name

    predictions = build_prediction_table(
        identifiers=identifiers,
        actual=targets,
        probabilities=probabilities,
        threshold=threshold,
        model_name=model_name,
        split_name="test",
    )

    return (
        metrics,
        predictions,
    )


def build_classification_report_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final classification report."""

    report = classification_report(
        predictions["actual_target"].astype(int),
        predictions["predicted_target"].astype(int),
        labels=[0, 1],
        target_names=[
            "not_delayed",
            "delayed",
        ],
        output_dict=True,
        zero_division=0,
    )

    records = []

    for label, values in report.items():
        if isinstance(
            values,
            dict,
        ):
            records.append(
                {
                    "class": label,
                    "precision": (values.get("precision")),
                    "recall": (values.get("recall")),
                    "f1_score": (values.get("f1-score")),
                    "support": (values.get("support")),
                }
            )
        else:
            records.append(
                {
                    "class": label,
                    "precision": (values),
                    "recall": np.nan,
                    "f1_score": np.nan,
                    "support": len(predictions),
                }
            )

    return pd.DataFrame(records)


def build_error_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return false-positive and false-negative test records."""

    required_columns = {
        "prediction_correct",
        "prediction_confidence",
        "prediction_outcome",
    }

    missing_columns = sorted(required_columns - set(predictions.columns))

    if missing_columns:
        raise ValueError(f"Prediction table is missing columns: {missing_columns}")

    error_table = predictions.loc[
        ~predictions["prediction_correct"].astype(bool)
    ].copy()

    error_table["high_confidence_error"] = error_table["prediction_confidence"] >= 0.75

    return error_table.sort_values(
        [
            "high_confidence_error",
            "prediction_confidence",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


def build_validation_test_comparison(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Compare locked validation results with final test results."""

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
        validation_value = float(validation_metrics[metric_name])

        test_value = float(test_metrics[metric_name])

        records.append(
            {
                "metric": (metric_name),
                "validation_value": (validation_value),
                "test_value": (test_value),
                "test_minus_validation": (test_value - validation_value),
                "absolute_difference": abs(test_value - validation_value),
            }
        )

    return pd.DataFrame(records)


def validate_final_outputs(
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    classification_table: pd.DataFrame,
    error_table: pd.DataFrame,
    expected_threshold: float,
) -> dict[str, Any]:
    """Validate the final Model 2 evaluation outputs."""

    required_metric_names = {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier_score",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }

    missing_metrics = sorted(required_metric_names - set(metrics))

    prediction_count_valid = bool(len(predictions) == 38)

    threshold_values = predictions["selected_threshold"].astype(float).unique().tolist()

    threshold_unchanged = bool(
        len(threshold_values) == 1
        and bool(
            np.isclose(
                threshold_values[0],
                expected_threshold,
            )
        )
    )

    confusion_total = sum(
        int(metrics[column])
        for column in (
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        )
    )

    confusion_total_valid = bool(confusion_total == 38)

    prediction_correct_count = int(predictions["prediction_correct"].astype(bool).sum())

    error_count = int(len(error_table))

    prediction_partition_valid = bool(prediction_correct_count + error_count == 38)

    probabilities_valid = bool(
        predictions["delay_probability"]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
    )

    classification_report_valid = bool(not classification_table.empty)

    validation_passed = bool(
        not missing_metrics
        and prediction_count_valid
        and threshold_unchanged
        and confusion_total_valid
        and prediction_partition_valid
        and probabilities_valid
        and classification_report_valid
    )

    return {
        "missing_metrics": (missing_metrics),
        "prediction_count": int(len(predictions)),
        "expected_prediction_count": 38,
        "prediction_count_valid": (prediction_count_valid),
        "threshold_values": [float(value) for value in threshold_values],
        "expected_threshold": float(expected_threshold),
        "threshold_unchanged": (threshold_unchanged),
        "confusion_total": int(confusion_total),
        "confusion_total_valid": (confusion_total_valid),
        "correct_prediction_count": (prediction_correct_count),
        "error_count": (error_count),
        "prediction_partition_valid": (prediction_partition_valid),
        "probabilities_valid": (probabilities_valid),
        "classification_report_valid": (classification_report_valid),
        "validation_passed": (validation_passed),
    }
