"""Explainability and error-analysis utilities for the final merge model."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_random_forest_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return Random Forest mean-decrease-in-impurity importance."""

    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            "The final model does not provide feature_importances_."
        )

    importances = np.asarray(
        model.feature_importances_,
        dtype=float,
    )

    if len(importances) != len(feature_names):
        raise ValueError(
            "Model importance count does not match the feature list."
        )

    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "impurity_importance": importances,
        }
    )

    total_importance = float(
        importance_table[
            "impurity_importance"
        ].sum()
    )

    if total_importance > 0:
        importance_table[
            "importance_percentage"
        ] = (
            importance_table[
                "impurity_importance"
            ]
            / total_importance
            * 100
        )
    else:
        importance_table[
            "importance_percentage"
        ] = 0.0

    importance_table = importance_table.sort_values(
        [
            "impurity_importance",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    importance_table.insert(
        0,
        "importance_rank",
        range(
            1,
            len(importance_table) + 1,
        ),
    )

    return importance_table


def calculate_test_permutation_importance(
    model: Any,
    features: pd.DataFrame,
    targets: pd.Series,
    feature_names: list[str],
    repeats: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate post-hoc test permutation importance.

    This is for final interpretation only and must not be used to
    modify, tune or reselect the final model.
    """

    missing_features = sorted(
        set(feature_names)
        - set(features.columns)
    )

    if missing_features:
        raise ValueError(
            "Test data is missing features: "
            f"{missing_features}"
        )

    selected_features = features[
        feature_names
    ]

    result = permutation_importance(
        estimator=model,
        X=selected_features,
        y=targets,
        scoring="roc_auc",
        n_repeats=repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "permutation_importance_mean": (
                result.importances_mean
            ),
            "permutation_importance_std": (
                result.importances_std
            ),
        }
    )

    importance_table[
        "positive_importance"
    ] = (
        importance_table[
            "permutation_importance_mean"
        ]
        > 0
    )

    importance_table[
        "importance_stability_ratio"
    ] = np.where(
        importance_table[
            "permutation_importance_std"
        ]
        > 0,
        importance_table[
            "permutation_importance_mean"
        ]
        / importance_table[
            "permutation_importance_std"
        ],
        np.nan,
    )

    importance_table = importance_table.sort_values(
        [
            "permutation_importance_mean",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    importance_table.insert(
        0,
        "importance_rank",
        range(
            1,
            len(importance_table) + 1,
        ),
    )

    return importance_table


def combine_importance_tables(
    impurity_importance: pd.DataFrame,
    permutation_importance_table: pd.DataFrame,
) -> pd.DataFrame:
    """Combine model-specific and model-agnostic importance results."""

    impurity_columns = [
        "feature",
        "importance_rank",
        "impurity_importance",
        "importance_percentage",
    ]

    permutation_columns = [
        "feature",
        "importance_rank",
        "permutation_importance_mean",
        "permutation_importance_std",
        "positive_importance",
        "importance_stability_ratio",
    ]

    combined = impurity_importance[
        impurity_columns
    ].rename(
        columns={
            "importance_rank": (
                "impurity_importance_rank"
            )
        }
    ).merge(
        permutation_importance_table[
            permutation_columns
        ].rename(
            columns={
                "importance_rank": (
                    "permutation_importance_rank"
                )
            }
        ),
        on="feature",
        how="outer",
        validate="one_to_one",
    )

    combined[
        "average_rank"
    ] = combined[
        [
            "impurity_importance_rank",
            "permutation_importance_rank",
        ]
    ].mean(axis=1)

    combined = combined.sort_values(
        [
            "average_rank",
            "permutation_importance_mean",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    combined.insert(
        0,
        "combined_rank",
        range(
            1,
            len(combined) + 1,
        ),
    )

    return combined


def build_probability_calibration_table(
    predictions: pd.DataFrame,
    bin_count: int = 10,
) -> pd.DataFrame:
    """Compare predicted merge probabilities with observed outcomes."""

    required_columns = {
        "actual_target",
        "merge_probability",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Predictions are missing columns: "
            f"{sorted(missing_columns)}"
        )

    calibration_data = predictions[
        [
            "actual_target",
            "merge_probability",
        ]
    ].copy()

    calibration_data[
        "probability_bin"
    ] = pd.cut(
        calibration_data[
            "merge_probability"
        ],
        bins=np.linspace(
            0,
            1,
            bin_count + 1,
        ),
        include_lowest=True,
        right=True,
    )

    calibration_table = (
        calibration_data.groupby(
            "probability_bin",
            observed=False,
        )
        .agg(
            prediction_count=(
                "actual_target",
                "size",
            ),
            average_predicted_probability=(
                "merge_probability",
                "mean",
            ),
            observed_merge_rate=(
                "actual_target",
                "mean",
            ),
        )
        .reset_index()
    )

    calibration_table[
        "calibration_gap"
    ] = (
        calibration_table[
            "average_predicted_probability"
        ]
        - calibration_table[
            "observed_merge_rate"
        ]
    )

    calibration_table[
        "absolute_calibration_gap"
    ] = calibration_table[
        "calibration_gap"
    ].abs()

    calibration_table[
        "probability_bin"
    ] = calibration_table[
        "probability_bin"
    ].astype(str)

    return calibration_table


def build_confidence_band_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize model accuracy across confidence bands."""

    required_columns = {
        "actual_target",
        "predicted_target",
        "prediction_correct",
        "prediction_confidence",
        "merge_probability",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Predictions are missing columns: "
            f"{sorted(missing_columns)}"
        )

    confidence_data = predictions.copy()

    confidence_data[
        "confidence_band"
    ] = pd.cut(
        confidence_data[
            "prediction_confidence"
        ],
        bins=[
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ],
        labels=[
            "50–60%",
            "60–70%",
            "70–80%",
            "80–90%",
            "90–100%",
        ],
        include_lowest=True,
        right=True,
    )

    summary = (
        confidence_data.groupby(
            "confidence_band",
            observed=False,
        )
        .agg(
            prediction_count=(
                "actual_target",
                "size",
            ),
            correct_predictions=(
                "prediction_correct",
                "sum",
            ),
            average_confidence=(
                "prediction_confidence",
                "mean",
            ),
            average_merge_probability=(
                "merge_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    summary[
        "accuracy"
    ] = np.where(
        summary[
            "prediction_count"
        ]
        > 0,
        summary[
            "correct_predictions"
        ]
        / summary[
            "prediction_count"
        ],
        np.nan,
    )

    summary[
        "confidence_band"
    ] = summary[
        "confidence_band"
    ].astype(str)

    return summary


def classify_prediction_error(
    actual_target: int,
    predicted_target: int,
) -> str:
    """Return the classification outcome type."""

    if actual_target == 1 and predicted_target == 1:
        return "true_positive"

    if actual_target == 0 and predicted_target == 0:
        return "true_negative"

    if actual_target == 0 and predicted_target == 1:
        return "false_positive"

    return "false_negative"


def build_error_analysis_table(
    predictions: pd.DataFrame,
    test_features: pd.DataFrame,
    identifiers: pd.Series,
) -> pd.DataFrame:
    """Create an enriched row-level error-analysis table."""

    if len(predictions) != len(test_features):
        raise ValueError(
            "Predictions and test features are misaligned."
        )

    if len(predictions) != len(identifiers):
        raise ValueError(
            "Predictions and identifiers are misaligned."
        )

    prediction_data = predictions.reset_index(
        drop=True
    ).copy()

    feature_data = test_features.reset_index(
        drop=True
    ).copy()

    identifier_data = identifiers.reset_index(
        drop=True
    )

    if "pr_number" not in prediction_data.columns:
        prediction_data.insert(
            0,
            "pr_number",
            identifier_data,
        )

    prediction_data[
        "prediction_outcome"
    ] = [
        classify_prediction_error(
            int(actual),
            int(predicted),
        )
        for actual, predicted in zip(
            prediction_data[
                "actual_target"
            ],
            prediction_data[
                "predicted_target"
            ],
            strict=True,
        )
    ]

    prediction_data[
        "probability_distance_from_threshold"
    ] = (
        prediction_data[
            "merge_probability"
        ]
        - prediction_data[
            "selected_threshold"
        ]
    ).abs()

    enriched = pd.concat(
        [
            prediction_data,
            feature_data,
        ],
        axis=1,
    )

    enriched[
        "requires_error_review"
    ] = ~enriched[
        "prediction_correct"
    ]

    return enriched.sort_values(
        [
            "requires_error_review",
            "prediction_confidence",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


def build_error_type_summary(
    error_analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize true and false prediction types."""

    required_columns = {
        "prediction_outcome",
        "prediction_confidence",
        "merge_probability",
    }

    missing_columns = (
        required_columns
        - set(error_analysis.columns)
    )

    if missing_columns:
        raise ValueError(
            "Error analysis is missing columns: "
            f"{sorted(missing_columns)}"
        )

    summary = (
        error_analysis.groupby(
            "prediction_outcome",
            observed=False,
        )
        .agg(
            row_count=(
                "prediction_outcome",
                "size",
            ),
            average_confidence=(
                "prediction_confidence",
                "mean",
            ),
            minimum_confidence=(
                "prediction_confidence",
                "min",
            ),
            maximum_confidence=(
                "prediction_confidence",
                "max",
            ),
            average_merge_probability=(
                "merge_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary.sort_values(
        "row_count",
        ascending=False,
    ).reset_index(drop=True)


def build_error_feature_profile(
    error_analysis: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """Compare feature averages for correct and incorrect predictions."""

    missing_features = sorted(
        set(selected_features)
        - set(error_analysis.columns)
    )

    if missing_features:
        raise ValueError(
            "Error-analysis data is missing features: "
            f"{missing_features}"
        )

    records = []

    correct_mask = error_analysis[
        "prediction_correct"
    ].astype(bool)

    error_mask = ~correct_mask

    for feature in selected_features:
        numeric_feature = pd.to_numeric(
            error_analysis[
                feature
            ],
            errors="coerce",
        )

        correct_values = numeric_feature[
            correct_mask
        ]

        error_values = numeric_feature[
            error_mask
        ]

        correct_mean = float(
            correct_values.mean()
        )

        error_mean = float(
            error_values.mean()
        )

        records.append(
            {
                "feature": feature,
                "correct_prediction_mean": (
                    correct_mean
                ),
                "incorrect_prediction_mean": (
                    error_mean
                ),
                "incorrect_minus_correct": (
                    error_mean
                    - correct_mean
                ),
                "absolute_mean_difference": abs(
                    error_mean
                    - correct_mean
                ),
            }
        )

    return pd.DataFrame(
        records
    ).sort_values(
        [
            "absolute_mean_difference",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)


def calculate_explainability_summary(
    predictions: pd.DataFrame,
    combined_importance: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate final model interpretation summary metrics."""

    actual = predictions[
        "actual_target"
    ].astype(int)

    predicted = predictions[
        "predicted_target"
    ].astype(int)

    probabilities = predictions[
        "merge_probability"
    ].astype(float)

    top_features = (
        combined_importance.head(10)[
            "feature"
        ].tolist()
    )

    high_confidence_errors = int(
        (
            (~predictions[
                "prediction_correct"
            ])
            & (
                predictions[
                    "prediction_confidence"
                ]
                >= 0.75
            )
        ).sum()
    )

    return {
        "row_count": len(
            predictions
        ),
        "accuracy": float(
            accuracy_score(
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
        "roc_auc": float(
            roc_auc_score(
                actual,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                actual,
                probabilities,
            )
        ),
        "incorrect_prediction_count": int(
            (
                actual
                != predicted
            ).sum()
        ),
        "high_confidence_error_count": (
            high_confidence_errors
        ),
        "top_combined_features": (
            top_features
        ),
    }


def validate_explainability_outputs(
    combined_importance: pd.DataFrame,
    calibration_table: pd.DataFrame,
    confidence_table: pd.DataFrame,
    error_analysis: pd.DataFrame,
    expected_feature_count: int,
    expected_row_count: int,
) -> dict[str, Any]:
    """Validate all Stage 5G analytical outputs."""

    feature_count_valid = (
        len(combined_importance)
        == expected_feature_count
    )

    row_count_valid = (
        len(error_analysis)
        == expected_row_count
    )

    importance_values_valid = bool(
        combined_importance[
            "impurity_importance"
        ].between(
            0,
            1,
            inclusive="both",
        ).all()
    )

    calibration_valid = {
        "prediction_count_total": int(
            calibration_table[
                "prediction_count"
            ].sum()
        ),
        "matches_expected_rows": (
            int(
                calibration_table[
                    "prediction_count"
                ].sum()
            )
            == expected_row_count
        ),
    }

    confidence_valid = {
        "prediction_count_total": int(
            confidence_table[
                "prediction_count"
            ].sum()
        ),
        "matches_expected_rows": (
            int(
                confidence_table[
                    "prediction_count"
                ].sum()
            )
            == expected_row_count
        ),
    }

    validation_passed = (
        feature_count_valid
        and row_count_valid
        and importance_values_valid
        and calibration_valid[
            "matches_expected_rows"
        ]
        and confidence_valid[
            "matches_expected_rows"
        ]
    )

    return {
        "expected_feature_count": (
            expected_feature_count
        ),
        "actual_feature_count": len(
            combined_importance
        ),
        "feature_count_valid": (
            feature_count_valid
        ),
        "expected_row_count": (
            expected_row_count
        ),
        "actual_row_count": len(
            error_analysis
        ),
        "row_count_valid": (
            row_count_valid
        ),
        "importance_values_valid": (
            importance_values_valid
        ),
        "calibration_validation": (
            calibration_valid
        ),
        "confidence_validation": (
            confidence_valid
        ),
        "validation_passed": (
            validation_passed
        ),
    }