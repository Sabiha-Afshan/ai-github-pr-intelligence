"""Explainability and error analysis for the final merge-delay model."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def calculate_logistic_coefficient_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return standardized Logistic Regression coefficient importance."""

    if not hasattr(model, "coef_"):
        raise TypeError(
            "The final merge-delay model does not provide coefficients."
        )

    coefficients = np.asarray(
        model.coef_,
        dtype=float,
    )

    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        raise ValueError(
            "Expected a binary Logistic Regression coefficient matrix."
        )

    coefficient_values = coefficients[0]

    if len(coefficient_values) != len(feature_names):
        raise ValueError(
            "Coefficient count does not match the feature count."
        )

    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficient_values,
        }
    )

    coefficient_table[
        "absolute_coefficient"
    ] = coefficient_table[
        "coefficient"
    ].abs()

    coefficient_table[
        "odds_ratio"
    ] = np.exp(
        coefficient_table[
            "coefficient"
        ]
    )

    coefficient_table[
        "effect_direction"
    ] = np.select(
        [
            coefficient_table[
                "coefficient"
            ]
            > 0,
            coefficient_table[
                "coefficient"
            ]
            < 0,
        ],
        [
            "increases_delay_risk",
            "reduces_delay_risk",
        ],
        default="neutral",
    )

    coefficient_table[
        "interpretation"
    ] = coefficient_table.apply(
        lambda row: (
            "A one-standard-deviation increase in this feature "
            f"multiplies the estimated delay odds by "
            f"{row['odds_ratio']:.3f}."
        ),
        axis=1,
    )

    coefficient_table = coefficient_table.sort_values(
        [
            "absolute_coefficient",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    coefficient_table.insert(
        0,
        "coefficient_rank",
        range(
            1,
            len(coefficient_table) + 1,
        ),
    )

    return coefficient_table


def calculate_test_permutation_importance(
    model: Any,
    features: pd.DataFrame,
    targets: pd.Series,
    feature_names: list[str],
    repeats: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate final test permutation importance.

    This is post-hoc interpretation only. It must not be used
    to modify the locked model, features or threshold.
    """

    missing_features = sorted(
        set(feature_names)
        - set(features.columns)
    )

    if missing_features:
        raise ValueError(
            "Test features are missing columns: "
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
        "permutation_rank",
        range(
            1,
            len(importance_table) + 1,
        ),
    )

    return importance_table


def combine_explainability_tables(
    coefficient_table: pd.DataFrame,
    permutation_table: pd.DataFrame,
) -> pd.DataFrame:
    """Combine coefficient and permutation evidence."""

    coefficient_columns = [
        "feature",
        "coefficient_rank",
        "coefficient",
        "absolute_coefficient",
        "odds_ratio",
        "effect_direction",
        "interpretation",
    ]

    permutation_columns = [
        "feature",
        "permutation_rank",
        "permutation_importance_mean",
        "permutation_importance_std",
        "positive_importance",
        "importance_stability_ratio",
    ]

    combined = coefficient_table[
        coefficient_columns
    ].merge(
        permutation_table[
            permutation_columns
        ],
        on="feature",
        how="outer",
        validate="one_to_one",
    )

    combined[
        "average_rank"
    ] = combined[
        [
            "coefficient_rank",
            "permutation_rank",
        ]
    ].mean(axis=1)

    combined = combined.sort_values(
        [
            "average_rank",
            "absolute_coefficient",
            "permutation_importance_mean",
        ],
        ascending=[
            True,
            False,
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


def calculate_local_logit_contributions(
    model: Any,
    features: pd.DataFrame,
    identifiers: pd.Series,
    actual_targets: pd.Series,
    predicted_targets: pd.Series,
    probabilities: pd.Series,
    top_feature_count: int = 10,
) -> pd.DataFrame:
    """Calculate row-level Logistic Regression contributions."""

    coefficients = np.asarray(
        model.coef_[0],
        dtype=float,
    )

    intercept = float(
        model.intercept_[0]
    )

    feature_names = list(
        features.columns
    )

    if len(coefficients) != len(feature_names):
        raise ValueError(
            "Model coefficients do not match the feature schema."
        )

    contribution_matrix = (
        features.to_numpy(
            dtype=float
        )
        * coefficients
    )

    records = []

    for row_position in range(
        len(features)
    ):
        row_contributions = (
            contribution_matrix[
                row_position
            ]
        )

        top_positions = np.argsort(
            np.abs(
                row_contributions
            )
        )[::-1][
            :top_feature_count
        ]

        for local_rank, feature_position in enumerate(
            top_positions,
            start=1,
        ):
            contribution = float(
                row_contributions[
                    feature_position
                ]
            )

            records.append(
                {
                    "pr_number": (
                        identifiers.iloc[
                            row_position
                        ]
                    ),
                    "actual_target": int(
                        actual_targets.iloc[
                            row_position
                        ]
                    ),
                    "predicted_target": int(
                        predicted_targets.iloc[
                            row_position
                        ]
                    ),
                    "delay_probability": float(
                        probabilities.iloc[
                            row_position
                        ]
                    ),
                    "model_intercept": (
                        intercept
                    ),
                    "local_rank": (
                        local_rank
                    ),
                    "feature": (
                        feature_names[
                            feature_position
                        ]
                    ),
                    "standardized_feature_value": float(
                        features.iloc[
                            row_position,
                            feature_position,
                        ]
                    ),
                    "coefficient": float(
                        coefficients[
                            feature_position
                        ]
                    ),
                    "logit_contribution": (
                        contribution
                    ),
                    "contribution_direction": (
                        "toward_delayed"
                        if contribution > 0
                        else (
                            "toward_not_delayed"
                            if contribution < 0
                            else "neutral"
                        )
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def build_error_analysis(
    predictions: pd.DataFrame,
    local_contributions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create row-level and feature-level error analysis."""

    required_prediction_columns = {
        "pr_number",
        "actual_target",
        "predicted_target",
        "delay_probability",
        "prediction_confidence",
        "prediction_correct",
        "prediction_outcome",
    }

    missing_columns = sorted(
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Predictions are missing columns: "
            f"{missing_columns}"
        )

    errors = predictions.loc[
        ~predictions[
            "prediction_correct"
        ].astype(bool)
    ].copy()

    errors[
        "high_confidence_error"
    ] = (
        errors[
            "prediction_confidence"
        ]
        >= 0.75
    )

    error_contributions = (
        local_contributions.merge(
            errors[
                [
                    "pr_number",
                    "prediction_outcome",
                    "prediction_confidence",
                    "high_confidence_error",
                ]
            ],
            on="pr_number",
            how="inner",
            validate="many_to_one",
        )
    )

    feature_summary = (
        error_contributions.groupby(
            [
                "prediction_outcome",
                "feature",
                "contribution_direction",
            ],
            observed=False,
        )
        .agg(
            occurrence_count=(
                "feature",
                "size",
            ),
            average_absolute_contribution=(
                "logit_contribution",
                lambda values: float(
                    np.abs(values).mean()
                ),
            ),
            average_logit_contribution=(
                "logit_contribution",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "prediction_outcome",
                "occurrence_count",
                "average_absolute_contribution",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    errors = errors.sort_values(
        [
            "high_confidence_error",
            "prediction_confidence",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    return (
        errors,
        feature_summary,
    )


def build_probability_band_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize prediction performance by probability band."""

    working = predictions.copy()

    working[
        "probability_band"
    ] = pd.cut(
        working[
            "delay_probability"
        ],
        bins=[
            0.0,
            0.25,
            0.50,
            0.75,
            1.00,
        ],
        labels=[
            "0–25%",
            "25–50%",
            "50–75%",
            "75–100%",
        ],
        include_lowest=True,
        right=True,
    )

    summary = (
        working.groupby(
            "probability_band",
            observed=False,
        )
        .agg(
            prediction_count=(
                "actual_target",
                "size",
            ),
            observed_delay_rate=(
                "actual_target",
                "mean",
            ),
            average_predicted_probability=(
                "delay_probability",
                "mean",
            ),
            correct_predictions=(
                "prediction_correct",
                "sum",
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
        "calibration_gap"
    ] = (
        summary[
            "average_predicted_probability"
        ]
        - summary[
            "observed_delay_rate"
        ]
    )

    summary[
        "probability_band"
    ] = summary[
        "probability_band"
    ].astype(str)

    return summary


def validate_explainability_outputs(
    coefficient_table: pd.DataFrame,
    permutation_table: pd.DataFrame,
    combined_table: pd.DataFrame,
    local_contributions: pd.DataFrame,
    errors: pd.DataFrame,
    probability_bands: pd.DataFrame,
    expected_feature_count: int,
    expected_test_rows: int,
    top_feature_count: int,
) -> dict[str, Any]:
    """Validate all Stage 6G outputs."""

    coefficient_count_valid = bool(
        len(coefficient_table)
        == expected_feature_count
    )

    permutation_count_valid = bool(
        len(permutation_table)
        == expected_feature_count
    )

    combined_count_valid = bool(
        len(combined_table)
        == expected_feature_count
    )

    expected_local_rows = (
        expected_test_rows
        * top_feature_count
    )

    local_count_valid = bool(
        len(local_contributions)
        == expected_local_rows
    )

    contribution_values_valid = bool(
        np.isfinite(
            local_contributions[
                "logit_contribution"
            ].to_numpy(
                dtype=float
            )
        ).all()
    )

    probability_band_total_valid = bool(
        int(
            probability_bands[
                "prediction_count"
            ].sum()
        )
        == expected_test_rows
    )

    errors_valid = bool(
        len(errors)
        <= expected_test_rows
    )

    validation_passed = bool(
        coefficient_count_valid
        and permutation_count_valid
        and combined_count_valid
        and local_count_valid
        and contribution_values_valid
        and probability_band_total_valid
        and errors_valid
    )

    return {
        "expected_feature_count": (
            expected_feature_count
        ),
        "coefficient_feature_count": len(
            coefficient_table
        ),
        "permutation_feature_count": len(
            permutation_table
        ),
        "combined_feature_count": len(
            combined_table
        ),
        "coefficient_count_valid": (
            coefficient_count_valid
        ),
        "permutation_count_valid": (
            permutation_count_valid
        ),
        "combined_count_valid": (
            combined_count_valid
        ),
        "expected_local_contribution_rows": (
            expected_local_rows
        ),
        "actual_local_contribution_rows": len(
            local_contributions
        ),
        "local_count_valid": (
            local_count_valid
        ),
        "contribution_values_valid": (
            contribution_values_valid
        ),
        "error_count": len(
            errors
        ),
        "errors_valid": (
            errors_valid
        ),
        "probability_band_total_valid": (
            probability_band_total_valid
        ),
        "validation_passed": (
            validation_passed
        ),
    }