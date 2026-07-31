"""Utilities for building the unified PR intelligence dataset."""

from typing import Any

import numpy as np
import pandas as pd

PR_IDENTIFIER_COLUMN = "pr_number"


def resolve_configuration_threshold(
    configuration: dict[str, Any],
) -> float:
    """Resolve a locked model threshold from supported configuration keys."""

    threshold_keys = [
        "threshold",
        "selected_threshold",
        "locked_threshold",
    ]

    threshold = None

    for key in threshold_keys:
        if key in configuration:
            threshold = configuration[key]
            break

    if threshold is None:
        raise KeyError("Locked model configuration does not contain a threshold.")

    threshold_value = float(threshold)

    if not 0 <= threshold_value <= 1:
        raise ValueError("Locked model threshold must be between zero and one.")

    return threshold_value


def resolve_configuration_features(
    configuration: dict[str, Any],
) -> list[str]:
    """Resolve and validate locked model feature names."""

    feature_keys = [
        "features",
        "feature_names",
        "approved_features",
    ]

    feature_names = None

    for key in feature_keys:
        if key in configuration:
            feature_names = configuration[key]
            break

    if not isinstance(
        feature_names,
        list,
    ):
        raise ValueError("Locked model configuration does not contain a feature list.")

    normalized_features = [str(feature) for feature in feature_names]

    if not normalized_features:
        raise ValueError("Locked model configuration contains no features.")

    if len(normalized_features) != len(set(normalized_features)):
        raise ValueError("Locked model feature names contain duplicates.")

    configured_feature_count = configuration.get("feature_count")

    if configured_feature_count is not None and int(configured_feature_count) != len(
        normalized_features
    ):
        raise ValueError("Configured feature count does not match the feature list.")

    return normalized_features


def resolve_preprocessor_input_features(
    preprocessor: Any,
    fallback_features: list[str],
) -> list[str]:
    """Resolve the raw feature order expected by a fitted preprocessor."""

    preprocessor_features = getattr(
        preprocessor,
        "feature_names_in_",
        None,
    )

    if preprocessor_features is None:
        return list(fallback_features)

    resolved_features = [str(feature) for feature in preprocessor_features]

    if not resolved_features:
        raise ValueError("The fitted preprocessor contains no input feature names.")

    return resolved_features


def resolve_model_feature_order(
    model: Any,
    preprocessor: Any,
    configured_features: list[str],
) -> list[str]:
    """Resolve the transformed feature order expected by the model."""

    model_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if model_features is not None:
        return [str(feature) for feature in model_features]

    if hasattr(
        preprocessor,
        "get_feature_names_out",
    ):
        return [str(feature) for feature in preprocessor.get_feature_names_out()]

    return list(configured_features)


def validate_scoring_source(
    dataframe: pd.DataFrame,
    required_features: list[str],
) -> dict[str, Any]:
    """Validate one model-scoring source dataset."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        *required_features,
    }

    missing_columns = sorted(required_columns - set(dataframe.columns))

    duplicate_pr_count = (
        int(dataframe.duplicated(subset=[PR_IDENTIFIER_COLUMN]).sum())
        if PR_IDENTIFIER_COLUMN in dataframe.columns
        else None
    )

    infinite_value_count = 0
    non_numeric_features = []

    for feature in required_features:
        if feature not in dataframe.columns:
            continue

        numeric_values = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        )

        source_non_missing_count = int(dataframe[feature].notna().sum())

        numeric_non_missing_count = int(numeric_values.notna().sum())

        if source_non_missing_count != numeric_non_missing_count:
            non_numeric_features.append(feature)

        finite_values = numeric_values.dropna()

        infinite_value_count += int((~np.isfinite(finite_values)).sum())

    validation_passed = bool(
        not missing_columns
        and duplicate_pr_count == 0
        and not non_numeric_features
        and infinite_value_count == 0
        and len(dataframe) > 0
    )

    return {
        "row_count": int(len(dataframe)),
        "required_feature_count": int(len(required_features)),
        "missing_columns": (missing_columns),
        "duplicate_pr_count": (duplicate_pr_count),
        "non_numeric_features": (non_numeric_features),
        "infinite_value_count": int(infinite_value_count),
        "validation_passed": (validation_passed),
    }


def transform_model_features(
    dataframe: pd.DataFrame,
    preprocessor: Any,
    raw_feature_names: list[str],
    model_feature_names: list[str],
) -> pd.DataFrame:
    """Apply a fitted preprocessor without refitting it."""

    missing_features = sorted(set(raw_feature_names) - set(dataframe.columns))

    if missing_features:
        raise ValueError(
            f"Scoring source is missing preprocessor features: {missing_features}"
        )

    raw_features = dataframe[raw_feature_names].apply(
        pd.to_numeric,
        errors="coerce",
    )

    transformed_values = preprocessor.transform(raw_features)

    transformed_array = np.asarray(
        transformed_values,
        dtype=float,
    )

    if not np.isfinite(transformed_array).all():
        raise ValueError(
            "Transformed scoring features contain missing or infinite values."
        )

    if transformed_array.shape[1] != len(model_feature_names):
        raise ValueError(
            "Transformed feature count does not match the locked model feature count."
        )

    return pd.DataFrame(
        transformed_array,
        columns=model_feature_names,
        index=dataframe.index,
    )


def score_binary_model(
    dataframe: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    configuration: dict[str, Any],
    probability_column: str,
    prediction_column: str,
    confidence_column: str,
    threshold_column: str,
) -> pd.DataFrame:
    """Score a fitted binary model without retraining or threshold changes."""

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError("The locked model does not support predict_proba.")

    configured_features = resolve_configuration_features(configuration)

    raw_feature_names = resolve_preprocessor_input_features(
        preprocessor=preprocessor,
        fallback_features=(configured_features),
    )

    model_feature_names = resolve_model_feature_order(
        model=model,
        preprocessor=preprocessor,
        configured_features=(configured_features),
    )

    source_validation = validate_scoring_source(
        dataframe=dataframe,
        required_features=(raw_feature_names),
    )

    if not source_validation["validation_passed"]:
        raise ValueError(f"Model scoring source failed validation: {source_validation}")

    transformed_features = transform_model_features(
        dataframe=dataframe,
        preprocessor=preprocessor,
        raw_feature_names=(raw_feature_names),
        model_feature_names=(model_feature_names),
    )

    probabilities = model.predict_proba(transformed_features)[:, 1]

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if not np.isfinite(probability_array).all():
        raise ValueError("Model probabilities contain missing or infinite values.")

    if not ((probability_array >= 0) & (probability_array <= 1)).all():
        raise ValueError("Model probabilities must be between zero and one.")

    threshold = resolve_configuration_threshold(configuration)

    predictions = (probability_array >= threshold).astype(int)

    confidence = np.where(
        predictions == 1,
        probability_array,
        1 - probability_array,
    )

    model_name = str(
        configuration.get(
            "model_name",
            configuration.get(
                "candidate_id",
                "locked_binary_model",
            ),
        )
    )

    return pd.DataFrame(
        {
            PR_IDENTIFIER_COLUMN: (
                dataframe[PR_IDENTIFIER_COLUMN].reset_index(drop=True)
            ),
            probability_column: (probability_array),
            prediction_column: (predictions),
            confidence_column: (confidence),
            threshold_column: (threshold),
            f"{prediction_column}_model_name": (model_name),
        }
    )


def calculate_review_priority_score(
    policy_risk_score: float,
    merge_probability: float,
    delay_probability: float | None,
) -> float:
    """Calculate the unified review-priority score."""

    normalized_policy_score = float(
        np.clip(
            policy_risk_score,
            0,
            100,
        )
    )

    normalized_merge_probability = float(
        np.clip(
            merge_probability,
            0,
            1,
        )
    )

    merge_blocker_component = (1 - normalized_merge_probability) * 20

    delay_component = 0.0

    if delay_probability is not None and not pd.isna(delay_probability):
        normalized_delay_probability = float(
            np.clip(
                delay_probability,
                0,
                1,
            )
        )

        delay_component = normalized_delay_probability * 15

    final_score = (
        normalized_policy_score * 0.65 + merge_blocker_component + delay_component
    )

    return float(
        round(
            min(
                final_score,
                100.0,
            ),
            4,
        )
    )


def classify_review_priority(
    review_priority_score: float,
    policy_risk_band: str,
    manual_review_required: bool,
) -> str:
    """Convert the unified score into a review-priority band."""

    normalized_policy_band = str(policy_risk_band).strip().lower()

    if normalized_policy_band == "critical" or review_priority_score >= 70:
        return "Critical"

    if (
        normalized_policy_band == "high"
        or manual_review_required
        or review_priority_score >= 50
    ):
        return "High"

    if review_priority_score >= 25:
        return "Moderate"

    return "Routine"


def recommend_next_action(
    policy_risk_band: str,
    manual_review_required: bool,
    merge_prediction: int,
    merge_probability: float,
    delay_prediction: int | float | None,
    delay_probability: float | None,
) -> str:
    """Return one concise operational recommendation."""

    normalized_policy_band = str(policy_risk_band).strip().lower()

    if normalized_policy_band == "critical":
        return (
            "Escalate for mandatory policy, security or senior "
            "maintainer review before approval."
        )

    if normalized_policy_band == "high" or manual_review_required:
        return (
            "Complete the triggered policy checks and obtain manual "
            "review before progressing."
        )

    if int(merge_prediction) == 0 or float(merge_probability) < 0.40:
        return (
            "Address likely merge blockers and strengthen the PR "
            "description, tests or reviewer readiness."
        )

    delay_is_positive = bool(
        delay_prediction is not None
        and not pd.isna(delay_prediction)
        and int(delay_prediction) == 1
    )

    delay_is_high = bool(
        delay_probability is not None
        and not pd.isna(delay_probability)
        and float(delay_probability) >= 0.75
    )

    if delay_is_positive or delay_is_high:
        return (
            "Prioritize reviewer assignment and follow-up because "
            "the PR has elevated merge-delay risk."
        )

    return (
        "Proceed through the standard review workflow and monitor "
        "for new policy or delay signals."
    )


def build_unified_dataset(
    core_dataframe: pd.DataFrame,
    merge_scores: pd.DataFrame,
    delay_scores: pd.DataFrame,
    policy_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Combine core data, model outputs and policy intelligence."""

    required_core_columns = {
        PR_IDENTIFIER_COLUMN,
    }

    required_merge_columns = {
        PR_IDENTIFIER_COLUMN,
        "merge_probability",
        "merge_prediction",
        "merge_prediction_confidence",
        "merge_prediction_threshold",
    }

    required_delay_columns = {
        PR_IDENTIFIER_COLUMN,
        "delay_probability",
        "delay_prediction",
        "delay_prediction_confidence",
        "delay_prediction_threshold",
    }

    required_policy_columns = {
        PR_IDENTIFIER_COLUMN,
        "policy_risk_score",
        "policy_risk_band",
        "triggered_rule_count",
        "triggered_rules",
        "triggered_categories",
        "recommended_actions",
        "manual_review_required",
    }

    dataset_requirements = [
        (
            "core",
            core_dataframe,
            required_core_columns,
        ),
        (
            "merge scores",
            merge_scores,
            required_merge_columns,
        ),
        (
            "delay scores",
            delay_scores,
            required_delay_columns,
        ),
        (
            "policy scores",
            policy_scores,
            required_policy_columns,
        ),
    ]

    for (
        dataset_name,
        dataframe,
        required_columns,
    ) in dataset_requirements:
        missing_columns = sorted(required_columns - set(dataframe.columns))

        if missing_columns:
            raise ValueError(
                f"{dataset_name} dataset is missing columns: {missing_columns}"
            )

        duplicate_count = int(dataframe.duplicated(subset=[PR_IDENTIFIER_COLUMN]).sum())

        if duplicate_count > 0:
            raise ValueError(
                f"{dataset_name} contains {duplicate_count} duplicate PR identifiers."
            )

    unified = (
        core_dataframe.merge(
            merge_scores,
            on=PR_IDENTIFIER_COLUMN,
            how="left",
            validate="one_to_one",
        )
        .merge(
            delay_scores,
            on=PR_IDENTIFIER_COLUMN,
            how="left",
            validate="one_to_one",
        )
        .merge(
            policy_scores,
            on=PR_IDENTIFIER_COLUMN,
            how="left",
            validate="one_to_one",
        )
    )

    if unified["merge_probability"].isna().any():
        raise ValueError("One or more PRs are missing Model 1 merge scores.")

    if unified["policy_risk_score"].isna().any():
        raise ValueError("One or more PRs are missing deterministic policy scores.")

    unified["delay_score_available"] = unified["delay_probability"].notna()

    unified["review_priority_score"] = [
        calculate_review_priority_score(
            policy_risk_score=(policy_score),
            merge_probability=(merge_probability),
            delay_probability=(delay_probability),
        )
        for (
            policy_score,
            merge_probability,
            delay_probability,
        ) in zip(
            unified["policy_risk_score"],
            unified["merge_probability"],
            unified["delay_probability"],
            strict=True,
        )
    ]

    unified["review_priority"] = [
        classify_review_priority(
            review_priority_score=(priority_score),
            policy_risk_band=(policy_band),
            manual_review_required=bool(manual_review),
        )
        for (
            priority_score,
            policy_band,
            manual_review,
        ) in zip(
            unified["review_priority_score"],
            unified["policy_risk_band"],
            unified["manual_review_required"],
            strict=True,
        )
    ]

    unified["recommended_next_action"] = [
        recommend_next_action(
            policy_risk_band=(policy_band),
            manual_review_required=bool(manual_review),
            merge_prediction=int(merge_prediction),
            merge_probability=float(merge_probability),
            delay_prediction=(delay_prediction),
            delay_probability=(delay_probability),
        )
        for (
            policy_band,
            manual_review,
            merge_prediction,
            merge_probability,
            delay_prediction,
            delay_probability,
        ) in zip(
            unified["policy_risk_band"],
            unified["manual_review_required"],
            unified["merge_prediction"],
            unified["merge_probability"],
            unified["delay_prediction"],
            unified["delay_probability"],
            strict=True,
        )
    ]

    priority_order = {
        "Critical": 1,
        "High": 2,
        "Moderate": 3,
        "Routine": 4,
    }

    unified["review_priority_order"] = unified["review_priority"].map(priority_order)

    unified = unified.sort_values(
        [
            "review_priority_order",
            "review_priority_score",
            "policy_risk_score",
            PR_IDENTIFIER_COLUMN,
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    return unified


def build_priority_summary(
    unified_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the unified review-priority distribution."""

    required_columns = {
        "review_priority",
        "review_priority_score",
        "policy_risk_score",
        "merge_probability",
        "delay_probability",
        "delay_score_available",
        "manual_review_required",
    }

    missing_columns = sorted(required_columns - set(unified_dataframe.columns))

    if missing_columns:
        raise ValueError(
            f"Unified dataset is missing priority columns: {missing_columns}"
        )

    summary = (
        unified_dataframe.groupby(
            "review_priority",
            observed=False,
        )
        .agg(
            pr_count=(
                "review_priority",
                "size",
            ),
            average_priority_score=(
                "review_priority_score",
                "mean",
            ),
            average_policy_score=(
                "policy_risk_score",
                "mean",
            ),
            average_merge_probability=(
                "merge_probability",
                "mean",
            ),
            average_delay_probability=(
                "delay_probability",
                "mean",
            ),
            delay_score_available_count=(
                "delay_score_available",
                "sum",
            ),
            manual_review_count=(
                "manual_review_required",
                "sum",
            ),
        )
        .reset_index()
    )

    priority_order = {
        "Critical": 1,
        "High": 2,
        "Moderate": 3,
        "Routine": 4,
    }

    summary["_priority_order"] = summary["review_priority"].map(priority_order)

    return (
        summary.sort_values("_priority_order")
        .drop(columns=["_priority_order"])
        .reset_index(drop=True)
    )


def validate_unified_outputs(
    core_dataframe: pd.DataFrame,
    unified_dataframe: pd.DataFrame,
    merge_scores: pd.DataFrame,
    delay_scores: pd.DataFrame,
    policy_scores: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the unified PR intelligence dataset."""

    expected_row_count = int(len(core_dataframe))

    expected_delay_score_count = int(len(delay_scores))

    row_count_valid = bool(len(unified_dataframe) == expected_row_count)

    unique_pr_count_valid = bool(
        unified_dataframe[PR_IDENTIFIER_COLUMN].nunique() == expected_row_count
    )

    merge_score_count_valid = bool(
        merge_scores[PR_IDENTIFIER_COLUMN].nunique() == expected_row_count
        and unified_dataframe["merge_probability"].notna().sum() == expected_row_count
    )

    delay_score_count_valid = bool(
        unified_dataframe["delay_score_available"].astype(bool).sum()
        == expected_delay_score_count
    )

    policy_score_count_valid = bool(
        policy_scores[PR_IDENTIFIER_COLUMN].nunique() == expected_row_count
        and unified_dataframe["policy_risk_score"].notna().sum() == expected_row_count
    )

    probability_ranges_valid = bool(
        unified_dataframe["merge_probability"]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
        and unified_dataframe.loc[
            unified_dataframe["delay_score_available"].astype(bool),
            "delay_probability",
        ]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
    )

    priority_score_range_valid = bool(
        unified_dataframe["review_priority_score"]
        .between(
            0,
            100,
            inclusive="both",
        )
        .all()
    )

    priority_values_valid = bool(
        set(unified_dataframe["review_priority"].unique()).issubset(
            {
                "Routine",
                "Moderate",
                "High",
                "Critical",
            }
        )
    )

    recommendation_complete = bool(
        unified_dataframe["recommended_next_action"]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )

    validation_passed = bool(
        row_count_valid
        and unique_pr_count_valid
        and merge_score_count_valid
        and delay_score_count_valid
        and policy_score_count_valid
        and probability_ranges_valid
        and priority_score_range_valid
        and priority_values_valid
        and recommendation_complete
    )

    return {
        "expected_row_count": (expected_row_count),
        "actual_row_count": int(len(unified_dataframe)),
        "row_count_valid": (row_count_valid),
        "unique_pr_count_valid": (unique_pr_count_valid),
        "expected_merge_score_count": (expected_row_count),
        "actual_merge_score_count": int(
            unified_dataframe["merge_probability"].notna().sum()
        ),
        "merge_score_count_valid": (merge_score_count_valid),
        "expected_delay_score_count": (expected_delay_score_count),
        "actual_delay_score_count": int(
            unified_dataframe["delay_score_available"].astype(bool).sum()
        ),
        "delay_score_count_valid": (delay_score_count_valid),
        "policy_score_count_valid": (policy_score_count_valid),
        "probability_ranges_valid": (probability_ranges_valid),
        "priority_score_range_valid": (priority_score_range_valid),
        "priority_values_valid": (priority_values_valid),
        "recommendation_complete": (recommendation_complete),
        "validation_passed": (validation_passed),
    }
