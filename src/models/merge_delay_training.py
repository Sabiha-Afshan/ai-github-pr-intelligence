"""Baseline model training for Model 2 merge-delay prediction."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.merge_delay_evaluation import (
    build_prediction_table,
    build_threshold_table,
    calculate_binary_metrics,
    select_validation_threshold,
)


PR_IDENTIFIER_COLUMN = "pr_number"
SPLIT_COLUMN = "split"
TARGET_COLUMN = "merge_delay_target"


@dataclass(frozen=True)
class Model2Split:
    """One preprocessed Model 2 split."""

    identifiers: pd.Series
    targets: pd.Series
    features: pd.DataFrame


@dataclass(frozen=True)
class TrainedMergeDelayModel:
    """A trained Model 2 baseline and its validation results."""

    model_name: str
    model: Any
    threshold: float
    training_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    threshold_table: pd.DataFrame
    validation_predictions: pd.DataFrame


def load_preprocessed_split(
    dataframe: pd.DataFrame,
    expected_split: str,
) -> Model2Split:
    """Extract identifiers, targets and features from one split."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = sorted(
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Preprocessed split is missing columns: "
            f"{missing_columns}"
        )

    split_values = (
        dataframe[
            SPLIT_COLUMN
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    if split_values != [
        expected_split
    ]:
        raise ValueError(
            f"Expected only {expected_split!r} rows, "
            f"but found {split_values}."
        )

    feature_columns = [
        column
        for column in dataframe.columns
        if column
        not in required_columns
    ]

    if not feature_columns:
        raise ValueError(
            "No preprocessed features were found."
        )

    identifiers = (
        dataframe[
            PR_IDENTIFIER_COLUMN
        ]
        .reset_index(drop=True)
    )

    targets = (
        pd.to_numeric(
            dataframe[
                TARGET_COLUMN
            ],
            errors="raise",
        )
        .astype(int)
        .reset_index(drop=True)
    )

    features = (
        dataframe[
            feature_columns
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .astype(float)
        .reset_index(drop=True)
    )

    feature_array = features.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        feature_array
    ).all():
        raise ValueError(
            "Preprocessed features contain missing or infinite values."
        )

    if set(
        targets.unique()
    ) != {0, 1}:
        raise ValueError(
            f"{expected_split} split must contain both classes."
        )

    return Model2Split(
        identifiers=identifiers,
        targets=targets,
        features=features,
    )


def validate_split_compatibility(
    training_split: Model2Split,
    validation_split: Model2Split,
) -> dict[str, Any]:
    """Confirm training and validation feature schemas match."""

    training_columns = list(
        training_split.features.columns
    )

    validation_columns = list(
        validation_split.features.columns
    )

    feature_columns_match = (
        training_columns
        == validation_columns
    )

    validation_passed = (
        feature_columns_match
        and len(training_split.features)
        > 0
        and len(validation_split.features)
        > 0
    )

    return {
        "training_rows": len(
            training_split.features
        ),
        "validation_rows": len(
            validation_split.features
        ),
        "feature_count": len(
            training_columns
        ),
        "feature_columns_match": (
            feature_columns_match
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def train_and_evaluate_model(
    model_name: str,
    model: Any,
    training_split: Model2Split,
    validation_split: Model2Split,
) -> TrainedMergeDelayModel:
    """Train one model and select its validation threshold."""

    model.fit(
        training_split.features,
        training_split.targets,
    )

    training_probabilities = (
        model.predict_proba(
            training_split.features
        )[:, 1]
    )

    validation_probabilities = (
        model.predict_proba(
            validation_split.features
        )[:, 1]
    )

    threshold_table = build_threshold_table(
        actual=(
            validation_split.targets
        ),
        probabilities=(
            validation_probabilities
        ),
        minimum_threshold=0.20,
        maximum_threshold=0.80,
        threshold_step=0.025,
    )

    selected_validation_metrics = (
        select_validation_threshold(
            threshold_table
        )
    )

    threshold = float(
        selected_validation_metrics[
            "threshold"
        ]
    )

    training_metrics = (
        calculate_binary_metrics(
            actual=(
                training_split.targets
            ),
            probabilities=(
                training_probabilities
            ),
            threshold=threshold,
        )
    )

    validation_metrics = (
        calculate_binary_metrics(
            actual=(
                validation_split.targets
            ),
            probabilities=(
                validation_probabilities
            ),
            threshold=threshold,
        )
    )

    training_metrics[
        "model_name"
    ] = model_name

    validation_metrics[
        "model_name"
    ] = model_name

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
            threshold=threshold,
            model_name=model_name,
            split_name="validation",
        )
    )

    return TrainedMergeDelayModel(
        model_name=model_name,
        model=model,
        threshold=threshold,
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


def train_logistic_regression(
    training_split: Model2Split,
    validation_split: Model2Split,
) -> TrainedMergeDelayModel:
    """Train the interpretable Model 2 baseline."""

    model = LogisticRegression(
        class_weight="balanced",
        penalty="l2",
        solver="liblinear",
        max_iter=3000,
        random_state=42,
    )

    return train_and_evaluate_model(
        model_name=(
            "merge_delay_logistic_regression"
        ),
        model=model,
        training_split=training_split,
        validation_split=validation_split,
    )


def train_random_forest(
    training_split: Model2Split,
    validation_split: Model2Split,
) -> TrainedMergeDelayModel:
    """Train the contributor-neutral Random Forest baseline."""

    model = RandomForestClassifier(
        n_estimators=800,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight=(
            "balanced_subsample"
        ),
        bootstrap=True,
        max_samples=0.85,
        random_state=42,
        n_jobs=-1,
    )

    return train_and_evaluate_model(
        model_name=(
            "merge_delay_random_forest"
        ),
        model=model,
        training_split=training_split,
        validation_split=validation_split,
    )