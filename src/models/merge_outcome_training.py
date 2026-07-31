"""Training utilities for merge-outcome models."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
)

from src.models.merge_outcome_evaluation import (
    build_threshold_table,
    create_prediction_table,
    get_positive_probability,
    select_validation_threshold,
)

RANDOM_STATE = 42

NEUTRAL_LOGISTIC_MODEL_NAME = "contributor_neutral_logistic_regression"

NEUTRAL_RANDOM_FOREST_MODEL_NAME = "contributor_neutral_random_forest"

AUTHOR_RANDOM_FOREST_MODEL_NAME = "author_association_random_forest_benchmark"


@dataclass(frozen=True)
class TrainedModelResult:
    """Result produced for one trained model."""

    model_name: str
    model: Any
    selected_threshold: float
    validation_metrics: dict[str, Any]
    threshold_table: pd.DataFrame
    predictions: pd.DataFrame
    training_seconds: float


def build_neutral_logistic_regression() -> LogisticRegression:
    """Build the contributor-neutral Logistic Regression model."""

    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def build_neutral_random_forest() -> RandomForestClassifier:
    """Build the contributor-neutral Random Forest model."""

    return RandomForestClassifier(
        n_estimators=750,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_author_benchmark_pipeline(
    numeric_features: list[str],
) -> Pipeline:
    """Build the author-association Random Forest benchmark."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "most_frequent_imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "author_association",
                categorical_pipeline,
                ["author_association"],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    classifier = RandomForestClassifier(
        n_estimators=750,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def train_and_validate_model(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_identifiers: pd.Series,
) -> TrainedModelResult:
    """Train one model and select its validation threshold."""

    if len(x_train) != len(y_train):
        raise ValueError("Training features and targets are misaligned.")

    if len(x_validation) != len(y_validation):
        raise ValueError("Validation features and targets are misaligned.")

    training_started = perf_counter()

    model.fit(
        x_train,
        y_train,
    )

    training_seconds = perf_counter() - training_started

    probabilities = get_positive_probability(
        model,
        x_validation,
    )

    threshold_table = build_threshold_table(
        targets=y_validation,
        probabilities=probabilities,
    )

    selected_metrics = select_validation_threshold(threshold_table)

    selected_threshold = float(selected_metrics["threshold"])

    selected_metrics.update(
        {
            "model_name": model_name,
            "validation_rows": len(y_validation),
            "training_rows": len(y_train),
            "training_seconds": float(training_seconds),
        }
    )

    predictions = create_prediction_table(
        identifiers=(validation_identifiers),
        targets=y_validation,
        probabilities=probabilities,
        threshold=selected_threshold,
        model_name=model_name,
    )

    threshold_table.insert(
        0,
        "model_name",
        model_name,
    )

    threshold_table["selected_threshold"] = np.isclose(
        threshold_table["threshold"],
        selected_threshold,
    )

    return TrainedModelResult(
        model_name=model_name,
        model=model,
        selected_threshold=(selected_threshold),
        validation_metrics=(selected_metrics),
        threshold_table=(threshold_table),
        predictions=predictions,
        training_seconds=float(training_seconds),
    )


def train_merge_outcome_models(
    neutral_x_train: pd.DataFrame,
    neutral_y_train: pd.Series,
    neutral_x_validation: pd.DataFrame,
    neutral_y_validation: pd.Series,
    validation_identifiers: pd.Series,
    author_x_train: pd.DataFrame,
    author_x_validation: pd.DataFrame,
) -> dict[str, TrainedModelResult]:
    """Train the three required merge-outcome models."""

    if list(neutral_x_train.columns) != list(neutral_x_validation.columns):
        raise ValueError("Neutral train and validation columns differ.")

    expected_author_columns = {
        *neutral_x_train.columns,
        "author_association",
    }

    if set(author_x_train.columns) != expected_author_columns:
        raise ValueError("Author benchmark training columns are incorrect.")

    if set(author_x_validation.columns) != expected_author_columns:
        raise ValueError("Author benchmark validation columns are incorrect.")

    logistic_result = train_and_validate_model(
        model_name=(NEUTRAL_LOGISTIC_MODEL_NAME),
        model=(build_neutral_logistic_regression()),
        x_train=neutral_x_train,
        y_train=neutral_y_train,
        x_validation=(neutral_x_validation),
        y_validation=(neutral_y_validation),
        validation_identifiers=(validation_identifiers),
    )

    neutral_forest_result = train_and_validate_model(
        model_name=(NEUTRAL_RANDOM_FOREST_MODEL_NAME),
        model=(build_neutral_random_forest()),
        x_train=neutral_x_train,
        y_train=neutral_y_train,
        x_validation=(neutral_x_validation),
        y_validation=(neutral_y_validation),
        validation_identifiers=(validation_identifiers),
    )

    author_benchmark = build_author_benchmark_pipeline(
        numeric_features=list(neutral_x_train.columns)
    )

    author_forest_result = train_and_validate_model(
        model_name=(AUTHOR_RANDOM_FOREST_MODEL_NAME),
        model=author_benchmark,
        x_train=author_x_train,
        y_train=neutral_y_train,
        x_validation=(author_x_validation),
        y_validation=(neutral_y_validation),
        validation_identifiers=(validation_identifiers),
    )

    return {
        logistic_result.model_name: (logistic_result),
        neutral_forest_result.model_name: (neutral_forest_result),
        author_forest_result.model_name: (author_forest_result),
    }


def build_model_comparison(
    results: dict[
        str,
        TrainedModelResult,
    ],
) -> pd.DataFrame:
    """Build the validation model-comparison table."""

    records = [result.validation_metrics for result in results.values()]

    comparison = pd.DataFrame(records)

    comparison["model_role"] = comparison["model_name"].map(
        {
            NEUTRAL_LOGISTIC_MODEL_NAME: ("Interpretable neutral baseline"),
            NEUTRAL_RANDOM_FOREST_MODEL_NAME: ("Final contributor-neutral candidate"),
            AUTHOR_RANDOM_FOREST_MODEL_NAME: ("Fairness and shortcut benchmark"),
        }
    )

    comparison["eligible_for_final_application"] = (
        comparison["model_name"] == NEUTRAL_RANDOM_FOREST_MODEL_NAME
    )

    comparison = comparison.sort_values(
        [
            "roc_auc",
            "f1",
            "balanced_accuracy",
        ],
        ascending=[
            False,
            False,
            False,
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

    return comparison


def validate_training_results(
    results: dict[
        str,
        TrainedModelResult,
    ],
    expected_validation_rows: int,
) -> dict[str, Any]:
    """Validate the complete Stage 5D result."""

    expected_models = {
        NEUTRAL_LOGISTIC_MODEL_NAME,
        NEUTRAL_RANDOM_FOREST_MODEL_NAME,
        AUTHOR_RANDOM_FOREST_MODEL_NAME,
    }

    actual_models = set(results)

    prediction_counts = {
        model_name: len(result.predictions) for model_name, result in results.items()
    }

    thresholds = {
        model_name: (result.selected_threshold)
        for model_name, result in results.items()
    }

    metrics_valid = all(
        0 <= float(result.validation_metrics[metric_name]) <= 1
        for result in results.values()
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

    thresholds_valid = all(0 < threshold < 1 for threshold in thresholds.values())

    prediction_counts_valid = all(
        count == expected_validation_rows for count in (prediction_counts.values())
    )

    validation_passed = (
        actual_models == expected_models
        and metrics_valid
        and thresholds_valid
        and prediction_counts_valid
    )

    return {
        "expected_models": sorted(expected_models),
        "actual_models": sorted(actual_models),
        "selected_thresholds": thresholds,
        "prediction_counts": (prediction_counts),
        "metrics_valid": metrics_valid,
        "thresholds_valid": (thresholds_valid),
        "prediction_counts_valid": (prediction_counts_valid),
        "validation_passed": (validation_passed),
    }
