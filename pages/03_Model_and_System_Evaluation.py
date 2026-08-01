"""Combined machine-learning, retrieval and governed-AI evaluation page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config.settings import get_settings
from src.ui.streamlit_data import (
    apply_global_page_style,
    extract_report_status,
    flatten_metrics,
    load_csv,
    load_json,
    render_project_sidebar,
)
from src.utils.logging import get_logger
from src.utils.paths import create_required_directories


settings = get_settings()
logger = get_logger(__name__)

create_required_directories()
apply_global_page_style()
render_project_sidebar(
    settings=settings,
    current_page="Model & System Evaluation",
)

logger.info("Model & System Evaluation page loaded.")

st.title("Model & System Evaluation")
st.caption(
    "Evaluate predictive models, retrieval quality, citation safety, "
    "groundedness, deterministic repair and end-to-end behaviour."
)


model1_report = load_json(
    "data/reports/stage_5f_test_metrics.json"
)

model2_report = load_json(
    "data/reports/stage_6f_test_metrics.json"
)

retrieval_report = load_json(
    "data/reports/stage_8g_governed_retrieval_evaluation.json"
)

production_report = load_json(
    "data/reports/stage_10f_real_deterministic_repair.json"
)

model1_predictions = load_csv(
    "data/reports/stage_5f_test_predictions.csv"
)

model2_predictions = load_csv(
    "data/reports/stage_6f_test_predictions.csv"
)

model1_importance = load_csv(
    "data/reports/stage_5g_permutation_importance.csv"
)

if model1_importance.empty:
    model1_importance = load_csv(
        "data/reports/stage_6g_permutation_importance.csv"
    )


st.markdown("## Predictive model results")

model_columns = st.columns(2)

with model_columns[0]:
    st.markdown("### Merge-outcome model")

    metrics = flatten_metrics(
        model1_report
    )

    preferred_keys = [
        key
        for key in metrics
        if any(
            name in key.lower()
            for name in [
                "accuracy",
                "balanced",
                "precision",
                "recall",
                "f1",
                "roc",
                "average_precision",
                "log_loss",
                "brier",
            ]
        )
    ]

    if preferred_keys:
        display_metrics = {
            key.split(".")[-1]: metrics[key]
            for key in preferred_keys[:12]
        }

        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": display_metrics.keys(),
                    "Value": display_metrics.values(),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Merge-outcome metrics report was not found."
        )

with model_columns[1]:
    st.markdown("### Merge-delay model")

    metrics = flatten_metrics(
        model2_report
    )

    preferred_keys = [
        key
        for key in metrics
        if any(
            name in key.lower()
            for name in [
                "accuracy",
                "balanced",
                "precision",
                "recall",
                "f1",
                "roc",
                "average_precision",
                "log_loss",
                "brier",
            ]
        )
    ]

    if preferred_keys:
        display_metrics = {
            key.split(".")[-1]: metrics[key]
            for key in preferred_keys[:12]
        }

        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": display_metrics.keys(),
                    "Value": display_metrics.values(),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Merge-delay metrics report was not found."
        )


st.markdown("## Prediction diagnostics")

diagnostic_columns = st.columns(2)

with diagnostic_columns[0]:
    st.markdown("### Merge-outcome test predictions")

    if not model1_predictions.empty:
        probability_column = next(
            (
                column
                for column in [
                    "predicted_probability",
                    "merge_probability",
                    "probability",
                ]
                if column in model1_predictions.columns
            ),
            None,
        )

        if probability_column:
            probability_band = pd.cut(
                pd.to_numeric(
                    model1_predictions[probability_column],
                    errors="coerce",
                ),
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                include_lowest=True,
            )

            st.bar_chart(
                probability_band.value_counts(sort=False)
                .rename_axis("Probability band")
                .to_frame("Test rows"),
                use_container_width=True,
            )

        st.dataframe(
            model1_predictions.head(100),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Merge-outcome test prediction file was not found."
        )

with diagnostic_columns[1]:
    st.markdown("### Merge-delay test predictions")

    if not model2_predictions.empty:
        probability_column = next(
            (
                column
                for column in [
                    "predicted_probability",
                    "delay_probability",
                    "probability",
                ]
                if column in model2_predictions.columns
            ),
            None,
        )

        if probability_column:
            probability_band = pd.cut(
                pd.to_numeric(
                    model2_predictions[probability_column],
                    errors="coerce",
                ),
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                include_lowest=True,
            )

            st.bar_chart(
                probability_band.value_counts(sort=False)
                .rename_axis("Probability band")
                .to_frame("Test rows"),
                use_container_width=True,
            )

        st.dataframe(
            model2_predictions.head(100),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Merge-delay test prediction file was not found."
        )


if not model1_importance.empty:
    st.markdown("## Explainability")

    importance_column = next(
        (
            column
            for column in [
                "importance_mean",
                "permutation_importance_mean",
                "importance",
            ]
            if column in model1_importance.columns
        ),
        None,
    )

    feature_column = next(
        (
            column
            for column in [
                "feature",
                "feature_name",
                "column",
            ]
            if column in model1_importance.columns
        ),
        None,
    )

    if importance_column and feature_column:
        top_importance = (
            model1_importance[
                [feature_column, importance_column]
            ]
            .assign(
                **{
                    importance_column: pd.to_numeric(
                        model1_importance[importance_column],
                        errors="coerce",
                    )
                }
            )
            .dropna()
            .sort_values(
                importance_column,
                ascending=False,
            )
            .head(15)
            .set_index(feature_column)
        )

        st.bar_chart(
            top_importance,
            use_container_width=True,
        )


st.markdown("## Governed retrieval evaluation")

if retrieval_report:
    retrieval_status = extract_report_status(
        retrieval_report
    )

    retrieval_metrics = flatten_metrics(
        retrieval_report
    )

    retrieval_metric_columns = st.columns(5)

    candidate_patterns = [
        ("Hit Rate@5", ["hit_rate_at_5", "hit_rate"]),
        ("Precision@5", ["precision_at_5", "precision"]),
        ("MRR", ["mean_reciprocal_rank", "mrr"]),
        ("Negative abstention", ["negative_abstention_rate"]),
        ("Mean latency", ["mean_positive_query_latency", "mean_latency"]),
    ]

    for metric_column, (label, patterns) in zip(
        retrieval_metric_columns,
        candidate_patterns,
    ):
        matching_value = None

        for key, value in retrieval_metrics.items():
            if any(
                pattern in key.lower()
                for pattern in patterns
            ):
                matching_value = value
                break

        if matching_value is None:
            metric_column.metric(
                label,
                "Not available",
            )
        elif "latency" in label.lower():
            metric_column.metric(
                label,
                f"{matching_value:.2f} ms",
            )
        else:
            metric_column.metric(
                label,
                f"{matching_value:.1%}",
            )

    st.caption(
        f"Retrieval evaluation status: {retrieval_status}"
    )

    with st.expander(
        "Complete retrieval report",
        expanded=False,
    ):
        st.json(
            retrieval_report
        )
else:
    st.info(
        "Stage 8G retrieval evaluation report was not found."
    )


st.markdown("## End-to-end governed AI evaluation")

if production_report:
    production_metric_columns = st.columns(6)

    production_metric_columns[0].metric(
        "Safe pipeline rate",
        f"{float(production_report.get('safe_pipeline_rate', 0)):.1%}",
    )

    production_metric_columns[1].metric(
        "Answer release rate",
        f"{float(production_report.get('answer_release_rate', 0)):.1%}",
    )

    production_metric_columns[2].metric(
        "Repair success rate",
        f"{float(production_report.get('repair_success_rate', 0)):.1%}",
    )

    production_metric_columns[3].metric(
        "Citation blocks",
        str(
            production_report.get(
                "citation_block_count",
                0,
            )
        ),
    )

    production_metric_columns[4].metric(
        "Groundedness blocks",
        str(
            production_report.get(
                "groundedness_block_count",
                0,
            )
        ),
    )

    production_metric_columns[5].metric(
        "Mean repair latency",
        (
            f"{float(production_report.get('mean_repair_latency_ms', 0)):.3f} ms"
        ),
    )

    with st.expander(
        "Complete production evaluation report",
        expanded=False,
    ):
        st.json(
            production_report
        )
else:
    st.info(
        "Stage 10F production evaluation report was not found."
    )


st.markdown("## Evaluation interpretation")

st.success(
    "A result is released only when citation validation and "
    "claim-to-evidence groundedness both pass. Unsupported answers are "
    "withheld, and out-of-domain questions skip generation."
)

st.warning(
    "Model metrics describe historical test performance, not guaranteed "
    "future outcomes. Predictions remain decision-support signals."
)
