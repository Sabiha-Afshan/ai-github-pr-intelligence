"""Reusable Plotly charts for GitHub PR exploratory analysis."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

OUTCOME_ORDER = [
    "Merged",
    "Unmerged",
]

SIZE_BAND_ORDER = [
    "Very small",
    "Small",
    "Medium",
    "Large",
    "Very large",
]

LIFECYCLE_BAND_ORDER = [
    "Under 1 hour",
    "1-24 hours",
    "1-7 days",
    "8-30 days",
    "31-90 days",
    "Over 90 days",
]


def apply_chart_layout(
    figure: go.Figure,
    title: str,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    height: int = 500,
) -> go.Figure:
    """Apply a consistent layout to a Plotly chart."""

    figure.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        margin={
            "l": 50,
            "r": 30,
            "t": 80,
            "b": 50,
        },
        legend_title_text="",
        hovermode="closest",
        template="plotly_white",
    )

    return figure


def create_outcome_distribution_chart(
    dataframe: pd.DataFrame,
) -> go.Figure:
    """Create a merged-versus-unmerged distribution chart."""

    if "merge_outcome" not in dataframe.columns:
        raise ValueError("merge_outcome column is missing.")

    summary = (
        dataframe["merge_outcome"]
        .value_counts()
        .reindex(
            OUTCOME_ORDER,
            fill_value=0,
        )
        .rename_axis("merge_outcome")
        .reset_index(name="pr_count")
    )

    total_prs = int(summary["pr_count"].sum())

    summary["percentage"] = (
        summary["pr_count"].div(total_prs).mul(100).round(2) if total_prs > 0 else 0.0
    )

    figure = px.bar(
        summary,
        x="merge_outcome",
        y="pr_count",
        text="pr_count",
        custom_data=[
            "percentage",
        ],
    )

    figure.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "PRs: %{y}<br>"
            "Dataset share: "
            "%{customdata[0]:.2f}%"
            "<extra></extra>"
        ),
    )

    return apply_chart_layout(
        figure,
        title="Pull Request Outcome Distribution",
        xaxis_title="Outcome",
        yaxis_title="Number of PRs",
    )


def create_quarterly_volume_chart(
    quarterly_metrics: pd.DataFrame,
) -> go.Figure:
    """Create quarterly merged and unmerged PR trends."""

    required_columns = {
        "period",
        "merged_prs",
        "unmerged_prs",
    }

    missing_columns = required_columns - set(quarterly_metrics.columns)

    if missing_columns:
        raise ValueError(
            f"Quarterly metrics are missing columns: {sorted(missing_columns)}"
        )

    chart_data = quarterly_metrics.copy().sort_values("period").reset_index(drop=True)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data["period"],
            y=chart_data["merged_prs"],
            mode="lines+markers",
            name="Merged",
            line={
                "width": 4,
                "dash": "solid",
            },
            marker={
                "size": 9,
                "symbol": "circle",
            },
            hovertemplate=("<b>%{x}</b><br>Merged PRs: %{y}<extra></extra>"),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data["period"],
            y=chart_data["unmerged_prs"],
            mode="lines+markers",
            name="Unmerged",
            line={
                "width": 2,
                "dash": "dash",
            },
            marker={
                "size": 7,
                "symbol": "diamond-open",
            },
            hovertemplate=("<b>%{x}</b><br>Unmerged PRs: %{y}<extra></extra>"),
        )
    )

    figure.add_annotation(
        text=(
            "Merged and unmerged counts overlap because "
            "the dataset was time-matched within each quarter."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=1.12,
        showarrow=False,
        align="left",
    )

    return apply_chart_layout(
        figure,
        title="Quarterly PR Outcome Trend",
        xaxis_title="Quarter",
        yaxis_title="Number of PRs",
        height=520,
    )


def create_size_band_merge_rate_chart(
    size_band_metrics: pd.DataFrame,
) -> go.Figure:
    """Create merge-rate comparison by PR size."""

    required_columns = {
        "change_size_band",
        "total_prs",
        "merge_rate_percent",
    }

    missing_columns = required_columns - set(size_band_metrics.columns)

    if missing_columns:
        raise ValueError(
            f"Size-band metrics are missing columns: {sorted(missing_columns)}"
        )

    chart_data = size_band_metrics.copy()

    chart_data["change_size_band"] = pd.Categorical(
        chart_data["change_size_band"].astype("string"),
        categories=SIZE_BAND_ORDER,
        ordered=True,
    )

    chart_data = chart_data.sort_values("change_size_band")

    figure = px.bar(
        chart_data,
        x="change_size_band",
        y="merge_rate_percent",
        text="merge_rate_percent",
        custom_data=[
            "total_prs",
        ],
    )

    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Merge rate: %{y:.2f}%<br>"
            "PR count: %{customdata[0]}"
            "<extra></extra>"
        ),
    )

    figure.update_yaxes(
        range=[
            0,
            100,
        ]
    )

    return apply_chart_layout(
        figure,
        title="Merge Rate by Pull Request Size",
        xaxis_title="Change-size band",
        yaxis_title="Merge rate (%)",
    )


def create_lifecycle_merge_rate_chart(
    lifecycle_metrics: pd.DataFrame,
) -> go.Figure:
    """Create merge rates across lifecycle-duration bands."""

    required_columns = {
        "lifecycle_band",
        "total_prs",
        "merge_rate_percent",
    }

    missing_columns = required_columns - set(lifecycle_metrics.columns)

    if missing_columns:
        raise ValueError(
            f"Lifecycle metrics are missing columns: {sorted(missing_columns)}"
        )

    chart_data = lifecycle_metrics.copy()

    chart_data["lifecycle_band"] = pd.Categorical(
        chart_data["lifecycle_band"].astype("string"),
        categories=LIFECYCLE_BAND_ORDER,
        ordered=True,
    )

    chart_data = chart_data.sort_values("lifecycle_band")

    figure = px.bar(
        chart_data,
        x="lifecycle_band",
        y="merge_rate_percent",
        text="merge_rate_percent",
        custom_data=[
            "total_prs",
        ],
    )

    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Merge rate: %{y:.2f}%<br>"
            "PR count: %{customdata[0]}"
            "<extra></extra>"
        ),
    )

    figure.update_yaxes(
        range=[
            0,
            100,
        ]
    )

    return apply_chart_layout(
        figure,
        title="Merge Rate by PR Lifecycle Duration",
        xaxis_title="Lifecycle duration",
        yaxis_title="Merge rate (%)",
        height=520,
    )


def create_file_category_chart(
    file_category_metrics: pd.DataFrame,
) -> go.Figure:
    """Compare merge rates with and without file categories."""

    required_columns = {
        "file_category",
        "merge_rate_with_category",
        "merge_rate_without_category",
    }

    missing_columns = required_columns - set(file_category_metrics.columns)

    if missing_columns:
        raise ValueError(
            f"File-category metrics are missing columns: {sorted(missing_columns)}"
        )

    chart_data = file_category_metrics.melt(
        id_vars=[
            "file_category",
        ],
        value_vars=[
            "merge_rate_with_category",
            "merge_rate_without_category",
        ],
        var_name="comparison_group",
        value_name="merge_rate_percent",
    )

    chart_data["comparison_group"] = chart_data["comparison_group"].map(
        {
            "merge_rate_with_category": ("Category present"),
            "merge_rate_without_category": ("Category absent"),
        }
    )

    figure = px.bar(
        chart_data,
        x="file_category",
        y="merge_rate_percent",
        color="comparison_group",
        barmode="group",
        text="merge_rate_percent",
    )

    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=("<b>%{x}</b><br>%{fullData.name}: %{y:.2f}%<extra></extra>"),
    )

    figure.update_yaxes(
        range=[
            0,
            100,
        ]
    )

    return apply_chart_layout(
        figure,
        title="Merge Rate by Changed-File Category",
        xaxis_title="File category",
        yaxis_title="Merge rate (%)",
        height=540,
    )


def create_effect_size_chart(
    outcome_comparisons: pd.DataFrame,
    maximum_metrics: int = 10,
) -> go.Figure:
    """Create a ranked chart of merged-unmerged effect sizes."""

    required_columns = {
        "metric",
        "standardized_mean_difference",
        "effect_size_label",
    }

    missing_columns = required_columns - set(outcome_comparisons.columns)

    if missing_columns:
        raise ValueError(
            f"Outcome comparison data is missing columns: {sorted(missing_columns)}"
        )

    chart_data = outcome_comparisons.copy()

    chart_data["standardized_mean_difference"] = pd.to_numeric(
        chart_data["standardized_mean_difference"],
        errors="coerce",
    )

    chart_data = chart_data.dropna(
        subset=[
            "standardized_mean_difference",
        ]
    )

    chart_data["absolute_effect_size"] = chart_data[
        "standardized_mean_difference"
    ].abs()

    chart_data = (
        chart_data.sort_values(
            "absolute_effect_size",
            ascending=False,
        )
        .head(maximum_metrics)
        .sort_values("standardized_mean_difference")
    )

    figure = px.bar(
        chart_data,
        x="standardized_mean_difference",
        y="metric",
        orientation="h",
        text="standardized_mean_difference",
        custom_data=[
            "effect_size_label",
        ],
    )

    figure.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Effect size: %{x:.4f}<br>"
            "Magnitude: %{customdata[0]}"
            "<extra></extra>"
        ),
    )

    figure.add_vline(
        x=0,
        line_width=1,
        line_dash="dash",
    )

    return apply_chart_layout(
        figure,
        title=("Largest Differences Between Merged and Unmerged PRs"),
        xaxis_title=("Standardized mean difference"),
        yaxis_title="Metric",
        height=600,
    )


def create_log_complexity_boxplot(
    dataframe: pd.DataFrame,
    metric: str = "total_changes",
) -> go.Figure:
    """Create an outlier-aware complexity boxplot."""

    required_columns = {
        metric,
        "merge_outcome",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Complexity chart is missing columns: {sorted(missing_columns)}"
        )

    chart_data = dataframe[
        [
            metric,
            "merge_outcome",
        ]
    ].copy()

    chart_data[metric] = pd.to_numeric(
        chart_data[metric],
        errors="coerce",
    )

    chart_data = chart_data.dropna(
        subset=[
            metric,
            "merge_outcome",
        ]
    )

    chart_data = chart_data[chart_data[metric] >= 0]

    logarithmic_column = f"log1p_{metric}"

    chart_data[logarithmic_column] = np.log1p(chart_data[metric])

    figure = px.box(
        chart_data,
        x="merge_outcome",
        y=logarithmic_column,
        category_orders={
            "merge_outcome": (OUTCOME_ORDER),
        },
        points="outliers",
        custom_data=[
            metric,
        ],
    )

    figure.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"Original {metric}: "
            "%{customdata[0]:,.0f}<br>"
            "Log-transformed value: "
            "%{y:.3f}"
            "<extra></extra>"
        ),
    )

    return apply_chart_layout(
        figure,
        title=(f"Distribution of {metric.replace('_', ' ').title()} by Outcome"),
        xaxis_title="Outcome",
        yaxis_title=(f"log(1 + {metric.replace('_', ' ')})"),
        height=520,
    )


def build_all_eda_charts(
    dataframe: pd.DataFrame,
    quarterly_metrics: pd.DataFrame,
    size_band_metrics: pd.DataFrame,
    outcome_comparisons: pd.DataFrame,
    file_category_metrics: pd.DataFrame,
    lifecycle_metrics: pd.DataFrame,
) -> dict[str, go.Figure]:
    """Build every reusable Stage 4C chart."""

    return {
        "outcome_distribution": (create_outcome_distribution_chart(dataframe)),
        "quarterly_volume": (create_quarterly_volume_chart(quarterly_metrics)),
        "size_band_merge_rate": (create_size_band_merge_rate_chart(size_band_metrics)),
        "lifecycle_merge_rate": (create_lifecycle_merge_rate_chart(lifecycle_metrics)),
        "file_category_merge_rate": (create_file_category_chart(file_category_metrics)),
        "outcome_effect_sizes": (create_effect_size_chart(outcome_comparisons)),
        "total_changes_boxplot": (
            create_log_complexity_boxplot(
                dataframe,
                metric="total_changes",
            )
        ),
        "commit_count_boxplot": (
            create_log_complexity_boxplot(
                dataframe,
                metric="commit_count",
            )
        ),
    }


def describe_chart_collection(
    figures: dict[str, go.Figure],
) -> dict[str, Any]:
    """Return metadata about generated charts."""

    return {
        "chart_count": len(figures),
        "chart_names": sorted(figures),
        "all_figures_valid": all(
            isinstance(
                figure,
                go.Figure,
            )
            for figure in figures.values()
        ),
        "trace_counts": {name: len(figure.data) for name, figure in figures.items()},
    }
