"""Combined exploratory analysis and pull-request explorer."""

from __future__ import annotations

import json
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.config.settings import get_settings
from src.ui.streamlit_data import (
    apply_global_page_style,
    available_column,
    boolean_series,
    load_unified_intelligence,
    numeric_series,
    render_project_sidebar,
    safe_text,
    select_created_column,
    select_pr_number_column,
    select_repository_column,
    select_title_column,
)
from src.utils.logging import get_logger
from src.utils.paths import create_required_directories


settings = get_settings()
logger = get_logger(__name__)

create_required_directories()
apply_global_page_style()
render_project_sidebar(
    settings=settings,
    current_page="Data & PR Explorer",
)

logger.info("Data & PR Explorer page loaded.")

st.title("Data & PR Explorer")
st.caption(
    "Explore the pull-request dataset, investigate repository activity, "
    "filter records and inspect the evidence behind individual PRs."
)


def _find_first_available_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    direct_match = available_column(
        dataframe,
        candidates,
    )

    if direct_match is not None:
        return direct_match

    lowercase_lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        matched_column = lowercase_lookup.get(
            candidate.strip().lower()
        )

        if matched_column is not None:
            return matched_column

    return None


def _derive_merged_status(
    dataframe: pd.DataFrame,
) -> pd.Series | None:
    explicit_column = _find_first_available_column(
        dataframe,
        [
            "merged",
            "actual_merged",
            "merge_outcome",
            "is_merged",
            "was_merged",
            "merged_flag",
            "historical_merge_outcome",
            "actual_merge_outcome",
            "merge_label",
            "target",
            "label",
        ],
    )

    if explicit_column is not None:
        return boolean_series(
            dataframe[explicit_column]
        )

    merged_at_column = _find_first_available_column(
        dataframe,
        [
            "merged_at",
            "merge_timestamp",
            "merged_date",
            "merge_date",
        ],
    )

    if merged_at_column is not None:
        return pd.to_datetime(
            dataframe[merged_at_column],
            errors="coerce",
            utc=True,
        ).notna()

    merge_duration_column = _find_first_available_column(
        dataframe,
        [
            "merge_duration_hours",
            "time_to_merge_hours",
            "merge_time_hours",
            "hours_to_merge",
        ],
    )

    if merge_duration_column is not None:
        return pd.to_numeric(
            dataframe[merge_duration_column],
            errors="coerce",
        ).notna()

    return None


def _normalise_prediction_label(
    value: Any,
    positive_label: str,
    negative_label: str,
) -> str:
    if pd.isna(value):
        return "Not available"

    raw_value = str(value).strip().lower()

    positive_values = {
        "1",
        "true",
        "yes",
        "positive",
        "merged",
        "merge",
        "will merge",
        "predicted to merge",
        "delayed",
        "delay",
        "predicted delayed",
    }

    negative_values = {
        "0",
        "false",
        "no",
        "negative",
        "not merged",
        "not_merged",
        "not merge",
        "will not merge",
        "predicted not to merge",
        "not delayed",
        "no delay",
        "predicted not delayed",
    }

    numeric_value = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.notna(numeric_value):
        if float(numeric_value) == 1:
            return positive_label

        if float(numeric_value) == 0:
            return negative_label

    if raw_value in positive_values:
        return positive_label

    if raw_value in negative_values:
        return negative_label

    return str(value)


def _serialisable_record(
    row: pd.Series,
) -> dict[str, Any]:
    record: dict[str, Any] = {}

    for key, value in row.to_dict().items():
        if pd.isna(value):
            record[str(key)] = None
        elif isinstance(value, pd.Timestamp):
            record[str(key)] = value.isoformat()
        elif hasattr(value, "item"):
            try:
                record[str(key)] = value.item()
            except Exception:
                record[str(key)] = str(value)
        else:
            record[str(key)] = value

    return record


def _horizontal_count_chart(
    chart_data: pd.DataFrame,
    category_column: str,
    count_column: str,
    sort_order: list[str],
    height: int,
) -> alt.Chart:
    bars = (
        alt.Chart(chart_data)
        .mark_bar(
            cornerRadiusEnd=5,
            size=24,
        )
        .encode(
            x=alt.X(
                f"{count_column}:Q",
                title="Pull requests",
                axis=alt.Axis(
                    grid=True,
                    tickMinStep=1,
                ),
            ),
            y=alt.Y(
                f"{category_column}:N",
                title=None,
                sort=sort_order,
            ),
            tooltip=[
                alt.Tooltip(
                    f"{category_column}:N",
                    title=category_column,
                ),
                alt.Tooltip(
                    f"{count_column}:Q",
                    title="PR count",
                    format=",",
                ),
            ],
        )
    )

    labels = (
        alt.Chart(chart_data)
        .mark_text(
            align="left",
            baseline="middle",
            dx=6,
            fontWeight="bold",
        )
        .encode(
            x=alt.X(
                f"{count_column}:Q"
            ),
            y=alt.Y(
                f"{category_column}:N",
                sort=sort_order,
            ),
            text=alt.Text(
                f"{count_column}:Q",
                format=",",
            ),
        )
    )

    return (
        bars + labels
    ).properties(
        height=height,
    )


def _empty_chart_message(
    message: str,
) -> None:
    st.info(
        message
    )


dataframe = load_unified_intelligence()

if dataframe.empty:
    st.error(
        "The unified PR intelligence dataset could not be loaded."
    )
    st.stop()


pr_number_column = select_pr_number_column(
    dataframe
)

title_column = select_title_column(
    dataframe
)

repository_column = select_repository_column(
    dataframe
)

created_column = select_created_column(
    dataframe
)

author_column = _find_first_available_column(
    dataframe,
    [
        "author",
        "user_login",
        "creator",
        "created_by",
        "author_login",
    ],
)

status_column = _find_first_available_column(
    dataframe,
    [
        "state",
        "status",
        "pr_state",
        "pull_request_state",
    ],
)

merged_at_column = _find_first_available_column(
    dataframe,
    [
        "merged_at",
        "merge_timestamp",
        "merged_date",
        "merge_date",
    ],
)

merged_status = _derive_merged_status(
    dataframe
)

risk_column = _find_first_available_column(
    dataframe,
    [
        "policy_risk_band",
        "risk_band",
        "risk_level",
        "policy_risk_level",
    ],
)

priority_column = _find_first_available_column(
    dataframe,
    [
        "review_priority",
        "priority_band",
        "unified_review_priority",
    ],
)

manual_review_column = _find_first_available_column(
    dataframe,
    [
        "manual_review_required",
        "requires_manual_review",
    ],
)

merge_prediction_column = _find_first_available_column(
    dataframe,
    [
        "merge_prediction",
        "predicted_merge",
        "predicted_merge_outcome",
        "merge_outcome_prediction",
        "model1_prediction",
        "model_1_prediction",
        "predicted_merged",
    ],
)

delay_prediction_column = _find_first_available_column(
    dataframe,
    [
        "delay_prediction",
        "predicted_delay",
        "delayed_merge_prediction",
        "model2_prediction",
    ],
)

merge_probability_column = _find_first_available_column(
    dataframe,
    [
        "merge_probability",
        "model1_probability",
        "predicted_merge_probability",
        "merge_outcome_probability",
        "probability_merged",
        "positive_class_probability",
    ],
)

delay_probability_column = _find_first_available_column(
    dataframe,
    [
        "delay_probability",
        "model2_probability",
        "predicted_delay_probability",
    ],
)

merge_duration_column = _find_first_available_column(
    dataframe,
    [
        "merge_duration_hours",
        "time_to_merge_hours",
        "merge_time_hours",
        "hours_to_merge",
    ],
)

changed_files_column = _find_first_available_column(
    dataframe,
    [
        "changed_files",
        "changed_file_count",
        "files_changed",
    ],
)

additions_column = _find_first_available_column(
    dataframe,
    [
        "additions",
        "lines_added",
        "added_lines",
    ],
)

deletions_column = _find_first_available_column(
    dataframe,
    [
        "deletions",
        "lines_deleted",
        "deleted_lines",
    ],
)

total_changes_column = _find_first_available_column(
    dataframe,
    [
        "total_changes",
        "changed_lines",
        "total_changed_lines",
    ],
)

comments_column = _find_first_available_column(
    dataframe,
    [
        "comment_count",
        "comments",
        "total_comments",
        "review_comment_count",
    ],
)

description_word_count_column = _find_first_available_column(
    dataframe,
    [
        "description_word_count",
        "body_word_count",
        "description_length_words",
    ],
)

description_column = _find_first_available_column(
    dataframe,
    [
        "description",
        "body",
        "pr_description",
    ],
)


st.markdown("## 1. Dataset profile")

profile_columns = st.columns(5)

profile_columns[0].metric(
    "Rows",
    f"{len(dataframe):,}",
    help=(
        "The total number of pull-request records in the unified "
        "intelligence dataset."
    ),
)

profile_columns[1].metric(
    "Columns",
    f"{len(dataframe.columns):,}",
    help=(
        "The total number of raw, engineered, predictive and governance "
        "fields available in the dataset."
    ),
)

profile_columns[2].metric(
    "Duplicate PRs",
    (
        f"{int(dataframe[pr_number_column].duplicated().sum()):,}"
        if pr_number_column
        else "Not available"
    ),
    help=(
        "The number of repeated pull-request identifiers in the dataset."
    ),
)

profile_columns[3].metric(
    "Missing cells",
    f"{int(dataframe.isna().sum().sum()):,}",
    help=(
        "The total number of blank values across all rows and columns. "
        "Some missing values are expected because certain fields apply "
        "only to merged PRs or specific analytical stages."
    ),
)

profile_columns[4].metric(
    "Repositories",
    (
        f"{dataframe[repository_column].nunique(dropna=True):,}"
        if repository_column
        else "1"
    ),
    help=(
        "The number of repositories represented in the current dataset."
    ),
)


st.markdown("## 2. Historical and predictive patterns")

pattern_columns = st.columns(2)

with pattern_columns[0]:
    st.markdown("### a) Historical outcome balance")
    st.caption(
        "Shows the historical class balance used to understand Model 1's "
        "merge-outcome prediction problem."
    )

    if merged_status is not None:
        outcome_data = pd.DataFrame(
            {
                "Outcome": [
                    "Merged",
                    "Not merged",
                ],
                "PR count": [
                    int(
                        merged_status.sum()
                    ),
                    int(
                        (~merged_status).sum()
                    ),
                ],
            }
        )

        st.altair_chart(
            _horizontal_count_chart(
                chart_data=outcome_data,
                category_column="Outcome",
                count_column="PR count",
                sort_order=[
                    "Merged",
                    "Not merged",
                ],
                height=170,
            ),
            width="stretch",
        )
    else:
        _empty_chart_message(
            "Historical merge-outcome data is unavailable."
        )

with pattern_columns[1]:
    st.markdown("### b) Merge-probability distribution")
    st.caption(
        "Shows how Model 1's predicted merge probabilities are distributed "
        "across the pull-request population."
    )

    if merge_probability_column:
        probability_values = numeric_series(
            dataframe[merge_probability_column]
        ).dropna()

        if (
            not probability_values.empty
            and probability_values.max() > 1
        ):
            probability_values = (
                probability_values / 100.0
            )

        probability_values = probability_values.clip(
            lower=0,
            upper=1,
        )

        probability_labels = [
            "0–20%",
            "20–40%",
            "40–60%",
            "60–80%",
            "80–100%",
        ]

        probability_bands = pd.cut(
            probability_values,
            bins=[
                -0.001,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ],
            labels=probability_labels,
            include_lowest=True,
        )

        probability_distribution = (
            probability_bands.value_counts(
                sort=False
            )
            .rename_axis(
                "Merge probability"
            )
            .reset_index(
                name="PR count"
            )
        )

        probability_distribution[
            "Merge probability"
        ] = (
            probability_distribution[
                "Merge probability"
            ]
            .astype(str)
        )

        st.altair_chart(
            _horizontal_count_chart(
                chart_data=probability_distribution,
                category_column="Merge probability",
                count_column="PR count",
                sort_order=probability_labels,
                height=220,
            ),
            width="stretch",
        )
    else:
        _empty_chart_message(
            "Merge-probability data is unavailable."
        )


operational_columns = st.columns(2)

with operational_columns[0]:
    st.markdown("### c) Merge-time distribution")
    st.caption(
        "Shows how long successfully merged pull requests took to merge "
        "and provides operational context for Model 2."
    )

    if merge_duration_column:
        merge_time_values = numeric_series(
            dataframe[merge_duration_column]
        ).dropna()

        merge_time_labels = [
            "Within 24 hours",
            "1–2 days",
            "2–3 days",
            "3–7 days",
            "More than 7 days",
        ]

        merge_time_bands = pd.cut(
            merge_time_values,
            bins=[
                -0.001,
                24,
                48,
                72,
                168,
                float("inf"),
            ],
            labels=merge_time_labels,
            include_lowest=True,
        )

        merge_time_distribution = (
            merge_time_bands.value_counts(
                sort=False
            )
            .rename_axis(
                "Merge time"
            )
            .reset_index(
                name="PR count"
            )
        )

        merge_time_distribution[
            "Merge time"
        ] = (
            merge_time_distribution[
                "Merge time"
            ]
            .astype(str)
        )

        st.altair_chart(
            _horizontal_count_chart(
                chart_data=merge_time_distribution,
                category_column="Merge time",
                count_column="PR count",
                sort_order=merge_time_labels,
                height=220,
            ),
            width="stretch",
        )
    elif merged_at_column and created_column:
        created_dates = pd.to_datetime(
            dataframe[created_column],
            errors="coerce",
            utc=True,
        )

        merged_dates = pd.to_datetime(
            dataframe[merged_at_column],
            errors="coerce",
            utc=True,
        )

        derived_merge_hours = (
            merged_dates - created_dates
        ).dt.total_seconds() / 3600

        derived_merge_hours = derived_merge_hours[
            derived_merge_hours.ge(0)
        ].dropna()

        if not derived_merge_hours.empty:
            merge_time_labels = [
                "Within 24 hours",
                "1–2 days",
                "2–3 days",
                "3–7 days",
                "More than 7 days",
            ]

            merge_time_bands = pd.cut(
                derived_merge_hours,
                bins=[
                    -0.001,
                    24,
                    48,
                    72,
                    168,
                    float("inf"),
                ],
                labels=merge_time_labels,
                include_lowest=True,
            )

            merge_time_distribution = (
                merge_time_bands.value_counts(
                    sort=False
                )
                .rename_axis(
                    "Merge time"
                )
                .reset_index(
                    name="PR count"
                )
            )

            merge_time_distribution[
                "Merge time"
            ] = (
                merge_time_distribution[
                    "Merge time"
                ]
                .astype(str)
            )

            st.altair_chart(
                _horizontal_count_chart(
                    chart_data=merge_time_distribution,
                    category_column="Merge time",
                    count_column="PR count",
                    sort_order=merge_time_labels,
                    height=220,
                ),
                width="stretch",
            )
        else:
            _empty_chart_message(
                "Merge-time data is unavailable."
            )
    else:
        _empty_chart_message(
            "Merge-time data is unavailable."
        )

with operational_columns[1]:
    st.markdown("### d) PR-size distribution")
    st.caption(
        "Groups pull requests by total code changes to show the overall "
        "complexity profile of the dataset."
    )

    if total_changes_column:
        size_values = numeric_series(
            dataframe[total_changes_column]
        ).fillna(0)

        size_labels = [
            "Very small",
            "Small",
            "Medium",
            "Large",
            "Very large",
        ]

        size_band = pd.cut(
            size_values,
            bins=[
                -1,
                10,
                100,
                500,
                2000,
                float("inf"),
            ],
            labels=size_labels,
        )

        size_distribution = (
            size_band.value_counts(
                sort=False
            )
            .rename_axis(
                "PR size"
            )
            .reset_index(
                name="PR count"
            )
        )

        size_distribution[
            "PR size"
        ] = (
            size_distribution[
                "PR size"
            ]
            .astype(str)
        )

        st.altair_chart(
            _horizontal_count_chart(
                chart_data=size_distribution,
                category_column="PR size",
                count_column="PR count",
                sort_order=size_labels,
                height=220,
            ),
            width="stretch",
        )
    else:
        _empty_chart_message(
            "PR-size data is unavailable."
        )


st.markdown("## 3. Repository activity over time")

if created_column:
    created_dates = pd.to_datetime(
        dataframe[created_column],
        errors="coerce",
        utc=True,
    )

    activity_data = pd.DataFrame(
        {
            "Created date": created_dates.dt.tz_convert(
                None
            ),
        }
    ).dropna()

    if not activity_data.empty:
        activity_data[
            "Month"
        ] = activity_data[
            "Created date"
        ].dt.to_period(
            "M"
        ).dt.to_timestamp()

        monthly_volume = (
            activity_data.groupby(
                "Month",
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size": "PR count",
                }
            )
        )

        activity_chart = (
            alt.Chart(
                monthly_volume
            )
            .mark_line(
                point=True,
            )
            .encode(
                x=alt.X(
                    "Month:T",
                    title="Month",
                ),
                y=alt.Y(
                    "PR count:Q",
                    title="Pull requests",
                    axis=alt.Axis(
                        tickMinStep=1,
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Month:T",
                        title="Month",
                        format="%b %Y",
                    ),
                    alt.Tooltip(
                        "PR count:Q",
                        title="PR count",
                        format=",",
                    ),
                ],
            )
            .properties(
                height=320,
            )
        )

        st.altair_chart(
            activity_chart,
            width="stretch",
        )
    else:
        st.info(
            "PR-volume-over-time data is unavailable."
        )
else:
    st.info(
        "PR-volume-over-time data is unavailable."
    )


st.markdown("## 4. Filter and search pull requests")

filter_row_one = st.columns(4)

search_text = filter_row_one[0].text_input(
    "Search PR number or title",
    placeholder="Example: 5336 or send_file",
)

author_options = [
    "All"
]

if author_column:
    author_options += sorted(
        dataframe[author_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

selected_author = filter_row_one[1].selectbox(
    "Author",
    options=author_options,
)

status_options = [
    "All"
]

if status_column:
    status_options += sorted(
        dataframe[status_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

selected_status = filter_row_one[2].selectbox(
    "Status",
    options=status_options,
)

risk_options = [
    "All"
]

if risk_column:
    risk_options += sorted(
        dataframe[risk_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

selected_risk = filter_row_one[3].selectbox(
    "Policy risk",
    options=risk_options,
)


filter_row_two = st.columns(4)

priority_options = [
    "All"
]

if priority_column:
    priority_options += sorted(
        dataframe[priority_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

selected_priority = filter_row_two[0].selectbox(
    "Review priority",
    options=priority_options,
)

merge_prediction_options = [
    "All",
    "Predicted to merge",
    "Predicted not to merge",
]

selected_merge_prediction = filter_row_two[1].selectbox(
    "Model 1 prediction",
    options=merge_prediction_options,
)

delay_prediction_options = [
    "All",
    "Predicted delayed",
    "Predicted not delayed",
]

selected_delay_prediction = filter_row_two[2].selectbox(
    "Model 2 prediction",
    options=delay_prediction_options,
)

selected_review = filter_row_two[3].selectbox(
    "Manual review",
    options=[
        "All",
        "Required",
        "Not required",
    ],
)


working_dataframe = dataframe.copy()

if merged_status is not None:
    working_dataframe[
        "_historical_outcome"
    ] = merged_status.map(
        {
            True: "Merged",
            False: "Not merged",
        }
    )

if merge_prediction_column:
    working_dataframe[
        "_merge_prediction_label"
    ] = working_dataframe[
        merge_prediction_column
    ].apply(
        lambda value: _normalise_prediction_label(
            value,
            "Predicted to merge",
            "Predicted not to merge",
        )
    )
elif merge_probability_column:
    merge_probability_values = numeric_series(
        working_dataframe[
            merge_probability_column
        ]
    )

    if (
        not merge_probability_values.dropna().empty
        and merge_probability_values.dropna().max() > 1
    ):
        merge_probability_values = (
            merge_probability_values / 100.0
        )

    working_dataframe[
        "_merge_prediction_label"
    ] = pd.Series(
        "Not available",
        index=working_dataframe.index,
        dtype="object",
    )

    working_dataframe.loc[
        merge_probability_values.ge(0.5),
        "_merge_prediction_label",
    ] = "Predicted to merge"

    working_dataframe.loc[
        merge_probability_values.lt(0.5)
        & merge_probability_values.notna(),
        "_merge_prediction_label",
    ] = "Predicted not to merge"

if delay_prediction_column:
    working_dataframe[
        "_delay_prediction_label"
    ] = working_dataframe[
        delay_prediction_column
    ].apply(
        lambda value: _normalise_prediction_label(
            value,
            "Predicted delayed",
            "Predicted not delayed",
        )
    )
elif delay_probability_column:
    delay_probability_values = numeric_series(
        working_dataframe[
            delay_probability_column
        ]
    )

    if (
        not delay_probability_values.dropna().empty
        and delay_probability_values.dropna().max() > 1
    ):
        delay_probability_values = (
            delay_probability_values / 100.0
        )

    working_dataframe[
        "_delay_prediction_label"
    ] = pd.Series(
        "Not available",
        index=working_dataframe.index,
        dtype="object",
    )

    working_dataframe.loc[
        delay_probability_values.ge(0.5),
        "_delay_prediction_label",
    ] = "Predicted delayed"

    working_dataframe.loc[
        delay_probability_values.lt(0.5)
        & delay_probability_values.notna(),
        "_delay_prediction_label",
    ] = "Predicted not delayed"


filtered = working_dataframe.copy()

if search_text.strip():
    search_mask = pd.Series(
        False,
        index=filtered.index,
    )

    if pr_number_column:
        search_mask = search_mask | (
            filtered[pr_number_column]
            .astype(str)
            .str.contains(
                search_text.strip(),
                case=False,
                na=False,
            )
        )

    if title_column:
        search_mask = search_mask | (
            filtered[title_column]
            .astype(str)
            .str.contains(
                search_text.strip(),
                case=False,
                na=False,
            )
        )

    filtered = filtered[
        search_mask
    ]

if author_column and selected_author != "All":
    filtered = filtered[
        filtered[author_column]
        .astype(str)
        .eq(
            selected_author
        )
    ]

if status_column and selected_status != "All":
    filtered = filtered[
        filtered[status_column]
        .astype(str)
        .eq(
            selected_status
        )
    ]

if risk_column and selected_risk != "All":
    filtered = filtered[
        filtered[risk_column]
        .astype(str)
        .eq(
            selected_risk
        )
    ]

if priority_column and selected_priority != "All":
    filtered = filtered[
        filtered[priority_column]
        .astype(str)
        .eq(
            selected_priority
        )
    ]

if (
    selected_merge_prediction != "All"
    and "_merge_prediction_label" in filtered.columns
):
    filtered = filtered[
        filtered["_merge_prediction_label"]
        .eq(
            selected_merge_prediction
        )
    ]

if (
    selected_delay_prediction != "All"
    and "_delay_prediction_label" in filtered.columns
):
    filtered = filtered[
        filtered["_delay_prediction_label"]
        .eq(
            selected_delay_prediction
        )
    ]

if manual_review_column and selected_review != "All":
    required = (
        selected_review
        == "Required"
    )

    filtered = filtered[
        boolean_series(
            filtered[manual_review_column]
        ).eq(
            required
        )
    ]


st.markdown("## 5. Searchable PR table")

display_column_mapping: list[tuple[str | None, str]] = [
    (
        pr_number_column,
        "PR number",
    ),
    (
        title_column,
        "Title",
    ),
    (
        author_column,
        "Author",
    ),
    (
        status_column,
        "Status",
    ),
    (
        repository_column,
        "Repository",
    ),
    (
        created_column,
        "Created at",
    ),
    (
        "_historical_outcome"
        if "_historical_outcome" in filtered.columns
        else None,
        "Historical outcome",
    ),
    (
        "_merge_prediction_label"
        if "_merge_prediction_label" in filtered.columns
        else None,
        "Model 1 prediction",
    ),
    (
        merge_probability_column,
        "Merge probability",
    ),
    (
        "_delay_prediction_label"
        if "_delay_prediction_label" in filtered.columns
        else None,
        "Model 2 prediction",
    ),
    (
        delay_probability_column,
        "Delay probability",
    ),
    (
        risk_column,
        "Policy risk",
    ),
    (
        priority_column,
        "Review priority",
    ),
    (
        manual_review_column,
        "Manual review",
    ),
]

display_columns = [
    column
    for column, _ in display_column_mapping
    if column is not None
    and column in filtered.columns
]

rename_mapping = {
    column: display_name
    for column, display_name in display_column_mapping
    if column is not None
    and column in filtered.columns
}

display_frame = filtered[
    display_columns
].rename(
    columns=rename_mapping
).copy()

if "Merge probability" in display_frame.columns:
    display_frame[
        "Merge probability"
    ] = numeric_series(
        display_frame[
            "Merge probability"
        ]
    )

if "Delay probability" in display_frame.columns:
    display_frame[
        "Delay probability"
    ] = numeric_series(
        display_frame[
            "Delay probability"
        ]
    )

st.caption(
    f"Showing {len(filtered):,} of {len(dataframe):,} pull requests."
)

st.dataframe(
    display_frame,
    width="stretch",
    hide_index=True,
    height=500,
)


st.markdown("## 6. Inspect one pull request")

if not pr_number_column:
    st.warning(
        "A PR-number column was not found, so record inspection is unavailable."
    )
    st.stop()

available_pr_numbers = (
    filtered[pr_number_column]
    .dropna()
    .drop_duplicates()
    .tolist()
)

if not available_pr_numbers:
    st.info(
        "No pull requests match the current filters."
    )
    st.stop()

selected_pr_number = st.selectbox(
    "Select PR number",
    options=available_pr_numbers,
)

selected_matches = filtered[
    filtered[pr_number_column]
    == selected_pr_number
]

if selected_matches.empty:
    st.info(
        "The selected pull request could not be resolved."
    )
    st.stop()

selected_index = selected_matches.index[0]
selected_row = filtered.loc[
    selected_index
]


identity_columns = st.columns(4)

identity_columns[0].metric(
    "PR number",
    safe_text(
        selected_row.get(
            pr_number_column
        )
    ),
)

identity_columns[1].metric(
    "Repository",
    safe_text(
        selected_row.get(
            repository_column
        )
        if repository_column
        else None
    ),
)

selected_outcome = (
    merged_status.loc[
        selected_index
    ]
    if merged_status is not None
    and selected_index in merged_status.index
    else pd.NA
)

identity_columns[2].metric(
    "Historical outcome",
    (
        "Merged"
        if selected_outcome is True
        else (
            "Not merged"
            if selected_outcome is False
            else "Not available"
        )
    ),
)

selected_manual_review = (
    boolean_series(
        pd.Series(
            [
                selected_row.get(
                    manual_review_column
                )
            ]
        )
    ).iloc[0]
    if manual_review_column
    else pd.NA
)

identity_columns[3].metric(
    "Manual review",
    (
        "Required"
        if selected_manual_review is True
        else (
            "Not required"
            if selected_manual_review is False
            else "Not available"
        )
    ),
)


st.markdown(
    f"### {safe_text(selected_row.get(title_column) if title_column else None)}"
)

selected_detail_columns = st.columns(3)

with selected_detail_columns[0]:
    st.markdown("#### Identity and activity")

    identity_fields = [
        author_column,
        status_column,
        created_column,
        merged_at_column,
        "target_branch"
        if "target_branch" in selected_row.index
        else None,
        "base_branch"
        if "base_branch" in selected_row.index
        else None,
    ]

    identity_fields = [
        field
        for field in identity_fields
        if field is not None
        and field in selected_row.index
    ]

    identity_table = pd.DataFrame(
        {
            "Field": identity_fields,
            "Value": [
                safe_text(
                    selected_row.get(
                        field
                    )
                )
                for field in identity_fields
            ],
        }
    )

    st.dataframe(
        identity_table,
        width="stretch",
        hide_index=True,
    )

with selected_detail_columns[1]:
    st.markdown("#### Predictive intelligence")

    predictive_fields = [
        merge_prediction_column,
        merge_probability_column,
        delay_prediction_column,
        delay_probability_column,
        merge_duration_column,
    ]

    predictive_fields = [
        field
        for field in predictive_fields
        if field is not None
        and field in selected_row.index
    ]

    predictive_table = pd.DataFrame(
        {
            "Field": predictive_fields,
            "Value": [
                safe_text(
                    selected_row.get(
                        field
                    )
                )
                for field in predictive_fields
            ],
        }
    )

    st.dataframe(
        predictive_table,
        width="stretch",
        hide_index=True,
    )

with selected_detail_columns[2]:
    st.markdown("#### Governance intelligence")

    governance_fields = [
        risk_column,
        priority_column,
        manual_review_column,
        "triggered_rules"
        if "triggered_rules" in selected_row.index
        else None,
        "recommended_action"
        if "recommended_action" in selected_row.index
        else None,
    ]

    governance_fields = [
        field
        for field in governance_fields
        if field is not None
        and field in selected_row.index
    ]

    governance_table = pd.DataFrame(
        {
            "Field": governance_fields,
            "Value": [
                safe_text(
                    selected_row.get(
                        field
                    )
                )
                for field in governance_fields
            ],
        }
    )

    st.dataframe(
        governance_table,
        width="stretch",
        hide_index=True,
    )


st.markdown("#### Complexity and engagement")

complexity_fields = [
    changed_files_column,
    additions_column,
    deletions_column,
    total_changes_column,
    "commit_count"
    if "commit_count" in selected_row.index
    else None,
    comments_column,
    description_word_count_column,
]

complexity_fields = [
    field
    for field in complexity_fields
    if field is not None
    and field in selected_row.index
]

complexity_table = pd.DataFrame(
    {
        "Field": complexity_fields,
        "Value": [
            safe_text(
                selected_row.get(
                    field
                )
            )
            for field in complexity_fields
        ],
    }
)

st.dataframe(
    complexity_table,
    width="stretch",
    hide_index=True,
)


if description_column:
    with st.expander(
        "PR description",
        expanded=False,
    ):
        st.write(
            safe_text(
                selected_row.get(
                    description_column
                )
            )
        )


with st.expander(
    "View complete selected-record JSON",
    expanded=False,
):
    st.json(
        _serialisable_record(
            selected_row
        )
    )