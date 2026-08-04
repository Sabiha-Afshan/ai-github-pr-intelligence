"""Combined exploratory analysis and pull-request explorer."""

from __future__ import annotations

import ast
import html
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

MERGE_DECISION_THRESHOLD = 0.425
DELAY_DECISION_THRESHOLD = 0.75

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



def _reset_filters() -> None:
    """Reset every search and filter control to its default value."""

    st.session_state["pr_search_text"] = ""
    st.session_state["author_filter"] = "All"
    st.session_state["risk_filter"] = "All"
    st.session_state["priority_filter"] = "All"
    st.session_state["merge_prediction_filter"] = "All"
    st.session_state["delay_prediction_filter"] = "All"
    st.session_state["manual_review_filter"] = "All"




def _field_candidates_present(
    row: pd.Series,
    candidates: list[str],
) -> list[str]:
    """Return candidate fields that exist and contain a usable value."""

    present_fields: list[str] = []

    for field in candidates:
        if field not in row.index:
            continue

        value = row.get(
            field
        )

        if pd.isna(value):
            continue

        if isinstance(
            value,
            str,
        ) and not value.strip():
            continue

        present_fields.append(
            field
        )

    return present_fields


def _render_field_table(
    row: pd.Series,
    fields: list[str],
    empty_message: str,
) -> None:
    """Render a two-column field/value table or a clear empty-state message."""

    usable_fields = _field_candidates_present(
        row,
        fields,
    )

    if not usable_fields:
        st.info(
            empty_message
        )
        return

    table = pd.DataFrame(
        {
            "Field": usable_fields,
            "Value": [
                safe_text(
                    row.get(
                        field
                    )
                )
                for field in usable_fields
            ],
        }
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )


def _normalise_list_like(
    value: Any,
) -> list[str]:
    """Convert lists or list-like strings into a clean list of text values."""

    if value is None:
        return []

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(
        value,
        str,
    ):
        stripped = value.strip()

        if not stripped:
            return []

        try:
            parsed = ast.literal_eval(
                stripped
            )

            if isinstance(
                parsed,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]
        except (
            ValueError,
            SyntaxError,
        ):
            pass

        separators = [
            "|",
            ",",
            ";",
        ]

        for separator in separators:
            if separator in stripped:
                return [
                    item.strip()
                    for item in stripped.split(
                        separator
                    )
                    if item.strip()
                ]

        return [
            stripped
        ]

    return [
        str(value).strip()
    ]


def _split_recommendations(
    value: Any,
) -> list[str]:
    """Split recommendation text into readable individual items."""

    recommendations = _normalise_list_like(
        value
    )

    if len(recommendations) > 1:
        return recommendations

    if not recommendations:
        return []

    text = recommendations[0]

    sentence_parts = [
        part.strip()
        for part in text.split(
            ". "
        )
        if part.strip()
    ]

    return [
        (
            part
            if part.endswith(
                "."
            )
            else f"{part}."
        )
        for part in sentence_parts
    ]


def _build_rule_tooltips(
    row: pd.Series,
    rule_codes: list[str],
) -> dict[str, str]:
    """
    Return an exact tooltip for each deterministic policy rule.

    The rule registry is used instead of matching rules to a shared list of
    recommendations. This prevents a recommendation from being displayed
    against the wrong rule when only a subset of rules is triggered.
    """

    rule_registry: dict[str, str] = {
        "PR003": (
            "Testing: Automated test changes were not detected. "
            "Add or update automated tests, or document why additional "
            "tests are not required."
        ),
        "PR004": (
            "Documentation: Required documentation changes were not detected. "
            "Update user, developer or operational documentation where the "
            "change affects expected behaviour."
        ),
        "PR005": (
            "Security: Security-sensitive changes were detected. "
            "Require focused security review and verify that secrets, "
            "authentication and permission impacts have been assessed."
        ),
        "PR006": (
            "Security validation: Security-sensitive changes do not have "
            "sufficient test or validation evidence. Block approval until "
            "security-relevant tests or documented validation evidence are supplied."
        ),
        "PR007": (
            "Operations: Configuration or operational changes were detected. "
            "Confirm environment impact, deployment sequencing, rollback "
            "instructions and configuration validation."
        ),
        "PR011": (
            "Governance: No appropriate reviewer was detected. "
            "Assign an appropriate reviewer before the PR progresses toward approval."
        ),
        "PR012": (
            "Complexity: Several PR characteristics are unusual for this repository. "
            "Perform additional manual review before approval."
        ),
    }

    description_fields = [
        "triggered_rule_descriptions",
        "triggered_rule_details",
        "rule_descriptions",
        "policy_rule_descriptions",
        "rule_explanations",
    ]

    record_descriptions: dict[str, str] = {}

    for field in description_fields:
        if field not in row.index:
            continue

        value = row.get(field)

        if isinstance(value, dict):
            record_descriptions = {
                str(code): str(description)
                for code, description in value.items()
            }
            break

    tooltips: dict[str, str] = {}

    for code in rule_codes:
        if code in record_descriptions:
            tooltips[code] = record_descriptions[code]
        elif code in rule_registry:
            tooltips[code] = rule_registry[code]
        else:
            tooltips[code] = (
                f"{code}: An exact rule definition is not available in "
                "the current dashboard rule registry."
            )

    return tooltips



def _format_hours_duration(
    value: Any,
) -> str:
    """Format an hour value as hours or days plus hours."""

    numeric_value = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(numeric_value):
        return "Not available"

    hours = float(numeric_value)

    if hours < 24:
        return f"{hours:.1f} hours"

    days = hours / 24.0

    return (
        f"{days:.1f} days "
        f"({hours:,.1f} hours)"
    )


def _render_named_value_table(
    rows: list[tuple[str, Any]],
    empty_message: str,
    omit_zero_values: bool = False,
) -> None:
    """Render friendly field labels and values in a two-column table."""

    display_rows: list[dict[str, str]] = []

    for label, value in rows:
        if value is None or pd.isna(value):
            continue

        if isinstance(value, str) and not value.strip():
            continue

        numeric_value = pd.to_numeric(
            pd.Series([value]),
            errors="coerce",
        ).iloc[0]

        if (
            omit_zero_values
            and pd.notna(numeric_value)
            and float(numeric_value) == 0
        ):
            continue

        if pd.notna(numeric_value):
            formatted_value = (
                f"{int(numeric_value):,}"
                if float(numeric_value).is_integer()
                else f"{float(numeric_value):,.2f}"
            )
        else:
            formatted_value = safe_text(value)

        display_rows.append(
            {
                "Field": label,
                "Value": formatted_value,
            }
        )

    if not display_rows:
        st.info(empty_message)
        return

    st.dataframe(
        pd.DataFrame(display_rows),
        width="stretch",
        hide_index=True,
    )

def _render_governance_table(
    row: pd.Series,
    risk_field: str | None,
    priority_field: str | None,
    manual_review_field: str | None,
) -> None:
    """
    Render Governance intelligence as a bordered two-column table matching
    the visual structure of the Streamlit dataframes used beside it.

    Each triggered rule keeps an individual native browser tooltip.
    """

    table_rows: list[tuple[str, str]] = []

    if (
        risk_field is not None
        and risk_field in row.index
        and pd.notna(row.get(risk_field))
    ):
        table_rows.append(
            (
                "policy_risk_band",
                html.escape(safe_text(row.get(risk_field))),
            )
        )

    if (
        priority_field is not None
        and priority_field in row.index
        and pd.notna(row.get(priority_field))
    ):
        table_rows.append(
            (
                "review_priority",
                html.escape(safe_text(row.get(priority_field))),
            )
        )

    if (
        manual_review_field is not None
        and manual_review_field in row.index
        and pd.notna(row.get(manual_review_field))
    ):
        manual_value = boolean_series(
            pd.Series([row.get(manual_review_field)])
        ).iloc[0]

        if pd.isna(manual_value):
            manual_display = "Not available"
        else:
            manual_display = "True" if bool(manual_value) else "False"

        table_rows.append(
            (
                "manual_review_required",
                manual_display,
            )
        )

    rule_codes = _normalise_list_like(
        row.get("triggered_rules")
        if "triggered_rules" in row.index
        else None
    )

    if rule_codes:
        rule_tooltips = _build_rule_tooltips(
            row=row,
            rule_codes=rule_codes,
        )

        rules_html = " | ".join(
            (
                '<span class="governance-rule" title="'
                + html.escape(
                    rule_tooltips.get(
                        rule_code,
                        "Rule details are unavailable.",
                    ),
                    quote=True,
                )
                + '">'
                + html.escape(rule_code)
                + "</span>"
            )
            for rule_code in rule_codes
        )

        table_rows.append(
            (
                "triggered_rules",
                rules_html,
            )
        )

    if not table_rows:
        st.info(
            "Governance information is unavailable for this pull request."
        )
        return

    body_html = "".join(
        (
            "<tr>"
            f"<td>{html.escape(field_name)}</td>"
            f"<td>{field_value}</td>"
            "</tr>"
        )
        for field_name, field_value in table_rows
    )

    st.markdown(
        """
        <style>
        .governance-table-wrapper {
            width: 100%;
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 8px;
            overflow: hidden;
            background: var(--background-color);
            box-sizing: border-box;
            font-family: "Source Sans Pro", sans-serif;
            font-size: 14px;
            line-height: 1.25;
        }

        .governance-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            margin: 0;
            font-family: inherit;
            font-size: inherit;
            line-height: inherit;
        }

        .governance-table th,
        .governance-table td {
            height: 36px;
            padding: 8px 10px;
            text-align: left;
            vertical-align: middle;
            border-bottom: 1px solid rgba(49, 51, 63, 0.12);
            box-sizing: border-box;
            overflow-wrap: anywhere;
            font-family: inherit;
            font-size: 14px;
            font-weight: 400;
            color: inherit;
        }

        .governance-table th {
            background: rgba(128, 128, 128, 0.045);
        }

        .governance-table th:first-child,
        .governance-table td:first-child {
            width: 48%;
            border-right: 1px solid rgba(49, 51, 63, 0.12);
        }

        .governance-table tbody tr:last-child td {
            border-bottom: none;
        }

        .governance-rule {
            cursor: help;
            text-decoration-line: underline;
            text-decoration-style: dotted;
            text-underline-offset: 3px;
            white-space: nowrap;
            font-family: inherit;
            font-size: 14px;
            font-weight: 400;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="governance-table-wrapper">'
            '<table class="governance-table">'
            "<thead>"
            "<tr><th>Field</th><th>Value</th></tr>"
            "</thead>"
            "<tbody>"
            + body_html
            + "</tbody>"
            "</table>"
            "</div>"
        ),
        unsafe_allow_html=True,
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
        "merge_hours",
        "merge_duration_hours",
        "time_to_merge_hours",
        "merge_time_hours",
        "hours_to_merge",
    ],
)

resolution_hours_column = _find_first_available_column(
    dataframe,
    [
        "resolution_hours",
        "time_to_resolution_hours",
        "resolution_duration_hours",
        "hours_to_resolution",
    ],
)

test_files_changed_column = _find_first_available_column(
    dataframe,
    [
        "test_files_changed",
        "test_file_count",
        "tests_files_changed",
    ],
)

documentation_files_changed_column = _find_first_available_column(
    dataframe,
    [
        "documentation_files_changed",
        "docs_files_changed",
        "documentation_file_count",
    ],
)

configuration_files_changed_column = _find_first_available_column(
    dataframe,
    [
        "configuration_files_changed",
        "config_files_changed",
        "configuration_file_count",
    ],
)

security_sensitive_files_changed_column = _find_first_available_column(
    dataframe,
    [
        "security_sensitive_files_changed",
        "security_files_changed",
        "security_sensitive_file_count",
    ],
)

files_added_column = _find_first_available_column(
    dataframe,
    [
        "files_added",
        "added_files_count",
        "new_files_count",
    ],
)

files_modified_column = _find_first_available_column(
    dataframe,
    [
        "files_modified",
        "modified_files_count",
    ],
)

files_removed_column = _find_first_available_column(
    dataframe,
    [
        "files_removed",
        "removed_files_count",
        "deleted_files_count",
    ],
)

files_renamed_column = _find_first_available_column(
    dataframe,
    [
        "files_renamed",
        "renamed_files_count",
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
        "issue_comment_count",
        "issue_comments",
        "review_comment_count",
        "review_comments",
        "comments_count",
    ],
)

requested_reviewer_count_column = _find_first_available_column(
    dataframe,
    [
        "requested_reviewer_count",
        "reviewer_count",
        "requested_reviewers_count",
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


st.markdown("## 2. Pull-request complexity and merge patterns")

distribution_columns = st.columns(2)

with distribution_columns[0]:
    st.markdown("### a) PR-size distribution")
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

with distribution_columns[1]:
    st.markdown("### b) Merge-time distribution")
    st.caption(
        "Shows how long successfully merged pull requests took to merge "
        "and highlights unusually delayed cases."
    )

    merge_time_values = pd.Series(
        dtype="float64"
    )

    if merge_duration_column:
        merge_time_values = numeric_series(
            dataframe[merge_duration_column]
        ).dropna()

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

        merge_time_values = (
            merged_dates - created_dates
        ).dt.total_seconds() / 3600

        merge_time_values = merge_time_values[
            merge_time_values.ge(0)
        ].dropna()

    if not merge_time_values.empty:
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
    else:
        _empty_chart_message(
            "Merge-time data is unavailable."
        )


relationship_columns = st.columns(2)

with relationship_columns[0]:
    st.markdown("### c) Merge rate and sample size by PR size")

    st.caption(
        "Shows the percentage of pull requests merged within each size "
        "category. The tooltip also shows the number of PRs behind each "
        "rate so small groups can be interpreted cautiously."
    )

    if (
        total_changes_column
        and merged_status is not None
    ):
        size_outcome_data = pd.DataFrame(
            {
                "Total changes": numeric_series(
                    dataframe[total_changes_column]
                ),
                "Merged": merged_status,
            }
        ).dropna(
            subset=[
                "Total changes",
                "Merged",
            ]
        )

        size_labels = [
            "Very small",
            "Small",
            "Medium",
            "Large",
            "Very large",
        ]

        size_outcome_data[
            "PR size"
        ] = pd.cut(
            size_outcome_data[
                "Total changes"
            ],
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

        merge_rate_by_size = (
            size_outcome_data.groupby(
                "PR size",
                observed=False,
            )
            .agg(
                PR_count=(
                    "Merged",
                    "size",
                ),
                Merged_PRs=(
                    "Merged",
                    "sum",
                ),
                Merge_rate=(
                    "Merged",
                    "mean",
                ),
            )
            .reset_index()
        )

        merge_rate_by_size[
            "PR size"
        ] = (
            merge_rate_by_size[
                "PR size"
            ]
            .astype(str)
        )

        merge_rate_by_size[
            "Merge rate (%)"
        ] = (
            merge_rate_by_size[
                "Merge_rate"
            ]
            * 100
        )

        merge_rate_by_size[
            "Rate label"
        ] = merge_rate_by_size.apply(
            lambda row: (
                f"{row['Merge rate (%)']:.1f}% "
                f"({int(row['Merged_PRs'])}/{int(row['PR_count'])})"
            ),
            axis=1,
        )

        merge_rate_bars = (
            alt.Chart(
                merge_rate_by_size
            )
            .mark_bar(
                cornerRadiusEnd=5,
                size=42,
            )
            .encode(
                x=alt.X(
                    "PR size:N",
                    title="PR size",
                    sort=size_labels,
                ),
                y=alt.Y(
                    "Merge rate (%):Q",
                    title="Merge rate (%)",
                    scale=alt.Scale(
                        domain=[
                            0,
                            100,
                        ],
                    ),
                    axis=alt.Axis(
                        grid=True,
                        format=".0f",
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "PR size:N",
                        title="PR size",
                    ),
                    alt.Tooltip(
                        "PR_count:Q",
                        title="PR count",
                        format=",",
                    ),
                    alt.Tooltip(
                        "Merged_PRs:Q",
                        title="Merged PRs",
                        format=",",
                    ),
                    alt.Tooltip(
                        "Merge rate (%):Q",
                        title="Merge rate",
                        format=".1f",
                    ),
                ],
            )
        )

        merge_rate_labels = (
            alt.Chart(
                merge_rate_by_size
            )
            .mark_text(
                dy=-10,
                fontWeight="bold",
            )
            .encode(
                x=alt.X(
                    "PR size:N",
                    sort=size_labels,
                ),
                y=alt.Y(
                    "Merge rate (%):Q",
                ),
                text=alt.Text(
                    "Rate label:N",
                ),
            )
        )

        st.altair_chart(
            (
                merge_rate_bars
                + merge_rate_labels
            ).properties(
                height=330,
            ),
            width="stretch",
        )

        st.caption(
            "Rates for Large and Very large PRs should be interpreted "
            "carefully because these categories contain fewer records."
        )

    else:
        st.info(
            "Merge-rate-by-PR-size data is unavailable."
        )

with relationship_columns[1]:
    st.markdown("### d) Changed files versus merge time")

    st.caption(
        "Shows the relationship between changed files and merge time for "
        "merged PRs within the typical 99th-percentile range. Extreme "
        "long-running PRs are listed separately."
    )

    merge_time_for_scatter = pd.Series(
        pd.NA,
        index=dataframe.index,
        dtype="Float64",
    )

    if merge_duration_column:
        merge_time_for_scatter = numeric_series(
            dataframe[merge_duration_column]
        ).astype(
            "Float64"
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

        merge_time_for_scatter = (
            (
                merged_dates - created_dates
            ).dt.total_seconds()
            / 3600
        ).astype(
            "Float64"
        )

    if merged_status is not None:
        merge_time_for_scatter = merge_time_for_scatter.where(
            merged_status.eq(True)
        )

    if changed_files_column:
        changed_files_merge_time_data = pd.DataFrame(
            {
                "Changed files": numeric_series(
                    dataframe[changed_files_column]
                ),
                "Merge time (hours)": merge_time_for_scatter,
                "PR number": (
                    dataframe[pr_number_column]
                    .astype(str)
                    if pr_number_column
                    else dataframe.index.astype(str)
                ),
                "Title": (
                    dataframe[title_column]
                    .fillna("Not available")
                    .astype(str)
                    if title_column
                    else "Not available"
                ),
            }
        ).dropna(
            subset=[
                "Changed files",
                "Merge time (hours)",
            ]
        )

        changed_files_merge_time_data = (
            changed_files_merge_time_data[
                changed_files_merge_time_data[
                    "Changed files"
                ].ge(0)
                & changed_files_merge_time_data[
                    "Merge time (hours)"
                ].ge(0)
            ]
        )

        changed_files_merge_time_data[
            "Merge time (days)"
        ] = (
            changed_files_merge_time_data[
                "Merge time (hours)"
            ]
            / 24.0
        )

        if not changed_files_merge_time_data.empty:
            changed_files_limit = float(
                changed_files_merge_time_data[
                    "Changed files"
                ].quantile(
                    0.99
                )
            )

            merge_days_limit = float(
                changed_files_merge_time_data[
                    "Merge time (days)"
                ].quantile(
                    0.99
                )
            )

            typical_merge_data = changed_files_merge_time_data[
                changed_files_merge_time_data[
                    "Changed files"
                ].le(
                    changed_files_limit
                )
                & changed_files_merge_time_data[
                    "Merge time (days)"
                ].le(
                    merge_days_limit
                )
            ].copy()

            merge_outlier_data = changed_files_merge_time_data[
                changed_files_merge_time_data[
                    "Changed files"
                ].gt(
                    changed_files_limit
                )
                | changed_files_merge_time_data[
                    "Merge time (days)"
                ].gt(
                    merge_days_limit
                )
            ].copy()

            scatter_points = (
                alt.Chart(
                    typical_merge_data
                )
                .mark_circle(
                    size=70,
                    opacity=0.65,
                )
                .encode(
                    x=alt.X(
                        "Changed files:Q",
                        title="Changed files",
                        scale=alt.Scale(
                            domain=[
                                0,
                                changed_files_limit,
                            ],
                            nice=True,
                        ),
                        axis=alt.Axis(
                            grid=True,
                            tickMinStep=1,
                        ),
                    ),
                    y=alt.Y(
                        "Merge time (days):Q",
                        title="Merge time (days)",
                        scale=alt.Scale(
                            domain=[
                                0,
                                merge_days_limit,
                            ],
                            nice=True,
                        ),
                        axis=alt.Axis(
                            grid=True,
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "PR number:N",
                            title="PR number",
                        ),
                        alt.Tooltip(
                            "Title:N",
                            title="Title",
                        ),
                        alt.Tooltip(
                            "Changed files:Q",
                            title="Changed files",
                            format=",",
                        ),
                        alt.Tooltip(
                            "Merge time (days):Q",
                            title="Merge time (days)",
                            format=".1f",
                        ),
                        alt.Tooltip(
                            "Merge time (hours):Q",
                            title="Merge time (hours)",
                            format=".1f",
                        ),
                    ],
                )
            )

            trend_line = (
                alt.Chart(
                    typical_merge_data
                )
                .transform_regression(
                    "Changed files",
                    "Merge time (days)",
                )
                .mark_line(
                    strokeWidth=2,
                )
                .encode(
                    x=alt.X(
                        "Changed files:Q",
                    ),
                    y=alt.Y(
                        "Merge time (days):Q",
                    ),
                )
            )

            st.altair_chart(
                (
                    scatter_points
                    + trend_line
                )
                .properties(
                    height=330,
                )
                .interactive(),
                width="stretch",
            )

            st.caption(
                f"Main chart includes {len(typical_merge_data):,} merged PRs "
                f"within the 99th-percentile range. "
                f"{len(merge_outlier_data):,} extreme records are listed below."
            )

            if not merge_outlier_data.empty:
                with st.expander(
                    "View extreme merge-time outliers",
                    expanded=False,
                ):
                    st.dataframe(
                        merge_outlier_data[
                            [
                                "PR number",
                                "Title",
                                "Changed files",
                                "Merge time (days)",
                                "Merge time (hours)",
                            ]
                        ].sort_values(
                            by=[
                                "Merge time (days)",
                                "Changed files",
                            ],
                            ascending=False,
                        ),
                        width="stretch",
                        hide_index=True,
                    )

        else:
            st.info(
                "Changed-files-versus-merge-time data is unavailable."
            )

    else:
        st.info(
            "Changed-files-versus-merge-time data is unavailable."
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
            "Created date": created_dates,
        }
    ).dropna()

    if not activity_data.empty:
        activity_data[
            "Month"
        ] = (
            activity_data[
                "Created date"
            ]
            .dt.tz_convert(
                None
            )
            .dt.to_period(
                "M"
            )
            .dt.to_timestamp()
        )

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

        activity_line = (
            alt.Chart(
                monthly_volume
            )
            .mark_line(
                point=True,
                strokeWidth=2,
            )
            .encode(
                x=alt.X(
                    "Month:T",
                    title="Month",
                    axis=alt.Axis(
                        format="%b %Y",
                        labelAngle=-35,
                    ),
                ),
                y=alt.Y(
                    "PR count:Q",
                    title="Pull requests created",
                    axis=alt.Axis(
                        tickMinStep=1,
                        grid=True,
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
            activity_line,
            width="stretch",
        )

        peak_month_row = monthly_volume.loc[
            monthly_volume[
                "PR count"
            ].idxmax()
        ]

        st.caption(
            "This chart shows how pull-request creation volume changed "
            "over time. The highest monthly volume was "
            f"{int(peak_month_row['PR count']):,} PRs in "
            f"{pd.Timestamp(peak_month_row['Month']).strftime('%B %Y')}."
        )

    else:
        st.info(
            "Repository-activity data is unavailable."
        )

else:
    st.info(
        "Repository-activity data is unavailable."
    )


st.markdown("## 4. Filter and search pull requests")

filter_header_columns = st.columns(
    [
        5,
        1,
    ]
)

with filter_header_columns[0]:
    st.caption(
        "Use one or more filters to narrow the table. "
        "All filters work together."
    )

with filter_header_columns[1]:
    st.button(
        "Reset filters",
        on_click=_reset_filters,
        width="stretch",
        help=(
            "Clear the search and return every filter to All so the "
            "table shows the full pull-request population again."
        ),
    )


filter_row_one = st.columns(3)

search_text = filter_row_one[0].text_input(
    "Search PR number or title",
    placeholder="Example: 5336 or send_file",
    key="pr_search_text",
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

if (
    st.session_state.get(
        "author_filter",
        "All",
    )
    not in author_options
):
    st.session_state[
        "author_filter"
    ] = "All"

selected_author = filter_row_one[1].selectbox(
    "Author",
    options=author_options,
    key="author_filter",
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

if (
    st.session_state.get(
        "risk_filter",
        "All",
    )
    not in risk_options
):
    st.session_state[
        "risk_filter"
    ] = "All"

selected_risk = filter_row_one[2].selectbox(
    "Policy risk",
    options=risk_options,
    key="risk_filter",
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

if (
    st.session_state.get(
        "priority_filter",
        "All",
    )
    not in priority_options
):
    st.session_state[
        "priority_filter"
    ] = "All"

selected_priority = filter_row_two[0].selectbox(
    "Review priority",
    options=priority_options,
    key="priority_filter",
)

merge_prediction_options = [
    "All",
    "Predicted to merge",
    "Predicted not to merge",
]

if (
    st.session_state.get(
        "merge_prediction_filter",
        "All",
    )
    not in merge_prediction_options
):
    st.session_state[
        "merge_prediction_filter"
    ] = "All"

selected_merge_prediction = filter_row_two[1].selectbox(
    "Model 1 prediction",
    options=merge_prediction_options,
    key="merge_prediction_filter",
)

delay_prediction_options = [
    "All",
    "Predicted delayed",
    "Predicted not delayed",
]

if (
    st.session_state.get(
        "delay_prediction_filter",
        "All",
    )
    not in delay_prediction_options
):
    st.session_state[
        "delay_prediction_filter"
    ] = "All"

selected_delay_prediction = filter_row_two[2].selectbox(
    "Model 2 prediction",
    options=delay_prediction_options,
    key="delay_prediction_filter",
)

manual_review_options = [
    "All",
    "Required",
    "Not required",
]

if (
    st.session_state.get(
        "manual_review_filter",
        "All",
    )
    not in manual_review_options
):
    st.session_state[
        "manual_review_filter"
    ] = "All"

selected_review = filter_row_two[3].selectbox(
    "Manual review",
    options=manual_review_options,
    key="manual_review_filter",
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
    f"Showing {len(filtered):,} of {len(dataframe):,} pull requests. "
    "Select one row to inspect that PR below."
)

table_filter_signature = (
    search_text,
    selected_author,
    selected_risk,
    selected_priority,
    selected_merge_prediction,
    selected_delay_prediction,
    selected_review,
)

table_selection_key = (
    "pr_table_selection_"
    + str(
        abs(
            hash(
                table_filter_signature
            )
        )
    )
)

table_event = st.dataframe(
    display_frame,
    width="stretch",
    hide_index=True,
    height=500,
    on_select="rerun",
    selection_mode="single-row",
    key=table_selection_key,
)


st.markdown("## 6. Inspect one pull request")

if filtered.empty:
    st.info(
        "No pull requests match the current search and filter selections."
    )
    st.stop()

selected_positions: list[int] = []

try:
    selected_positions = list(
        table_event.selection.rows
    )
except (
    AttributeError,
    TypeError,
):
    try:
        selected_positions = list(
            table_event.get(
                "selection",
                {},
            ).get(
                "rows",
                [],
            )
        )
    except (
        AttributeError,
        TypeError,
    ):
        selected_positions = []

if len(filtered) == 1:
    selected_position = 0
elif selected_positions:
    selected_position = int(
        selected_positions[0]
    )
else:
    st.info(
        "Select a row in the searchable PR table above. "
        "When the filters return exactly one PR, its details will load automatically."
    )
    st.stop()

if (
    selected_position < 0
    or selected_position >= len(filtered)
):
    st.info(
        "The selected table row is no longer available after filtering. "
        "Select a row again."
    )
    st.stop()

selected_row = filtered.iloc[
    selected_position
]


selected_detail_columns = st.columns(3)

with selected_detail_columns[0]:
    st.markdown("#### Identity and activity")

    identity_rows: list[dict[str, str]] = []

    identity_field_labels: list[tuple[str | None, str]] = [
        (
            author_column,
            "Author",
        ),
        (
            created_column,
            "Created at",
        ),
        (
            merged_at_column,
            "Merged at",
        ),
        (
            "source_branch",
            "Source branch",
        ),
        (
            "head_branch",
            "Head branch",
        ),
        (
            "target_branch",
            "Target branch",
        ),
        (
            "base_branch",
            "Base branch",
        ),
        (
            "source_url",
            "Source URL",
        ),
        (
            "html_url",
            "GitHub URL",
        ),
        (
            "pr_url",
            "PR URL",
        ),
    ]

    for field, label in identity_field_labels:
        if (
            field is not None
            and field in selected_row.index
            and pd.notna(selected_row.get(field))
            and not (
                isinstance(selected_row.get(field), str)
                and not selected_row.get(field).strip()
            )
        ):
            identity_rows.append(
                {
                    "Field": label,
                    "Value": safe_text(
                        selected_row.get(field)
                    ),
                }
            )

    if (
        resolution_hours_column is not None
        and resolution_hours_column in selected_row.index
        and pd.notna(
            selected_row.get(
                resolution_hours_column
            )
        )
    ):
        identity_rows.append(
            {
                "Field": "Time to resolution",
                "Value": _format_hours_duration(
                    selected_row.get(
                        resolution_hours_column
                    )
                ),
            }
        )

    if identity_rows:
        st.dataframe(
            pd.DataFrame(identity_rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "Identity and activity information is unavailable for this pull request."
        )

with selected_detail_columns[1]:
    st.markdown("#### Predictive intelligence")

    predictive_rows: list[dict[str, str]] = []

    if (
        merge_prediction_column is not None
        and merge_prediction_column in selected_row.index
        and pd.notna(selected_row.get(merge_prediction_column))
    ):
        predictive_rows.append(
            {
                "Field": "Merge prediction",
                "Value": _normalise_prediction_label(
                    selected_row.get(merge_prediction_column),
                    "Predicted to merge",
                    "Predicted not to merge",
                ),
            }
        )

    if (
        merge_probability_column is not None
        and merge_probability_column in selected_row.index
        and pd.notna(selected_row.get(merge_probability_column))
    ):
        merge_probability_value = pd.to_numeric(
            pd.Series([selected_row.get(merge_probability_column)]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(merge_probability_value):
            merge_probability_value = float(merge_probability_value)
            if merge_probability_value > 1:
                merge_probability_value = merge_probability_value / 100.0

            predictive_rows.append(
                {
                    "Field": "Merge probability",
                    "Value": f"{merge_probability_value:.2%}",
                }
            )

    predictive_rows.append(
        {
            "Field": "Merge threshold",
            "Value": f"{MERGE_DECISION_THRESHOLD:.2%}",
        }
    )

    if (
        delay_prediction_column is not None
        and delay_prediction_column in selected_row.index
        and pd.notna(selected_row.get(delay_prediction_column))
    ):
        predictive_rows.append(
            {
                "Field": "Delay prediction",
                "Value": _normalise_prediction_label(
                    selected_row.get(delay_prediction_column),
                    "Predicted delayed",
                    "Predicted not delayed",
                ),
            }
        )

    if (
        delay_probability_column is not None
        and delay_probability_column in selected_row.index
        and pd.notna(selected_row.get(delay_probability_column))
    ):
        delay_probability_value = pd.to_numeric(
            pd.Series([selected_row.get(delay_probability_column)]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(delay_probability_value):
            delay_probability_value = float(delay_probability_value)
            if delay_probability_value > 1:
                delay_probability_value = delay_probability_value / 100.0

            predictive_rows.append(
                {
                    "Field": "Delay probability",
                    "Value": f"{delay_probability_value:.2%}",
                }
            )

    delay_prediction_available = (
        delay_prediction_column is not None
        and delay_prediction_column in selected_row.index
        and pd.notna(
            selected_row.get(
                delay_prediction_column
            )
        )
    )

    delay_probability_available = (
        delay_probability_column is not None
        and delay_probability_column in selected_row.index
        and pd.notna(
            selected_row.get(
                delay_probability_column
            )
        )
    )

    if (
        delay_prediction_available
        and delay_probability_available
    ):
        predictive_rows.append(
            {
                "Field": "Delay threshold",
                "Value": f"{DELAY_DECISION_THRESHOLD:.2%}",
            }
        )

    if (
        merge_duration_column is not None
        and merge_duration_column in selected_row.index
        and pd.notna(selected_row.get(merge_duration_column))
    ):
        merge_hours_value = pd.to_numeric(
            pd.Series([selected_row.get(merge_duration_column)]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(merge_hours_value):
            merge_hours_value = float(merge_hours_value)

            merge_time_display = _format_hours_duration(
                merge_hours_value
            )

            predictive_rows.append(
                {
                    "Field": "Actual merge time",
                    "Value": merge_time_display,
                }
            )

    if predictive_rows:
        st.dataframe(
            pd.DataFrame(predictive_rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "Predictive intelligence is unavailable for this pull request."
        )


with selected_detail_columns[2]:
    st.markdown("#### Governance intelligence")

    _render_governance_table(
        row=selected_row,
        risk_field=risk_column,
        priority_field=priority_column,
        manual_review_field=manual_review_column,
    )


st.markdown("#### Recommended action")

recommended_action_fields = [
    "recommended_next_action",
    "recommended_action",
    "next_action",
    "review_recommendation",
    "final_recommendation",
    "rule_recommendations",
    "policy_recommendations",
    "governance_note",
]

_render_field_table(
    row=selected_row,
    fields=recommended_action_fields,
    empty_message=(
        "No recommended action is available for this pull request."
    ),
)


lower_detail_columns = st.columns(3)

with lower_detail_columns[2]:
    st.markdown("#### Complexity and engagement")

    complexity_rows: list[tuple[str, Any]] = [
        (
            "Changed files",
            selected_row.get(changed_files_column)
            if changed_files_column is not None
            else None,
        ),
        (
            "Lines added",
            selected_row.get(additions_column)
            if additions_column is not None
            else None,
        ),
        (
            "Lines deleted",
            selected_row.get(deletions_column)
            if deletions_column is not None
            else None,
        ),
        (
            "Total line changes",
            selected_row.get(total_changes_column)
            if total_changes_column is not None
            else None,
        ),
        (
            "Commit count",
            selected_row.get("commit_count")
            if "commit_count" in selected_row.index
            else None,
        ),
        (
            "PR description word count",
            selected_row.get("body_word_count")
            if "body_word_count" in selected_row.index
            else None,
        ),
    ]

    _render_named_value_table(
        rows=complexity_rows,
        empty_message=(
            "Complexity and engagement information is unavailable."
        ),
    )


with lower_detail_columns[1]:
    st.markdown("#### File-level summary")

    _render_named_value_table(
        rows=[
            (
                "Test files changed",
                selected_row.get(test_files_changed_column)
                if test_files_changed_column is not None
                else None,
            ),
            (
                "Documentation files changed",
                selected_row.get(documentation_files_changed_column)
                if documentation_files_changed_column is not None
                else None,
            ),
            (
                "Configuration files changed",
                selected_row.get(configuration_files_changed_column)
                if configuration_files_changed_column is not None
                else None,
            ),
            (
                "Security-sensitive files changed",
                selected_row.get(security_sensitive_files_changed_column)
                if security_sensitive_files_changed_column is not None
                else None,
            ),
        ],
        empty_message=(
            "File-level summary is unavailable in the current dataset."
        ),
        omit_zero_values=False,
    )


with lower_detail_columns[0]:
    st.markdown("#### File operations")

    _render_named_value_table(
        rows=[
            (
                "Total files affected",
                selected_row.get(changed_files_column)
                if changed_files_column is not None
                else None,
            ),
            (
                "Files added",
                selected_row.get(files_added_column)
                if files_added_column is not None
                else None,
            ),
            (
                "Files modified",
                selected_row.get(files_modified_column)
                if files_modified_column is not None
                else None,
            ),
            (
                "Files removed",
                selected_row.get(files_removed_column)
                if files_removed_column is not None
                else None,
            ),
            (
                "Files renamed",
                selected_row.get(files_renamed_column)
                if files_renamed_column is not None
                else None,
            ),
        ],
        empty_message=(
            "File-operation information is unavailable in the current dataset."
        ),
        omit_zero_values=False,
    )


with st.expander(
    "View PR description",
    expanded=False,
):
    if (
        description_column is not None
        and description_column in selected_row.index
        and pd.notna(
            selected_row.get(
                description_column
            )
        )
        and str(
            selected_row.get(
                description_column
            )
        ).strip()
    ):
        st.markdown(
            str(
                selected_row.get(
                    description_column
                )
            )
        )
    else:
        st.info(
            "No PR description is available for this pull request."
        )
