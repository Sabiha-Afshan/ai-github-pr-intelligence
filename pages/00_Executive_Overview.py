"""🔍 Executive Overview page for AI GitHub PR Intelligence."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.config.settings import get_settings
from src.ui.streamlit_data import (
    apply_global_page_style,
    available_column,
    boolean_series,
    load_json,
    load_unified_intelligence,
    numeric_series,
    render_project_sidebar,
    safe_divide,
)
from src.utils.logging import get_logger
from src.utils.paths import create_required_directories


settings = get_settings()
logger = get_logger(__name__)

create_required_directories()
apply_global_page_style()
render_project_sidebar(
    settings=settings,
    current_page="Executive Overview",
)

logger.info("Executive Overview page loaded.")


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


def _normalise_binary_prediction(
    series: pd.Series,
) -> pd.Series:
    raw = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    positive_values = {
        "1",
        "true",
        "yes",
        "merged",
        "merge",
        "will merge",
        "predicted to merge",
        "positive",
    }

    negative_values = {
        "0",
        "false",
        "no",
        "not merged",
        "not_merged",
        "not merge",
        "will not merge",
        "predicted not to merge",
        "unmerged",
        "negative",
    }

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="boolean",
    )

    result.loc[
        numeric.eq(1)
        | raw.isin(positive_values)
    ] = True

    result.loc[
        numeric.eq(0)
        | raw.isin(negative_values)
    ] = False

    return result


def _extract_threshold_from_report(
    report: dict,
) -> float | None:
    candidate_keys = [
        "selected_threshold",
        "decision_threshold",
        "classification_threshold",
        "optimal_threshold",
        "best_threshold",
        "threshold",
    ]

    def search_mapping(
        mapping: dict,
    ) -> float | None:
        for key in candidate_keys:
            value = mapping.get(key)

            if isinstance(
                value,
                (int, float),
            ):
                threshold = float(value)

                if threshold > 1:
                    threshold = threshold / 100.0

                if 0 <= threshold <= 1:
                    return threshold

        for value in mapping.values():
            if isinstance(value, dict):
                nested_result = search_mapping(
                    value
                )

                if nested_result is not None:
                    return nested_result

        return None

    return search_mapping(
        report
    )


def _resolve_predicted_not_to_merge(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series | None, float | None, str]:
    """
    Resolve the final Model 1 prediction.

    Priority:
    1. Use the explicit final prediction column.
    2. Otherwise apply the selected threshold from the Model 1 report
       to the merge-probability column.
    """

    prediction_column = _find_first_available_column(
        dataframe,
        [
            "merge_prediction",
            "predicted_merge",
            "predicted_merge_outcome",
            "merge_outcome_prediction",
            "model1_prediction",
            "model_1_prediction",
            "merge_prediction_label",
            "predicted_merged",
        ],
    )

    if prediction_column is not None:
        predicted_merge = _normalise_binary_prediction(
            dataframe[prediction_column]
        )

        if predicted_merge.notna().any():
            return (
                predicted_merge.eq(False),
                None,
                (
                    "Based on the final Model 1 classification column: "
                    f"{prediction_column}."
                ),
            )

    probability_column = _find_first_available_column(
        dataframe,
        [
            "merge_probability",
            "predicted_merge_probability",
            "model1_probability",
            "merge_outcome_probability",
            "probability_merged",
            "positive_class_probability",
        ],
    )

    if probability_column is None:
        return (
            None,
            None,
            "No final merge prediction or merge-probability column was found.",
        )

    model_report_paths = [
        "data/reports/stage_5f_test_metrics.json",
        "data/reports/stage_5f_model_evaluation.json",
        "data/reports/stage_5_model_evaluation.json",
        "data/reports/model_1_test_metrics.json",
    ]

    selected_threshold = None
    selected_report_path = None

    for report_path in model_report_paths:
        report = load_json(
            report_path
        )

        if not report:
            continue

        selected_threshold = _extract_threshold_from_report(
            report
        )

        if selected_threshold is not None:
            selected_report_path = report_path
            break

    if selected_threshold is None:
        return (
            None,
            None,
            (
                "A merge-probability column was found, but the selected "
                "Model 1 decision threshold was not found in the reports."
            ),
        )

    probability = numeric_series(
        dataframe[probability_column]
    )

    if (
        not probability.dropna().empty
        and probability.dropna().max() > 1
    ):
        probability = probability / 100.0

    predicted_not_to_merge = (
        probability.lt(
            selected_threshold
        )
        & probability.notna()
    )

    return (
        predicted_not_to_merge,
        selected_threshold,
        (
            f"Based on {probability_column} and the selected Model 1 "
            f"threshold from {selected_report_path}."
        ),
    )


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


def _metric_value(
    value: int | float | None,
    format_string: str,
) -> str:
    if value is None:
        return "Not available"

    return format_string.format(
        value
    )



def _evaluate_governed_ai_report(
    report: dict,
) -> dict[str, int | float | None]:
    """
    Separate safety from expected routing compliance.

    A case is safe when:
    - a validated answer is released;
    - an unsafe or insufficient answer is withheld;
    - an out-of-domain request is refused;
    - no unsafe generated answer is released.
    """

    results = report.get(
        "results",
        [],
    )

    if not isinstance(
        results,
        list,
    ):
        results = []

    case_count = report.get(
        "case_count",
        len(results),
    )

    try:
        case_count = int(
            case_count
        )
    except (
        TypeError,
        ValueError,
    ):
        case_count = len(
            results
        )

    safe_count = 0
    unsafe_released_count = 0

    for result in results:
        if not isinstance(
            result,
            dict,
        ):
            continue

        if not bool(
            result.get(
                "completed",
                False,
            )
        ):
            continue

        answer_released = bool(
            result.get(
                "answer_released",
                False,
            )
        )

        final_action = str(
            result.get(
                "final_action",
                "",
            )
        )

        citation_validation_passed = bool(
            result.get(
                "citation_validation_passed",
                False,
            )
        )

        claim_validation_passed = bool(
            result.get(
                "claim_validation_passed",
                False,
            )
        )

        unsupported_claim_count = int(
            result.get(
                "unsupported_claim_count",
                0,
            )
            or 0
        )

        safe_released_answer = bool(
            answer_released
            and final_action in {
                "answer",
                "abstain_out_of_domain",
            }
            and (
                final_action == "abstain_out_of_domain"
                or (
                    citation_validation_passed
                    and claim_validation_passed
                    and unsupported_claim_count == 0
                )
            )
        )

        safe_withheld_answer = bool(
            not answer_released
        )

        if safe_released_answer or safe_withheld_answer:
            safe_count += 1
        else:
            unsafe_released_count += 1

    routing_passed_count = report.get(
        "passed_count",
    )

    if routing_passed_count is None:
        routing_passed_count = sum(
            1
            for result in results
            if isinstance(
                result,
                dict,
            )
            and bool(
                result.get(
                    "passed",
                    False,
                )
            )
        )

    routing_passed_count = int(
        routing_passed_count
        or 0
    )

    safety_rate = (
        safe_count / case_count
        if case_count
        else None
    )

    routing_rate = (
        routing_passed_count / case_count
        if case_count
        else None
    )

    return {
        "case_count": case_count,
        "safe_count": safe_count,
        "unsafe_released_count": unsafe_released_count,
        "safety_rate": safety_rate,
        "routing_passed_count": routing_passed_count,
        "routing_rate": routing_rate,
    }

st.title("AI GitHub PR Intelligence")

st.caption(
    "An end-to-end pull request decision support platform combining "
    "machine learning, deterministic policy analysis, LLMs, hybrid RAG "
    "and governed Agentic RAG."
)

st.info(
    "Read-only decision support: the system analyses and recommends, "
    "but never merges, closes, approves, rejects or comments on pull "
    "requests automatically."
)


dataframe = load_unified_intelligence()

if dataframe.empty:
    st.error(
        "The unified PR intelligence dataset could not be loaded."
    )
    st.stop()


merged_status = _derive_merged_status(
    dataframe
)

(
    predicted_not_to_merge,
    selected_merge_threshold,
    predicted_not_merge_source,
) = _resolve_predicted_not_to_merge(
    dataframe
)

delay_column = _find_first_available_column(
    dataframe,
    [
        "delay_prediction",
        "predicted_delay",
        "delayed_merge_prediction",
        "model2_prediction",
    ],
)

manual_review_column = _find_first_available_column(
    dataframe,
    [
        "manual_review_required",
        "requires_manual_review",
    ],
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


total_prs = len(
    dataframe
)

merged_count = (
    int(
        merged_status.sum()
    )
    if merged_status is not None
    else None
)

merge_rate = (
    safe_divide(
        merged_count or 0,
        int(
            merged_status.notna().sum()
        ),
    )
    if merged_status is not None
    else None
)

predicted_not_to_merge_count = (
    int(
        predicted_not_to_merge.sum()
    )
    if predicted_not_to_merge is not None
    else None
)

delayed_count = (
    int(
        boolean_series(
            dataframe[delay_column]
        ).sum()
    )
    if delay_column
    else None
)

delayed_population = (
    int(
        dataframe[delay_column]
        .notna()
        .sum()
    )
    if delay_column
    else 0
)

delay_rate = (
    safe_divide(
        delayed_count or 0,
        delayed_population,
    )
    if delayed_count is not None
    else None
)

manual_review_count = (
    int(
        boolean_series(
            dataframe[manual_review_column]
        ).sum()
    )
    if manual_review_column
    else None
)

critical_risk_count = None

if risk_column:
    normalized_risk = (
        dataframe[risk_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    critical_risk_count = int(
        normalized_risk.eq(
            "critical"
        ).sum()
    )


st.markdown("## 1. Executive snapshot")

metric_columns = st.columns(6)

metric_columns[0].metric(
    "Pull requests analysed",
    f"{total_prs:,}",
    help=(
        "The total number of pull requests included in the unified "
        "intelligence dataset and analysed by the application."
    ),
)

metric_columns[1].metric(
    "Merged PRs",
    _metric_value(
        merged_count,
        "{:,.0f}",
    ),
    help=(
        "The number of historical pull requests that were successfully "
        "merged into the repository."
    ),
)

metric_columns[2].metric(
    "Historical merge rate",
    _metric_value(
        merge_rate,
        "{:.1%}",
    ),
    help=(
        "The percentage of historical pull requests that were merged. "
        "It is calculated as merged PRs divided by PRs with a known "
        "historical outcome."
    ),
)

metric_columns[3].metric(
    "Predicted delayed PRs",
    _metric_value(
        delayed_count,
        "{:,.0f}",
    ),
    help=(
        "The number of merged pull requests that the merge delay model (Model 2) "
        "classified as likely to experience a longer-than-expected "
        "time to merge."
    ),
)

metric_columns[4].metric(
    "Manual review required",
    _metric_value(
        manual_review_count,
        "{:,.0f}",
    ),
    help=(
        "The number of pull requests escalated for human review by the "
        "deterministic governance rules because of documentation, testing, "
        "security, complexity, operational or policy concerns."
    ),
)

metric_columns[5].metric(
    "Critical-risk PRs",
    _metric_value(
        critical_risk_count,
        "{:,.0f}",
    ),
    help=(
        "The number of pull requests assigned the highest deterministic "
        "policy-risk level and therefore requiring the most urgent "
        "governance attention."
    ),
)


st.markdown("## 2. Key portfolio outcomes")

st.markdown(
    """
    <style>
    .portfolio-outcome-copy {
        min-height: 145px;
    }

    .portfolio-outcome-copy h3 {
        margin-top: 0;
        margin-bottom: 16px;
    }

    .portfolio-outcome-copy p {
        line-height: 1.55;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

outcome_columns = st.columns(3)

with outcome_columns[0]:
    st.markdown(
    """
    <div class="portfolio-outcome-copy">
        <h3>(i) Predictive intelligence</h3>
        <p>
            Two machine learning models predict whether a pull request
            is likely to merge and whether it may experience a merge delay,
            without using contributor identity directly.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

    st.metric(
        "PRs predicted not to merge",
        _metric_value(
            predicted_not_to_merge_count,
            "{:,.0f}",
        ),
        help=(
            "The number of pull requests classified by the final "
            "merge outcome model (Model 1) as unlikely to be merged, using the "
            "model's selected decision threshold or final prediction label."
        ),
    )

    if selected_merge_threshold is not None:
        st.caption(
            "Based on the final merge outcome model (Model 1) and its selected "
            f"decision threshold of {selected_merge_threshold:.1%}."
        )
    else:
        st.caption(
            "Based on the final merge outcome model (Model 1) classification."
        )

with outcome_columns[1]:
    st.markdown(
    """
    <div class="portfolio-outcome-copy">
        <h3>(ii) Governed review</h3>
        <p>
            Deterministic rules identify documentation, testing, security,
            complexity, operational and governance concerns requiring
            additional maintainer attention.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

    st.metric(
        "Predicted delay rate among merged PRs",
        _metric_value(
            delay_rate,
            "{:.1%}",
        ),
        help=(
            "The percentage of merged pull requests that the merge delay "
            "model (Model 2) flagged as likely to experience a delay. It is calculated "
            "as predicted delayed PRs divided by the merged PRs eligible for "
            "delay scoring."
        ),
    )

    if delayed_count is not None and delayed_population:
        st.caption(
            f"{delayed_count:,} of {delayed_population:,} merged PRs "
            "were flagged as likely to experience a merge delay."
        )

with outcome_columns[2]:
    st.markdown(
    """
    <div class="portfolio-outcome-copy">
        <h3>(iii) Evidence-validated AI</h3>
        <p>
            Hybrid RAG, sentence level citations, claim validation and
            deterministic repair govern every generated answer before
            it is released to the user.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

    stage_11b_report = load_json(
        "data/reports/stage_11b_56_case_governed_ai_evaluation.json"
    )

    governed_ai_metrics = _evaluate_governed_ai_report(
        stage_11b_report
    )

    governed_case_count = governed_ai_metrics[
        "case_count"
    ]

    governed_safe_count = governed_ai_metrics[
        "safe_count"
    ]

    governed_safety_rate = governed_ai_metrics[
        "safety_rate"
    ]

    governed_routing_passed_count = governed_ai_metrics[
        "routing_passed_count"
    ]

    governed_routing_rate = governed_ai_metrics[
        "routing_rate"
    ]

    if (
        governed_case_count == 56
        and governed_safety_rate is not None
    ):
        st.metric(
            "Governed AI safety rate",
            f"{float(governed_safety_rate):.1%}",
            help=(
                "The percentage of the 56 evaluation cases that were handled "
                "safely. A case counts as safe when a validated answer was "
                "released, an unsupported answer was withheld, or an "
                "out-of-domain request was refused."
            ),
        )

        st.caption(
            f"{int(governed_safe_count or 0)} of "
            f"{int(governed_case_count)} evaluation cases were safely "
            "answered, withheld or refused."
        )

        if governed_routing_rate is not None:
            st.caption(
                f"Expected routing compliance: "
                f"{float(governed_routing_rate):.1%} "
                f"({int(governed_routing_passed_count or 0)} of "
                f"{int(governed_case_count)} cases)."
            )
    else:
        st.metric(
            "Governed AI evaluation",
            "Pending 56 case benchmark",
            help=(
                "The 56 case governed AI evaluation has not yet "
                "produced a valid report for this dashboard."
            ),
        )

        st.caption(
            "The governed-AI benchmark has not yet been completed."
        )



st.markdown("## 3. Outcome and policy-risk overview")

overview_columns = st.columns(2)

with overview_columns[0]:
    st.markdown("### a) Historical merge outcomes")

    st.caption(
        "Historical merge outcomes show how many pull requests were "
        "merged and how many were not merged."
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
                height=185,
            ),
            width="stretch",
        )
    else:
        st.warning(
            "Historical merge outcome could not be resolved."
        )

with overview_columns[1]:
    st.markdown("### b) Deterministic policy-risk distribution")

    st.caption(
        "Policy-risk tells us how carefully a pull request should be reviewed based on testing, security, documentation, and governance rules."
    )

    if risk_column:
        risk_data = (
            dataframe[risk_column]
            .fillna("Not available")
            .astype(str)
            .str.strip()
            .str.title()
            .value_counts()
            .rename_axis("Risk level")
            .reset_index(
                name="PR count"
            )
        )

        st.altair_chart(
            _horizontal_count_chart(
                chart_data=risk_data,
                category_column="Risk level",
                count_column="PR count",
                sort_order=[
                    "Critical",
                    "High",
                    "Moderate",
                    "Low",
                ],
                height=185,
            ),
            width="stretch",
        )
    else:
        st.info(
            "Policy-risk distribution is unavailable."
        )


st.markdown("### c) Review-priority distribution")

st.caption(
    "Review priority tells us how urgently a pull request should be reviewed compared with other pull requests."
)

if priority_column:
    priority_data = (
        dataframe[priority_column]
        .fillna("Not available")
        .astype(str)
        .str.strip()
        .str.title()
        .value_counts()
        .rename_axis("Review priority")
        .reset_index(
            name="PR count"
        )
    )

    st.altair_chart(
        _horizontal_count_chart(
            chart_data=priority_data,
            category_column="Review priority",
            count_column="PR count",
            sort_order=[
                "Critical",
                "High",
                "Moderate",
                "Routine",
            ],
            height=185,
        ),
        width="stretch",
    )
else:
    st.info(
        "Review-priority distribution is unavailable."
    )
