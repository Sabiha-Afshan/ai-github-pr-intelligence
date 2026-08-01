"""Detailed pull-request intelligence page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config.settings import get_settings
from src.ui.streamlit_data import (
    apply_global_page_style,
    available_column,
    boolean_series,
    format_probability,
    load_unified_intelligence,
    parse_list_value,
    render_project_sidebar,
    safe_number,
    safe_text,
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
    current_page="PR Intelligence",
)

logger.info("PR Intelligence page loaded.")

st.title("PR Intelligence")
st.caption(
    "Inspect predictive outputs, deterministic policy results, risk "
    "signals and the governed review recommendation for one PR."
)


dataframe = load_unified_intelligence()

if dataframe.empty:
    st.error(
        "The unified PR intelligence dataset could not be loaded."
    )
    st.stop()


pr_number_column = select_pr_number_column(dataframe)
title_column = select_title_column(dataframe)
repository_column = select_repository_column(dataframe)

if not pr_number_column:
    st.error(
        "The dataset does not contain a PR-number column."
    )
    st.stop()


selected_pr_number = st.selectbox(
    "Select pull request",
    options=(
        dataframe[pr_number_column]
        .dropna()
        .sort_values()
        .tolist()
    ),
)

row = (
    dataframe[
        dataframe[pr_number_column]
        == selected_pr_number
    ]
    .iloc[0]
)


merge_probability_column = available_column(
    dataframe,
    ["merge_probability", "model1_probability", "predicted_merge_probability"],
)

merge_prediction_column = available_column(
    dataframe,
    ["merge_prediction", "model1_prediction", "predicted_merge"],
)

merge_threshold_column = available_column(
    dataframe,
    ["merge_decision_threshold", "model1_threshold", "merge_threshold"],
)

delay_probability_column = available_column(
    dataframe,
    ["delay_probability", "model2_probability", "predicted_delay_probability"],
)

delay_prediction_column = available_column(
    dataframe,
    ["delay_prediction", "model2_prediction", "predicted_delay"],
)

risk_score_column = available_column(
    dataframe,
    ["policy_risk_score", "risk_score", "total_risk_score"],
)

risk_band_column = available_column(
    dataframe,
    ["policy_risk_band", "risk_band", "risk_level", "policy_risk_level"],
)

manual_review_column = available_column(
    dataframe,
    ["manual_review_required", "requires_manual_review"],
)

priority_column = available_column(
    dataframe,
    ["review_priority", "priority_band", "unified_review_priority"],
)

priority_score_column = available_column(
    dataframe,
    ["review_priority_score", "priority_score", "governed_ranking_score"],
)

triggered_rules_column = available_column(
    dataframe,
    ["triggered_rules", "policy_rules_triggered"],
)

triggered_categories_column = available_column(
    dataframe,
    ["triggered_categories", "policy_categories_triggered"],
)

recommended_action_column = available_column(
    dataframe,
    ["recommended_next_action", "review_recommendation", "recommended_action"],
)


st.markdown(
    f"## PR #{safe_text(row.get(pr_number_column))}: "
    f"{safe_text(row.get(title_column) if title_column else None)}"
)

identity_columns = st.columns(4)

identity_columns[0].metric(
    "Repository",
    safe_text(
        row.get(repository_column)
        if repository_column
        else None
    ),
)

identity_columns[1].metric(
    "Author",
    safe_text(
        row.get("author")
    ),
)

identity_columns[2].metric(
    "State",
    safe_text(
        row.get("state")
    ),
)

identity_columns[3].metric(
    "Created",
    safe_text(
        row.get("created_at")
    ),
)


st.markdown("## Predictive intelligence")

prediction_columns = st.columns(4)

prediction_columns[0].metric(
    "Merge probability",
    (
        format_probability(
            row.get(
                merge_probability_column
            )
        )
        if merge_probability_column
        else "Not available"
    ),
)

prediction_columns[1].metric(
    "Merge prediction",
    (
        "Predicted to merge"
        if merge_prediction_column
        and boolean_series(
            pd.Series(
                [row.get(merge_prediction_column)]
            )
        ).iloc[0]
        else "Predicted not to merge"
    ),
)

prediction_columns[2].metric(
    "Delay probability",
    (
        format_probability(
            row.get(
                delay_probability_column
            )
        )
        if delay_probability_column
        and pd.notna(
            row.get(
                delay_probability_column
            )
        )
        else "Not applicable"
    ),
)

prediction_columns[3].metric(
    "Delay prediction",
    (
        "Delayed"
        if delay_prediction_column
        and boolean_series(
            pd.Series(
                [row.get(delay_prediction_column)]
            )
        ).iloc[0]
        else "Not delayed / not applicable"
    ),
)

if merge_threshold_column:
    st.caption(
        "Merge decision threshold: "
        + format_probability(
            row.get(
                merge_threshold_column
            )
        )
    )


st.markdown("## Policy and review intelligence")

policy_columns = st.columns(4)

policy_columns[0].metric(
    "Policy risk",
    safe_text(
        row.get(risk_band_column)
        if risk_band_column
        else None
    ),
)

policy_columns[1].metric(
    "Risk score",
    (
        f"{safe_number(row.get(risk_score_column)):.2f}"
        if risk_score_column
        else "Not available"
    ),
)

policy_columns[2].metric(
    "Review priority",
    safe_text(
        row.get(priority_column)
        if priority_column
        else None
    ),
)

policy_columns[3].metric(
    "Manual review",
    (
        "Required"
        if manual_review_column
        and boolean_series(
            pd.Series(
                [row.get(manual_review_column)]
            )
        ).iloc[0]
        else "Not required"
    ),
)


if priority_score_column:
    st.write(
        "**Unified priority score:** "
        f"{safe_number(row.get(priority_score_column)):.4f}"
    )

if recommended_action_column:
    st.info(
        "**Recommended next action:** "
        + safe_text(
            row.get(
                recommended_action_column
            )
        )
    )


rule_columns = st.columns(2)

with rule_columns[0]:
    st.markdown("### Triggered rules")

    triggered_rules = (
        parse_list_value(
            row.get(
                triggered_rules_column
            )
        )
        if triggered_rules_column
        else []
    )

    if triggered_rules:
        for rule in triggered_rules:
            st.write(f"- {rule}")
    else:
        st.write("No deterministic policy rules were triggered.")

with rule_columns[1]:
    st.markdown("### Triggered categories")

    triggered_categories = (
        parse_list_value(
            row.get(
                triggered_categories_column
            )
        )
        if triggered_categories_column
        else []
    )

    if triggered_categories:
        for category in triggered_categories:
            st.write(f"- {category}")
    else:
        st.write("No policy categories were triggered.")


st.markdown("## Change and quality evidence")

evidence_fields = [
    "description_present",
    "detailed_description",
    "description_word_count",
    "changed_files",
    "total_changes",
    "additions",
    "deletions",
    "commit_count",
    "requested_reviewer_count",
    "comment_count",
    "test_files_changed",
    "documentation_files_changed",
    "security_sensitive_files_changed",
    "configuration_files_changed",
]

available_evidence_fields = [
    field
    for field in evidence_fields
    if field in row.index
]

evidence_table = pd.DataFrame(
    {
        "Evidence field": available_evidence_fields,
        "Value": [
            safe_text(
                row.get(field)
            )
            for field in available_evidence_fields
        ],
    }
)

if not evidence_table.empty:
    st.dataframe(
        evidence_table,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "Detailed change-evidence fields were not found."
    )


st.markdown("## Decision interpretation")

merge_probability = (
    safe_number(
        row.get(
            merge_probability_column
        )
    )
    if merge_probability_column
    else 0.0
)

if merge_probability > 1:
    merge_probability /= 100.0

manual_review_required = (
    manual_review_column is not None
    and boolean_series(
        pd.Series(
            [row.get(manual_review_column)]
        )
    ).iloc[0]
)

risk_band = (
    safe_text(
        row.get(risk_band_column)
    )
    if risk_band_column
    else "Not available"
)

if manual_review_required:
    st.warning(
        "This PR requires human governance review. The model prediction "
        "should be treated as decision support only, and the triggered "
        "policy evidence should be reviewed before progression."
    )
elif merge_probability < 0.4:
    st.warning(
        "The predicted merge probability is low. Review the description, "
        "tests, change size, policy evidence and reviewer readiness."
    )
else:
    st.success(
        "No manual-review requirement is currently recorded. Standard "
        "maintainer review and repository checks still apply."
    )

st.write(
    f"**Current risk interpretation:** {risk_band}"
)


with st.expander(
    "Complete unified intelligence record",
    expanded=False,
):
    st.json(
        row.to_dict()
    )
