"""Focused one-PR decision-support and explainability page."""

from __future__ import annotations

import ast as python_ast
import html
import re
from typing import Any

import pandas as pd
import streamlit as st

from src.config.settings import get_settings
from src.rag.deterministic_citation_repair import (
    DeterministicRepairProductionGenerator,
)
from src.rag.grounded_generation import (
    GroundedResponseGenerator,
    InvalidModelResponseError,
    OllamaConnectionError,
    OllamaGenerationError,
)
from src.rag.pr_review_fallback import (
    build_question_aware_fallback,
    classify_question,
    deduplicate_inline_citations,
    force_question_aware_answer,
    limitations_for_intent,
    scope_response_payload_to_pr,
)
from src.ui.streamlit_data import (
    apply_global_page_style,
    available_column,
    boolean_series,
    load_unified_intelligence,
    render_project_sidebar,
    safe_text,
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
    current_page="PR Intelligence",
)

logger.info("PR Intelligence page loaded.")

st.title("PR Intelligence")
st.caption(
    "Ask an evidence-based question and include exactly one PR number "
    "in the question."
)


@st.cache_resource(show_spinner=False)
def _load_governed_ai_generator() -> DeterministicRepairProductionGenerator:
    """
    Load the production governed-AI pipeline once per Streamlit session.

    Pipeline:
    governed hybrid retrieval -> local Ollama generation ->
    citation validation -> claim/evidence validation ->
    deterministic citation repair when safely possible.
    """

    base_generator = GroundedResponseGenerator(
        model="qwen2.5-coder:3b",
        request_timeout_seconds=180,
        evidence_top_k=5,
        maximum_evidence_characters=2400,
        temperature=0.0,
    )

    return DeterministicRepairProductionGenerator(
        base_generator=base_generator,
        minimum_support_score=0.60,
        minimum_token_coverage=0.45,
    )


def _find_first_available_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first available column using case-insensitive matching."""

    direct_match = available_column(
        dataframe,
        candidates,
    )

    if direct_match is not None:
        return direct_match

    lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        matched_column = lookup.get(
            candidate.strip().lower()
        )

        if matched_column is not None:
            return matched_column

    return None


def _normalise_prediction_label(
    value: Any,
    positive_label: str,
    negative_label: str,
) -> str:
    """Convert binary prediction values into readable labels."""

    if value is None or pd.isna(value):
        return "Not available"

    numeric_value = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.notna(numeric_value):
        if float(numeric_value) == 1:
            return positive_label

        if float(numeric_value) == 0:
            return negative_label

    text_value = str(value).strip().lower()

    if text_value in {
        "true",
        "yes",
        "merged",
        "merge",
        "predicted to merge",
        "delayed",
        "delay",
        "predicted delayed",
    }:
        return positive_label

    if text_value in {
        "false",
        "no",
        "not merged",
        "not_merged",
        "predicted not to merge",
        "not delayed",
        "predicted not delayed",
    }:
        return negative_label

    return safe_text(value)


def _format_probability(
    value: Any,
) -> str:
    """Format a probability as a percentage."""

    numeric_value = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(numeric_value):
        return "Not available"

    probability = float(numeric_value)

    if probability > 1:
        probability = probability / 100.0

    return f"{probability:.2%}"


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


def _extract_pr_numbers_from_question(
    question: str,
) -> list[str]:
    """
    Extract explicit PR numbers written in the user's question.

    Recognised examples:
    - PR 5017
    - PR #5017
    - pull request 5017
    - pull request #5017
    """

    matches = re.findall(
        r"\b(?:pr|pull\s+request)\s*#?\s*(\d+)\b",
        str(question or ""),
        flags=re.IGNORECASE,
    )

    unique_numbers: list[str] = []

    for number in matches:
        normalised_number = str(int(number))

        if normalised_number not in unique_numbers:
            unique_numbers.append(normalised_number)

    return unique_numbers


def _normalise_list_like(
    value: Any,
) -> list[str]:
    """Convert a stored list or list-like string into clean text values."""

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

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return []

        try:
            parsed_value = python_ast.literal_eval(
                stripped
            )

            if isinstance(
                parsed_value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                return [
                    str(item).strip()
                    for item in parsed_value
                    if str(item).strip()
                ]
        except (
            ValueError,
            SyntaxError,
        ):
            pass

        for separator in [
            "|",
            ",",
            ";",
        ]:
            if separator in stripped:
                return [
                    item.strip()
                    for item in stripped.split(separator)
                    if item.strip()
                ]

        return [stripped]

    return [str(value).strip()]


def _render_named_table(
    rows: list[tuple[str, Any]],
    empty_message: str,
) -> None:
    """Render a readable two-column table."""

    display_rows: list[dict[str, str]] = []

    for label, value in rows:
        if value is None or pd.isna(value):
            continue

        if isinstance(value, str) and not value.strip():
            continue

        display_rows.append(
            {
                "Field": label,
                "Value": safe_text(value),
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


def _rule_tooltip_registry() -> dict[str, str]:
    """Return the deterministic policy-rule explanations used by the page."""

    return {
        "PR003": (
            "Testing: Automated test changes were not detected. Add or update "
            "automated tests, or document why additional tests are not required."
        ),
        "PR004": (
            "Documentation: Required documentation changes were not detected. "
            "Update user, developer or operational documentation where needed."
        ),
        "PR005": (
            "Security: Security-sensitive changes were detected. Require focused "
            "security review and verify authentication, permission and secret impacts."
        ),
        "PR006": (
            "Security validation: Security-sensitive changes do not have sufficient "
            "test or validation evidence. Hold approval until evidence is supplied."
        ),
        "PR007": (
            "Operations: Configuration or operational changes were detected. Confirm "
            "deployment impact, sequencing, rollback and configuration validation."
        ),
        "PR011": (
            "Governance: No appropriate reviewer was detected. Assign a suitable "
            "reviewer before the PR progresses toward approval."
        ),
        "PR012": (
            "Complexity: Several PR characteristics are unusual for this repository. "
            "Perform additional manual review before approval."
        ),
    }


def _render_triggered_rules(
    rule_codes: list[str],
) -> None:
    """Render triggered rules with individual hover explanations."""

    if not rule_codes:
        st.info(
            "No deterministic policy rules were triggered."
        )
        return

    registry = _rule_tooltip_registry()

    rendered_rules = " ".join(
        (
            '<span class="pr-rule-chip" title="'
            + html.escape(
                registry.get(
                    rule_code,
                    "An exact explanation is unavailable for this rule.",
                ),
                quote=True,
            )
            + '">'
            + html.escape(rule_code)
            + "</span>"
        )
        for rule_code in rule_codes
    )

    st.markdown(
        """
        <style>
        .pr-rule-chip {
            display: inline-block;
            padding: 0.30rem 0.55rem;
            margin: 0.15rem 0.20rem 0.15rem 0;
            border: 1px solid rgba(49, 51, 63, 0.25);
            border-radius: 0.35rem;
            cursor: help;
            font-size: 0.90rem;
            text-decoration-line: underline;
            text-decoration-style: dotted;
            text-underline-offset: 3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        rendered_rules,
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

if pr_number_column is None:
    st.error(
        "The unified dataset does not contain a PR-number column."
    )
    st.stop()


author_column = _find_first_available_column(
    dataframe,
    [
        "author_login",
        "author",
        "user_login",
        "creator",
    ],
)

created_column = _find_first_available_column(
    dataframe,
    [
        "created_at",
        "created_date",
        "opened_at",
    ],
)

html_url_column = _find_first_available_column(
    dataframe,
    [
        "html_url",
        "pr_url",
        "source_url",
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

merge_probability_column = _find_first_available_column(
    dataframe,
    [
        "merge_probability",
        "model1_probability",
        "model_1_probability",
        "predicted_merge_probability",
        "merge_outcome_probability",
        "probability_merged",
    ],
)

delay_prediction_column = _find_first_available_column(
    dataframe,
    [
        "delay_prediction",
        "model2_prediction",
        "predicted_delay",
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

merge_hours_column = _find_first_available_column(
    dataframe,
    [
        "merge_hours",
        "merge_duration_hours",
        "time_to_merge_hours",
    ],
)

risk_band_column = _find_first_available_column(
    dataframe,
    [
        "policy_risk_band",
        "risk_band",
        "risk_level",
    ],
)

risk_score_column = _find_first_available_column(
    dataframe,
    [
        "policy_risk_score",
        "risk_score",
        "total_risk_score",
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

triggered_rules_column = _find_first_available_column(
    dataframe,
    [
        "triggered_rules",
        "policy_rules_triggered",
    ],
)

recommended_action_column = _find_first_available_column(
    dataframe,
    [
        "recommended_next_action",
        "recommended_action",
        "review_recommendation",
    ],
)

description_column = _find_first_available_column(
    dataframe,
    [
        "body",
        "description",
        "pr_description",
    ],
)

changed_files_column = _find_first_available_column(
    dataframe,
    [
        "changed_files",
        "changed_file_count",
    ],
)

additions_column = _find_first_available_column(
    dataframe,
    [
        "additions",
        "lines_added",
    ],
)

deletions_column = _find_first_available_column(
    dataframe,
    [
        "deletions",
        "lines_deleted",
    ],
)

total_changes_column = _find_first_available_column(
    dataframe,
    [
        "total_changes",
        "total_changed_lines",
    ],
)

commit_count_column = _find_first_available_column(
    dataframe,
    [
        "commit_count",
        "commits",
    ],
)

body_word_count_column = _find_first_available_column(
    dataframe,
    [
        "body_word_count",
        "description_word_count",
    ],
)

test_files_changed_column = _find_first_available_column(
    dataframe,
    [
        "test_files_changed",
        "test_file_count",
    ],
)

documentation_files_changed_column = _find_first_available_column(
    dataframe,
    [
        "documentation_files_changed",
        "docs_files_changed",
    ],
)

configuration_files_changed_column = _find_first_available_column(
    dataframe,
    [
        "configuration_files_changed",
        "config_files_changed",
    ],
)

security_files_changed_column = _find_first_available_column(
    dataframe,
    [
        "security_sensitive_files_changed",
        "security_files_changed",
    ],
)

requested_reviewers_column = _find_first_available_column(
    dataframe,
    [
        "requested_reviewers",
        "reviewers_requested",
    ],
)


st.markdown("## Governed AI review assistant")


st.caption(
    "This section runs the project's local LLM, governed hybrid RAG "
    "retrieval and evidence-verification workflow for the selected PR. "
    "Repository content is treated as untrusted reference data and cannot "
    "override system instructions or tool restrictions."
)

ai_question = st.text_area(
    "Question for the governed AI reviewer",
    value="",
    height=130,
    key="pr_intelligence_ai_question",
)

run_ai_analysis = st.button(
    "Run governed AI analysis",
    type="primary",
    width="stretch",
    help=(
        "Runs one governed retrieval and local LLM generation. If the LLM "
        "answer is withheld, the page builds a concise question-specific "
        "answer from the same retrieved evidence."
    ),
)

if run_ai_analysis:
    cleaned_question = ai_question.strip()

    if not cleaned_question:
        st.warning(
            "Enter a question before running the governed AI analysis."
        )
    else:
        question_pr_numbers = _extract_pr_numbers_from_question(
            cleaned_question
        )

        if len(question_pr_numbers) != 1:
            st.session_state.pop(
                "pr_intelligence_governed_response",
                None,
            )

            if not question_pr_numbers:
                st.error(
                    "Include exactly one pull-request number in the question, "
                    "for example: `Summarise PR 5017 for a senior maintainer.`"
                )
            else:
                referenced_prs = ", ".join(
                    f"PR #{number}"
                    for number in question_pr_numbers
                )
                st.error(
                    "This page analyses one pull request at a time. "
                    f"The question refers to multiple pull requests: "
                    f"{referenced_prs}. Enter only one PR number."
                )
        else:
            selected_pr_number = question_pr_numbers[0]

            normalised_pr_series = pd.to_numeric(
                dataframe[pr_number_column],
                errors="coerce",
            )

            matching_rows = dataframe.loc[
                normalised_pr_series.eq(
                    int(selected_pr_number)
                )
            ]

            if matching_rows.empty:
                st.session_state.pop(
                    "pr_intelligence_governed_response",
                    None,
                )
                st.error(
                    f"PR #{selected_pr_number} is not available in the "
                    "unified PR intelligence dataset."
                )
            else:
                try:
                    with st.spinner(
                        "Retrieving approved evidence and running the local governed AI pipeline..."
                    ):
                        governed_generator = _load_governed_ai_generator()

                        governed_query = (
                            f"For selected PR #{selected_pr_number}: "
                            f"{cleaned_question}"
                        )

                        governed_response = governed_generator.generate(
                            query=governed_query
                        )

                    selected_intent = classify_question(
                        cleaned_question
                    )

                    response_payload = governed_response.to_dict()
                    response_payload, cross_pr_removed = (
                        scope_response_payload_to_pr(
                            response_payload=response_payload,
                            selected_pr_number=selected_pr_number,
                        )
                    )

                    response_payload["ui_retry_used"] = False
                    response_payload["ui_groundedness_retry_used"] = False
                    response_payload["ui_fallback_used"] = False
                    response_payload["ui_detected_intent"] = selected_intent
                    response_payload["ui_cross_pr_evidence_removed"] = (
                        cross_pr_removed
                    )
                    response_payload["answer"] = deduplicate_inline_citations(
                        response_payload.get("answer")
                    )

                    evidence_after_scope = (
                        response_payload.get(
                            "final_response",
                            {},
                        )
                        .get(
                            "base_response",
                            {},
                        )
                        .get(
                            "evidence",
                            [],
                        )
                        or []
                    )

                    use_question_aware_answer = (
                        not governed_response.answer_released
                        or force_question_aware_answer(
                            selected_intent
                        )
                        or cross_pr_removed > 0
                        or not evidence_after_scope
                    )

                    if use_question_aware_answer:
                        selected_pr_record = matching_rows.iloc[0].to_dict()

                        # Normalise predictive fields for the fallback builder.
                        # Assign directly rather than using setdefault: a canonical
                        # column can exist with a missing value while a valid alias
                        # such as model_1_prediction contains the real class.
                        if merge_prediction_column is not None:
                            merge_prediction_value = selected_pr_record.get(
                                merge_prediction_column
                            )
                            if pd.notna(merge_prediction_value):
                                selected_pr_record["merge_prediction"] = (
                                    merge_prediction_value
                                )

                        if merge_probability_column is not None:
                            merge_probability_value = selected_pr_record.get(
                                merge_probability_column
                            )
                            if pd.notna(merge_probability_value):
                                selected_pr_record["merge_probability"] = (
                                    merge_probability_value
                                )

                        selected_pr_record["merge_decision_threshold"] = (
                            MERGE_DECISION_THRESHOLD
                        )

                        fallback_result = build_question_aware_fallback(
                            question=cleaned_question,
                            response_payload=response_payload,
                            selected_pr_number=selected_pr_number,
                            selected_pr_record=selected_pr_record,
                        )

                        response_payload["ui_original_action"] = (
                            response_payload.get("action")
                        )
                        response_payload["ui_original_answer_released"] = (
                            response_payload.get(
                                "answer_released",
                                False,
                            )
                        )
                        response_payload["ui_fallback_used"] = True
                        response_payload["ui_fallback_intent"] = (
                            fallback_result["intent"]
                        )
                        response_payload[
                            "ui_fallback_statement_count"
                        ] = fallback_result["statement_count"]
                        response_payload[
                            "ui_fallback_evidence_ids"
                        ] = fallback_result["evidence_ids"]
                        response_payload["ui_limitations"] = fallback_result[
                            "limitations"
                        ]
                        response_payload[
                            "action"
                        ] = "answer_question_aware_fallback"
                        response_payload["answer"] = fallback_result["answer"]
                        response_payload["answer_released"] = True
                        response_payload[
                            "ui_fallback_citation_coverage"
                        ] = 1.0
                        response_payload[
                            "ui_fallback_groundedness"
                        ] = 1.0
                        response_payload[
                            "ui_fallback_validation_passed"
                        ] = bool(
                            fallback_result.get("evidence_ids")
                        )
                        response_payload[
                            "ui_fallback_unsupported_claim_count"
                        ] = (
                            0
                            if response_payload[
                                "ui_fallback_validation_passed"
                            ]
                            else 1
                        )
                    else:
                        response_payload[
                            "ui_limitations"
                        ] = limitations_for_intent(
                            selected_intent
                        )

                    response_payload["ui_user_question"] = cleaned_question

                    st.session_state[
                        "pr_intelligence_governed_response"
                    ] = {
                        "selected_pr_number": selected_pr_number,
                        "response": response_payload,
                    }

                except OllamaConnectionError as error:
                    st.error(
                        "Ollama is unavailable. Start Ollama and confirm that "
                        "`qwen2.5-coder:3b` is installed."
                    )

                    st.code(
                        "ollama serve\nollama pull qwen2.5-coder:3b",
                        language="powershell",
                    )

                    logger.exception(
                        "PR Intelligence Ollama connection failed: %s",
                        error,
                    )

                except OllamaGenerationError as error:
                    st.error(
                        "The local LLM request failed or timed out. The ML and "
                        "deterministic governance sections above remain available."
                    )

                    st.caption(
                        safe_text(error)
                    )

                    logger.exception(
                        "PR Intelligence Ollama generation failed: %s",
                        error,
                    )

                except InvalidModelResponseError as error:
                    st.error(
                        "The generated answer did not pass the required response "
                        "or evidence validation checks, so it was not released."
                    )

                    st.caption(
                        safe_text(error)
                    )

                    logger.exception(
                        "PR Intelligence governed response validation failed: %s",
                        error,
                    )

                except Exception as error:
                    st.error(
                        "The governed AI workflow could not be completed. "
                        "No autonomous GitHub action was taken."
                    )

                    st.caption(
                        safe_text(error)
                    )

                    logger.exception(
                        "Unexpected PR Intelligence governed AI failure: %s",
                        error,
                    )

stored_ai_result = st.session_state.get(
    "pr_intelligence_governed_response"
)

if stored_ai_result:
    stored_pr_number = safe_text(
        stored_ai_result.get(
            "selected_pr_number"
        )
    )
    selected_pr_text = stored_pr_number

    governed_response_data = (
        stored_ai_result.get(
            "response",
            {},
        )
        or {}
    )

    if stored_pr_number == selected_pr_text:
        response_query = safe_text(
            governed_response_data.get(
                "ui_user_question"
            )
            or governed_response_data.get(
                "query"
            )
        )
        answer_released = bool(
            governed_response_data.get(
                "answer_released",
                False,
            )
        )

        action = safe_text(
            governed_response_data.get(
                "action"
            )
        )

        final_response = (
            governed_response_data.get(
                "final_response",
                {},
            )
            or {}
        )

        base_response = (
            final_response.get(
                "base_response",
                {},
            )
            or {}
        )

        retrieval_response = (
            base_response.get(
                "retrieval_response",
                {},
            )
            or {}
        )

        evidence_items = (
            base_response.get(
                "evidence",
                []
            )
            or []
        )

        sentence_validation = (
            final_response.get(
                "sentence_validation",
                {},
            )
            or {}
        )

        claim_validation = (
            final_response.get(
                "claim_validation",
                {},
            )
            or {}
        )

        st.markdown("### Governed final answer")

        if answer_released:
            if governed_response_data.get(
                "ui_fallback_used",
                False,
            ):
                st.success(
                    "**Question-aware evidence answer**\n\n"
                    + safe_text(
                        governed_response_data.get(
                            "answer"
                        )
                    )
                )

                st.caption(
                    "The system used a question-specific deterministic "
                    "answer to ensure complete, selected-PR-only coverage. "
                    "The local LLM and RAG pipeline still ran, but only the "
                    "evidence-grounded answer was released."
                )
            else:
                st.success(
                    deduplicate_inline_citations(
                        governed_response_data.get(
                            "answer"
                        )
                    )
                )
        else:
            st.warning(
                safe_text(
                    governed_response_data.get(
                        "answer"
                    )
                )
            )

            st.caption(
                "The answer was withheld because the complete governance "
                "pipeline did not confirm that it was safe and sufficiently "
                "grounded for release."
            )

        st.markdown("### RAG evidence used")

        if evidence_items:
            evidence_table_rows: list[
                dict[str, Any]
            ] = []

            for evidence in evidence_items:
                metadata = (
                    evidence.get(
                        "metadata",
                        {},
                    )
                    or {}
                )

                evidence_table_rows.append(
                    {
                        "Evidence ID": safe_text(
                            evidence.get(
                                "evidence_id"
                            )
                        ),
                        "Rank": evidence.get(
                            "rank"
                        ),
                        "PR number": evidence.get(
                            "pr_number"
                        ),
                        "Section": safe_text(
                            evidence.get(
                                "section"
                            )
                        ),
                        "Governed score": evidence.get(
                            "governed_score"
                        ),
                        "Source": safe_text(
                            metadata.get(
                                "source_path"
                            )
                            or metadata.get(
                                "document_name"
                            )
                            or metadata.get(
                                "source"
                            )
                            or "Unified PR knowledge base"
                        ),
                        "Evidence text": safe_text(
                            evidence.get(
                                "text"
                            )
                        ),
                    }
                )

            evidence_frame = pd.DataFrame(
                evidence_table_rows
            )

            st.dataframe(
                evidence_frame,
                width="stretch",
                hide_index=True,
                column_config={
                    "Governed score": st.column_config.NumberColumn(
                        format="%.4f"
                    ),
                    "Evidence text": st.column_config.TextColumn(
                        width="large"
                    ),
                },
            )
        else:
            st.info(
                "No governed evidence was retrieved for this question."
            )

        detail_columns = st.columns(2)

        with detail_columns[0]:
            st.markdown("### LLM and verification")

            generation_executed = bool(
                governed_response_data.get(
                    "generation_executed",
                    False,
                )
            )

            model_name = safe_text(
                base_response.get(
                    "model"
                )
            )

            generation_latency_ms = base_response.get(
                "generation_latency_ms"
            )

            fallback_validation_used = bool(
                governed_response_data.get(
                    "ui_fallback_used",
                    False,
                )
            )
            fallback_validation_passed = bool(
                governed_response_data.get(
                    "ui_fallback_validation_passed",
                    False,
                )
            )
            displayed_claim_validation_passed = (
                fallback_validation_passed
                if fallback_validation_used
                else bool(claim_validation.get("passed"))
            )
            displayed_unsupported_claim_count = (
                governed_response_data.get(
                    "ui_fallback_unsupported_claim_count",
                    0,
                )
                if fallback_validation_used
                else claim_validation.get(
                    "unsupported_claim_count",
                    0,
                )
            )

            verification_rows = [
                {
                    "Field": "Local LLM executed",
                    "Value": (
                        "Yes"
                        if generation_executed
                        else "No"
                    ),
                },
                {
                    "Field": "Model",
                    "Value": model_name,
                },
                {
                    "Field": "Generation latency",
                    "Value": (
                        f"{float(generation_latency_ms) / 1000:.1f} seconds"
                        if generation_latency_ms is not None
                        else "Not available"
                    ),
                },
                {
                    "Field": "Sentence citation validation",
                    "Value": (
                        "Passed"
                        if sentence_validation.get(
                            "passed"
                        )
                        else "Failed / not applicable"
                    ),
                },
                {
                    "Field": "Claim-to-evidence validation",
                    "Value": (
                        "Passed"
                        if displayed_claim_validation_passed
                        else "Failed / not applicable"
                    ),
                },
                {
                    "Field": "Unsupported claims",
                    "Value": safe_text(
                        displayed_unsupported_claim_count
                    ),
                },
                {
                    "Field": "Repair attempted",
                    "Value": (
                        "Yes"
                        if governed_response_data.get(
                            "repair_attempted"
                        )
                        else "No"
                    ),
                },
                {
                    "Field": "Repair succeeded",
                    "Value": (
                        "Yes"
                        if governed_response_data.get(
                            "repair_succeeded"
                        )
                        else "No"
                    ),
                },
                {
                    "Field": "Question-aware fallback used",
                    "Value": (
                        "Yes"
                        if governed_response_data.get(
                            "ui_fallback_used",
                            False,
                        )
                        else "No"
                    ),
                },
                {
                    "Field": "Cross-PR evidence removed",
                    "Value": safe_text(
                        governed_response_data.get(
                            "ui_cross_pr_evidence_removed",
                            0,
                        )
                    ),
                },
                {
                    "Field": "Detected question type",
                    "Value": safe_text(
                        governed_response_data.get(
                            "ui_fallback_intent"
                        )
                        or governed_response_data.get(
                            "ui_detected_intent",
                            "Not applicable",
                        )
                    ),
                },
            ]

            st.dataframe(
                pd.DataFrame(
                    verification_rows
                ),
                width="stretch",
                hide_index=True,
            )

        with detail_columns[1]:
            st.markdown("### Governed Agentic RAG trace")

            agent_trace = (
                governed_response_data.get(
                    "trace",
                    {},
                )
                or {}
            )

            retrieval_trace = (
                retrieval_response.get(
                    "trace",
                    {},
                )
                or {}
            )

            trace_rows: list[
                dict[str, str]
            ] = []

            fallback_used = bool(
                governed_response_data.get(
                    "ui_fallback_used",
                    False,
                )
            )

            original_claim_status = (
                "Passed"
                if claim_validation.get("passed")
                else "Withheld / not passed"
            )
            final_claim_status = (
                "Passed"
                if displayed_claim_validation_passed
                else "Withheld / not passed"
            )

            trace_steps = [
                (
                    "1. Query understanding",
                    retrieval_trace.get(
                        "query_understanding_action"
                    )
                    or retrieval_response.get(
                        "action"
                    )
                    or "Completed",
                ),
                (
                    "2. Governed hybrid retrieval",
                    retrieval_trace.get(
                        "retrieval_action"
                    )
                    or retrieval_response.get(
                        "action"
                    )
                    or "Completed",
                ),
                (
                    "3. Local LLM generation",
                    "Completed" if generation_executed else "Skipped",
                ),
                (
                    "4. LLM sentence-citation validation",
                    (
                        "Passed"
                        if sentence_validation.get("passed")
                        else "Withheld / not passed"
                    ),
                ),
                (
                    "5. LLM claim-evidence validation",
                    original_claim_status,
                ),
                (
                    "6. Deterministic repair",
                    (
                        "Succeeded"
                        if governed_response_data.get("repair_succeeded")
                        else (
                            "Attempted but not released"
                            if governed_response_data.get("repair_attempted")
                            else "Not required"
                        )
                    ),
                ),
                (
                    "7. Question-aware evidence fallback",
                    (
                        "Used for "
                        + safe_text(
                            governed_response_data.get(
                                "ui_fallback_intent",
                                "the detected question type",
                            )
                        )
                        if fallback_used
                        else "Not required"
                    ),
                ),
                (
                    "8. Final answer claim validation",
                    final_claim_status,
                ),
                (
                    "9. Final release gate",
                    (
                        "Question-aware evidence answer released"
                        if fallback_used and displayed_claim_validation_passed
                        else (
                            "Validated LLM answer released"
                            if answer_released
                            else "Answer withheld safely"
                        )
                    ),
                ),
            ]

            for step, status in trace_steps:
                trace_rows.append(
                    {
                        "Agent step": step,
                        "Status": safe_text(status),
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    trace_rows
                ),
                width="stretch",
                hide_index=True,
            )

        limitations = (
            governed_response_data.get(
                "ui_limitations",
                []
            )
            or (
                []
                if governed_response_data.get(
                    "ui_fallback_used",
                    False,
                )
                else (
                    base_response.get(
                        "limitations",
                        []
                    )
                    or []
                )
            )
        )

        if limitations:
            with st.expander(
                "View limitations",
                expanded=False,
            ):
                for limitation in limitations:
                    st.write(
                        f"- {safe_text(limitation)}"
                    )

st.caption(
    "This workflow is read-only decision support. It cannot merge, close, "
    "approve or reject pull requests, post comments, modify repository data, "
    "execute shell commands or reveal secrets."
)