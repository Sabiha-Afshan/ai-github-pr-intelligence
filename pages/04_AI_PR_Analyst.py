"""AI PR Analyst Streamlit page."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from src.config.settings import get_settings
from src.rag.deterministic_citation_repair import (
    DeterministicProductionResponse,
    DeterministicRepairProductionGenerator,
)
from src.rag.grounded_generation import GroundedResponseGenerator
from src.utils.logging import get_logger
from src.utils.paths import create_required_directories


settings = get_settings()
logger = get_logger(__name__)

create_required_directories()


PAGE_TITLE = "AI PR Analyst"

PAGE_DESCRIPTION = (
    "Ask questions about pull requests using governed retrieval, "
    "local LLM generation, deterministic citation repair, and "
    "claim-to-evidence validation."
)

DEFAULT_MODEL = "qwen2.5-coder:3b"

EXAMPLE_QUERIES = [
    "What do we know about pull request 5336?",
    "Which pull requests have a low chance of being merged?",
    "Which changes require a human governance review?",
    "Which changes should maintainers review first?",
]

ACTION_LABELS = {
    "answer": "Answer released",
    "abstain_citation_validation": (
        "Answer withheld because citation validation failed"
    ),
    "abstain_groundedness_validation": (
        "Answer withheld because groundedness validation failed"
    ),
    "abstain_no_evidence": (
        "Answer withheld because sufficient evidence was not found"
    ),
    "abstain_out_of_domain": (
        "Question is outside the supported scope"
    ),
}


def _safe_get(
    value: Any,
    attribute: str,
    default: Any = None,
) -> Any:
    """Read an attribute from an object or a key from a dictionary."""

    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(attribute, default)

    return getattr(value, attribute, default)


def _format_boolean(value: bool) -> str:
    """Format a Boolean validation outcome for display."""

    return "Passed" if value else "Failed"


def _format_latency(
    value: float | int | None,
) -> str:
    """Format milliseconds as milliseconds or seconds."""

    if value is None:
        return "Not available"

    numeric_value = float(value)

    if numeric_value >= 1000:
        return f"{numeric_value / 1000:.2f} s"

    return f"{numeric_value:.2f} ms"


def _format_rate(
    value: float | int | None,
) -> str:
    """Format a zero-to-one rate as a percentage."""

    if value is None:
        return "Not executed"

    return f"{float(value) * 100:.1f}%"


@st.cache_resource(show_spinner=False)
def get_generator() -> DeterministicRepairProductionGenerator:
    """Create and cache the governed production generator."""

    base_generator = GroundedResponseGenerator(
        model=DEFAULT_MODEL,
        evidence_top_k=5,
        maximum_evidence_characters=2400,
        temperature=0.0,
        request_timeout_seconds=240,
    )

    return DeterministicRepairProductionGenerator(
        base_generator=base_generator,
        minimum_support_score=0.60,
        minimum_token_coverage=0.45,
    )


def _get_final_response(
    response: DeterministicProductionResponse,
) -> Any:
    return _safe_get(
        response,
        "final_response",
    )


def _get_evidence(
    response: DeterministicProductionResponse,
) -> list[Any]:
    final_response = _get_final_response(
        response
    )

    base_response = _safe_get(
        final_response,
        "base_response",
    )

    evidence = _safe_get(
        base_response,
        "evidence",
        [],
    )

    return list(evidence or [])


def _render_page_header() -> None:
    st.title(PAGE_TITLE)
    st.caption(PAGE_DESCRIPTION)

    st.info(
        "This page is read-only. It analyses stored pull-request "
        "evidence and does not modify GitHub repositories."
    )


def _render_page_sidebar() -> None:
    with st.sidebar:
        st.header("AI PR Analyst")

        st.write(
            f"**Application:** {settings.app_name}"
        )

        st.write(
            f"**Repository:** "
            f"{settings.github_repository_full_name}"
        )

        st.write(
            f"**Environment:** {settings.app_env}"
        )

        st.divider()

        st.subheader("System configuration")

        st.text_input(
            "Local LLM",
            value=DEFAULT_MODEL,
            disabled=True,
        )

        st.text_input(
            "Retrieval",
            value="Hybrid and governed retrieval",
            disabled=True,
        )

        st.text_input(
            "Safety mode",
            value="Fail closed",
            disabled=True,
        )

        st.divider()

        st.subheader("Example questions")

        for example_number, example_query in enumerate(
            EXAMPLE_QUERIES,
            start=1,
        ):
            if st.button(
                example_query,
                key=f"example_query_{example_number}",
                use_container_width=True,
            ):
                st.session_state[
                    "governed_query_input"
                ] = example_query


def _render_status_banner(
    response: DeterministicProductionResponse,
) -> None:
    action = str(
        _safe_get(
            response,
            "action",
            "unknown",
        )
    )

    answer_released = bool(
        _safe_get(
            response,
            "answer_released",
            False,
        )
    )

    label = ACTION_LABELS.get(
        action,
        action.replace("_", " ").title(),
    )

    if action == "answer" and answer_released:
        st.success(label)
    elif action == "abstain_out_of_domain":
        st.info(label)
    else:
        st.warning(label)


def _render_answer(
    response: DeterministicProductionResponse,
) -> None:
    st.subheader("Governed answer")

    answer = str(
        _safe_get(
            response,
            "answer",
            "",
        )
        or ""
    ).strip()

    if answer:
        st.markdown(answer)
    else:
        st.warning(
            "The governed pipeline returned no visible answer."
        )


def _render_summary_metrics(
    response: DeterministicProductionResponse,
    total_latency_ms: float,
) -> None:
    final_response = _get_final_response(
        response
    )

    sentence_validation = _safe_get(
        final_response,
        "sentence_validation",
    )

    claim_validation = _safe_get(
        final_response,
        "claim_validation",
    )

    generation_executed = bool(
        _safe_get(
            response,
            "generation_executed",
            False,
        )
    )

    repair_attempted = bool(
        _safe_get(
            response,
            "repair_attempted",
            False,
        )
    )

    repair_succeeded = bool(
        _safe_get(
            response,
            "repair_succeeded",
            False,
        )
    )

    answer_released = bool(
        _safe_get(
            response,
            "answer_released",
            False,
        )
    )

    citation_coverage = _safe_get(
        sentence_validation,
        "citation_coverage",
    )

    groundedness_rate = _safe_get(
        claim_validation,
        "groundedness_rate",
    )

    unsupported_claim_count = _safe_get(
        claim_validation,
        "unsupported_claim_count",
        0,
    )

    first_row = st.columns(4)

    first_row[0].metric(
        "Generation",
        (
            "Executed"
            if generation_executed
            else "Skipped"
        ),
    )

    first_row[1].metric(
        "Citation coverage",
        _format_rate(
            citation_coverage
        ),
    )

    first_row[2].metric(
        "Groundedness",
        _format_rate(
            groundedness_rate
        ),
    )

    first_row[3].metric(
        "Unsupported claims",
        str(unsupported_claim_count),
    )

    second_row = st.columns(4)

    second_row[0].metric(
        "Repair attempted",
        "Yes" if repair_attempted else "No",
    )

    second_row[1].metric(
        "Repair succeeded",
        "Yes" if repair_succeeded else "No",
    )

    second_row[2].metric(
        "Answer released",
        "Yes" if answer_released else "No",
    )

    second_row[3].metric(
        "Total latency",
        _format_latency(
            total_latency_ms
        ),
    )


def _render_evidence(
    response: DeterministicProductionResponse,
) -> None:
    evidence = _get_evidence(
        response
    )

    st.subheader("Retrieved evidence")

    if not evidence:
        st.info(
            "No pull-request evidence was used for this response."
        )
        return

    for evidence_item in evidence:
        evidence_id = str(
            _safe_get(
                evidence_item,
                "evidence_id",
                "Evidence",
            )
        )

        pr_number = _safe_get(
            evidence_item,
            "pr_number",
        )

        section = str(
            _safe_get(
                evidence_item,
                "section",
                "Evidence",
            )
        )

        rank = _safe_get(
            evidence_item,
            "rank",
        )

        title_parts = [
            evidence_id
        ]

        if pr_number is not None:
            title_parts.append(
                f"PR #{pr_number}"
            )

        if section:
            title_parts.append(
                section
            )

        if rank is not None:
            title_parts.append(
                f"Rank {rank}"
            )

        with st.expander(
            " · ".join(title_parts),
            expanded=False,
        ):
            st.write(
                str(
                    _safe_get(
                        evidence_item,
                        "text",
                        "",
                    )
                )
            )

            metadata = _safe_get(
                evidence_item,
                "metadata",
                {},
            )

            if metadata:
                st.markdown(
                    "**Metadata**"
                )

                st.json(
                    metadata
                )


def _render_validation_details(
    response: DeterministicProductionResponse,
) -> None:
    final_response = _get_final_response(
        response
    )

    sentence_validation = _safe_get(
        final_response,
        "sentence_validation",
    )

    claim_validation = _safe_get(
        final_response,
        "claim_validation",
    )

    with st.expander(
        "Validation and governance details",
        expanded=False,
    ):
        citation_column, groundedness_column = (
            st.columns(2)
        )

        citation_column.write(
            "**Sentence citation validation:** "
            + _format_boolean(
                bool(
                    _safe_get(
                        sentence_validation,
                        "passed",
                        False,
                    )
                )
            )
        )

        citation_column.write(
            "**Citation coverage:** "
            + _format_rate(
                _safe_get(
                    sentence_validation,
                    "citation_coverage",
                )
            )
        )

        citation_column.write(
            "**Invalid citations:** "
            + str(
                _safe_get(
                    sentence_validation,
                    "invalid_citations",
                    [],
                )
            )
        )

        groundedness_column.write(
            "**Claim groundedness validation:** "
            + _format_boolean(
                bool(
                    _safe_get(
                        claim_validation,
                        "passed",
                        False,
                    )
                )
            )
        )

        groundedness_column.write(
            "**Groundedness rate:** "
            + _format_rate(
                _safe_get(
                    claim_validation,
                    "groundedness_rate",
                )
            )
        )

        groundedness_column.write(
            "**Unsupported claims:** "
            + str(
                _safe_get(
                    claim_validation,
                    "unsupported_claim_count",
                    0,
                )
            )
        )

        claim_results = list(
            _safe_get(
                claim_validation,
                "claim_results",
                [],
            )
            or []
        )

        if not claim_results:
            return

        st.divider()
        st.markdown(
            "#### Claim-level results"
        )

        for result_number, claim_result in enumerate(
            claim_results,
            start=1,
        ):
            passed = bool(
                _safe_get(
                    claim_result,
                    "passed",
                    False,
                )
            )

            claim = str(
                _safe_get(
                    claim_result,
                    "claim",
                    "",
                )
            )

            status_text = (
                "Passed"
                if passed
                else "Blocked"
            )

            st.markdown(
                f"**Claim {result_number} — "
                f"{status_text}:** {claim}"
            )

            detail_columns = st.columns(3)

            detail_columns[0].write(
                "Token coverage: "
                + _format_rate(
                    _safe_get(
                        claim_result,
                        "token_coverage",
                    )
                )
            )

            detail_columns[1].write(
                "Entity coverage: "
                + _format_rate(
                    _safe_get(
                        claim_result,
                        "entity_coverage",
                    )
                )
            )

            detail_columns[2].write(
                "Support score: "
                + _format_rate(
                    _safe_get(
                        claim_result,
                        "support_score",
                    )
                )
            )

            failure_reasons = list(
                _safe_get(
                    claim_result,
                    "failure_reasons",
                    [],
                )
                or []
            )

            for reason in failure_reasons:
                st.error(
                    str(reason)
                )


def _render_repair_details(
    response: DeterministicProductionResponse,
) -> None:
    repair_result = _safe_get(
        response,
        "repair_result",
    )

    attempted = bool(
        _safe_get(
            repair_result,
            "attempted",
            False,
        )
    )

    if not attempted:
        return

    with st.expander(
        "Deterministic citation repair",
        expanded=False,
    ):
        succeeded = bool(
            _safe_get(
                repair_result,
                "succeeded",
                False,
            )
        )

        st.write(
            "**Repair succeeded:** "
            + (
                "Yes"
                if succeeded
                else "No"
            )
        )

        st.write(
            "**Repair latency:** "
            + _format_latency(
                _safe_get(
                    repair_result,
                    "latency_ms",
                    0.0,
                )
            )
        )

        st.write(
            "**Sentences requiring repair:** "
            + str(
                _safe_get(
                    repair_result,
                    "sentences_requiring_repair",
                    0,
                )
            )
        )

        st.write(
            "**Sentences repaired:** "
            + str(
                _safe_get(
                    repair_result,
                    "sentences_repaired",
                    0,
                )
            )
        )

        original_answer = str(
            _safe_get(
                repair_result,
                "original_answer",
                "",
            )
        )

        repaired_answer = str(
            _safe_get(
                repair_result,
                "repaired_answer",
                "",
            )
        )

        st.markdown(
            "#### Original model answer"
        )

        st.code(
            original_answer,
            language=None,
        )

        st.markdown(
            "#### Repaired answer"
        )

        st.code(
            repaired_answer,
            language=None,
        )

        decisions = list(
            _safe_get(
                repair_result,
                "decisions",
                [],
            )
            or []
        )

        relevant_decisions = [
            decision
            for decision in decisions
            if bool(
                _safe_get(
                    decision,
                    "repair_required",
                    False,
                )
            )
        ]

        if not relevant_decisions:
            return

        st.markdown(
            "#### Repair decisions"
        )

        for decision in relevant_decisions:
            st.write(
                "**Sentence:** "
                + str(
                    _safe_get(
                        decision,
                        "original_sentence",
                        "",
                    )
                )
            )

            st.write(
                "**Candidate evidence:** "
                + str(
                    _safe_get(
                        decision,
                        "candidate_evidence_ids",
                        [],
                    )
                )
            )

            st.write(
                "**Selected evidence:** "
                + str(
                    _safe_get(
                        decision,
                        "selected_evidence_id",
                        None,
                    )
                )
            )

            st.write(
                "**Reason:** "
                + str(
                    _safe_get(
                        decision,
                        "reason",
                        "",
                    )
                )
            )

            st.divider()


def _render_technical_trace(
    response: DeterministicProductionResponse,
) -> None:
    with st.expander(
        "Technical trace",
        expanded=False,
    ):
        try:
            payload = response.to_dict()
        except Exception:
            payload = {
                "action": _safe_get(
                    response,
                    "action",
                ),
                "answer": _safe_get(
                    response,
                    "answer",
                ),
                "trace": _safe_get(
                    response,
                    "trace",
                    {},
                ),
            }

        st.json(
            payload
        )


def _run_query(
    query: str,
) -> tuple[
    DeterministicProductionResponse | None,
    float,
    str | None,
]:
    started_at = time.perf_counter()

    try:
        generator = get_generator()

        response = generator.generate(
            query=query
        )

        total_latency_ms = (
            time.perf_counter()
            - started_at
        ) * 1000

        return (
            response,
            total_latency_ms,
            None,
        )

    except Exception as error:
        logger.exception(
            "AI PR Analyst request failed."
        )

        total_latency_ms = (
            time.perf_counter()
            - started_at
        ) * 1000

        error_message = (
            f"{type(error).__name__}: {error}"
        )

        return (
            None,
            total_latency_ms,
            error_message,
        )


def main() -> None:
    logger.info(
        "AI PR Analyst page loaded."
    )

    _render_page_header()
    _render_page_sidebar()

    st.subheader(
        "Ask the PR intelligence system"
    )

    query = st.text_area(
        "Question",
        key="governed_query_input",
        placeholder=(
            "Ask about a specific PR, merge probability, "
            "governance review, policy risk, or review priority."
        ),
        height=110,
    )

    run_button = st.button(
        "Run governed analysis",
        type="primary",
        use_container_width=True,
    )

    if not run_button:
        st.caption(
            "The first request may take longer while the "
            "embedding model and local LLM are loaded."
        )
        return

    cleaned_query = query.strip()

    if not cleaned_query:
        st.warning(
            "Enter a question before running the analysis."
        )
        return

    with st.spinner(
        "Retrieving evidence, generating the answer, "
        "and running governance checks..."
    ):
        (
            response,
            total_latency_ms,
            error_message,
        ) = _run_query(
            cleaned_query
        )

    if error_message is not None:
        st.error(
            "The governed pipeline could not complete the request."
        )

        st.code(
            error_message,
            language=None,
        )

        st.write(
            "Confirm that Ollama is running and that "
            f"`{DEFAULT_MODEL}` is installed."
        )

        st.metric(
            "Elapsed time before failure",
            _format_latency(
                total_latency_ms
            ),
        )

        return

    if response is None:
        st.error(
            "The governed pipeline returned no response."
        )
        return

    _render_status_banner(
        response
    )

    _render_answer(
        response
    )

    _render_summary_metrics(
        response=response,
        total_latency_ms=total_latency_ms,
    )

    st.divider()

    _render_evidence(
        response
    )

    _render_validation_details(
        response
    )

    _render_repair_details(
        response
    )

    _render_technical_trace(
        response
    )


main()
