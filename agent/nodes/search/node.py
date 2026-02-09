from __future__ import annotations

import os
import time

from logging_utils import get_logger
from nodes.prompt import FINAL_SYNTHESIS_SYSTEM_PROMPT, build_final_synthesis_prompt
from nodes.state import AgentState

logger = get_logger(__name__)
MIN_SOURCES_FOR_LLM = max(1, int(os.getenv("MIN_SOURCES_FOR_LLM", "4")))


def search_node(state: AgentState) -> dict:
    """
    Final synthesis node.

    Exactly one LLM call is used here for the final report.
    """
    started_at = time.perf_counter()
    errors = list(state.get("errors", []))
    prompt = (state.get("prompt") or "").strip()
    control = state.get("planner_control", {})
    sections = control.get("sections", []) if isinstance(control, dict) else []
    if not isinstance(sections, list):
        sections = []
    logger.info(
        "Search synthesis start. trace_id=%s sources_in=%d sections=%s",
        state.get("trace_id", ""),
        len(state.get("sources", [])),
        sections,
    )

    sources = [item for item in state.get("sources", []) if isinstance(item, dict)]
    evidence = [
        {
            "statement": item.get("summary", ""),
            "url": item.get("url", ""),
            "source": item.get("source", ""),
        }
        for item in sources
    ]

    report = ""
    if len(sources) < MIN_SOURCES_FOR_LLM:
        errors.append(
            f"Insufficient high-confidence sources for synthesis: {len(sources)} < {MIN_SOURCES_FOR_LLM}."
        )
        report = (
            "# Unable to Answer\n\n"
            "Insufficient high-confidence sources were retrieved to generate a reliable report.\n\n"
            f"- Required minimum sources: {MIN_SOURCES_FOR_LLM}\n"
            f"- Retrieved sources: {len(sources)}\n"
        )
        logger.warning(
            "Skipping LLM synthesis due to insufficient sources. sources=%d min_required=%d",
            len(sources),
            MIN_SOURCES_FOR_LLM,
        )
        return {
            "sources": sources,
            "evidence": evidence,
            "report": report,
            "errors": errors,
        }

    try:
        from llm.factory import get_llm

        llm = get_llm()
        llm_prompt = build_final_synthesis_prompt(
            prompt=prompt,
            sections=sections,
            sources=sources,
        )
        response = llm.generate(llm_prompt, system=FINAL_SYNTHESIS_SYSTEM_PROMPT)
        report = (response.text or "").strip()
        if not report:
            errors.append("Final synthesis LLM returned empty output.")
            logger.warning("Final synthesis LLM returned empty output.")
    except Exception as exc:
        errors.append(f"Final synthesis LLM error: {exc}")
        logger.exception("Final synthesis LLM error: %s", exc)
        report = ("# Error Generating Report\n\n. Please try after sometime !")

    logger.info(
        "Search synthesis complete. sources=%d evidence=%d report_len=%d errors=%d elapsed_ms=%.2f",
        len(sources),
        len(evidence),
        len(report),
        len(errors),
        (time.perf_counter() - started_at) * 1000,
    )

    return {
        "sources": sources,
        "evidence": evidence,
        "report": report,
        "errors": errors,
    }
