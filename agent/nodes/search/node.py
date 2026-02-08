from __future__ import annotations

import time

from logging_utils import get_logger
from nodes.prompt import FINAL_SYNTHESIS_SYSTEM_PROMPT, build_final_synthesis_prompt
from nodes.state import AgentState

logger = get_logger(__name__)


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in sources:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("source", "")),
            str(item.get("url", "")),
            str(item.get("title", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


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

    sources = _dedupe_sources(state.get("sources", []))
    evidence = [
        {
            "statement": item.get("summary", ""),
            "url": item.get("url", ""),
            "source": item.get("source", ""),
        }
        for item in sources
    ]

    report = ""
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
