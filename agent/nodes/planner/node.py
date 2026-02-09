from __future__ import annotations

import re
import time

from logging_utils import get_logger
from nodes.state import AgentState

DEFAULT_SECTIONS = ["definition", "benefits", "risks", "evaluation", "judgment"]
DEFAULT_TOOLS = ["elasticsearch"]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
    return [token for token in tokens if token and token not in STOPWORDS]


def _top_keywords(prompt: str, max_items: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for token in _tokenize(prompt):
        if len(token) < 4:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:max_items]]


def _build_queries(prompt: str, sections: list[str]) -> list[str]:
    cleaned_prompt = " ".join((prompt or "").strip().split())
    queries: list[str] = []
    if cleaned_prompt:
        queries.append(cleaned_prompt)

    for section in sections:
        if cleaned_prompt:
            queries.append(f"{cleaned_prompt} {section}")

    keywords = _top_keywords(cleaned_prompt, max_items=4)
    if keywords:
        queries.append(" ".join(keywords))

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(query.strip())
        if len(deduped) >= 6:
            break
    return deduped


def planner_node(state: AgentState) -> dict:
    """
    Deterministic planner node.

    Emits control data only. No LLM calls, no ranking.
    """
    started_at = time.perf_counter()
    prompt = (state.get("prompt") or "").strip()
    errors = list(state.get("errors", []))
    sections = list(DEFAULT_SECTIONS)
    queries = _build_queries(prompt, sections)
    logger.info(
        "Planner start. trace_id=%s prompt_len=%d",
        state.get("trace_id", ""),
        len(prompt),
    )

    if not prompt:
        errors.append("Planner received empty prompt.")
        logger.warning("Planner received empty prompt.")
    if not queries:
        errors.append("Planner produced no queries.")
        logger.warning("Planner produced no queries.")

    control = {
        "intent": "research",
        "sections": sections,
        "tools": list(DEFAULT_TOOLS),
    }
    logger.info(
        "Planner control emitted. sections=%s queries=%d elapsed_ms=%.2f",
        sections,
        len(queries),
        (time.perf_counter() - started_at) * 1000,
    )

    return {
        "plan": [section.title() for section in sections],
        "tool_decision": list(DEFAULT_TOOLS),
        "planner_control": control,
        "queries": queries,
        "errors": errors,
    }
