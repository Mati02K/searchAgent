from __future__ import annotations

import re
import time

from logging_utils import get_logger
from nodes.state import AgentState

DEFAULT_SECTIONS = ["definition", "benefits", "risks", "evaluation", "judgment"]
DEFAULT_TOOLS = ["elasticsearch"]
DEFAULT_TOPIC_INDEX_KEY = "general"
TOPIC_INDEX_KEYS = {"llm", "networking", "general"}
LLM_TEXT_MUST_INCLUDE = ["llm", "large language model", "text"]
LLM_TEXT_MUST_EXCLUDE = ["vision", "video", "tabular", "gesture"]
NETWORKING_MUST_INCLUDE = ["network", "routing", "latency", "bandwidth", "throughput", "tcp", "ip"]
NETWORKING_MUST_EXCLUDE = ["llm", "large language model", "synthetic data"]
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


def _build_relevance_rules(prompt: str) -> dict[str, list[str]]:
    lowered = (prompt or "").lower()
    if "llm" in lowered or "language model" in lowered or "text" in lowered:
        return {
            "must_include": list(LLM_TEXT_MUST_INCLUDE),
            "must_exclude": list(LLM_TEXT_MUST_EXCLUDE),
        }
    if any(
        keyword in lowered
        for keyword in (
            "network",
            "routing",
            "tcp",
            "ip",
            "bandwidth",
            "throughput",
            "packet",
            "latency",
            "dns",
            "http",
        )
    ):
        return {
            "must_include": list(NETWORKING_MUST_INCLUDE),
            "must_exclude": list(NETWORKING_MUST_EXCLUDE),
        }

    keywords = _top_keywords(prompt, max_items=3)
    return {
        "must_include": keywords,
        "must_exclude": [],
    }


def _choose_topic_index_key(prompt: str, relevance_rules: dict[str, list[str]]) -> str:
    lowered = (prompt or "").lower()
    if "llm" in lowered or "language model" in lowered or "synthetic data" in lowered:
        return "llm"
    if any(
        keyword in lowered
        for keyword in (
            "network",
            "routing",
            "tcp",
            "ip",
            "bandwidth",
            "throughput",
            "packet",
            "latency",
            "dns",
            "http",
        )
    ):
        return "networking"
    includes = {item.lower() for item in relevance_rules.get("must_include", [])}
    if {"llm", "large language model"} & includes:
        return "llm"
    if {"network", "routing", "tcp", "ip", "latency"} & includes:
        return "networking"
    return DEFAULT_TOPIC_INDEX_KEY


def planner_node(state: AgentState) -> dict:
    """
    Deterministic planner node.

    Emits control data only. No LLM calls, no ranking.
    """
    started_at = time.perf_counter()
    prompt = (state.get("prompt") or "").strip()
    errors = list(state.get("errors", []))
    sections = list(DEFAULT_SECTIONS)
    relevance_rules = _build_relevance_rules(prompt)
    topic_index_key = _choose_topic_index_key(prompt, relevance_rules)
    if topic_index_key not in TOPIC_INDEX_KEYS:
        topic_index_key = DEFAULT_TOPIC_INDEX_KEY
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
        "relevance_rules": relevance_rules,
        "target_index_key": topic_index_key,
    }
    logger.info(
        "Planner control emitted. target_index_key=%s sections=%s queries=%d must_include=%s must_exclude=%s elapsed_ms=%.2f",
        topic_index_key,
        sections,
        len(queries),
        relevance_rules.get("must_include", []),
        relevance_rules.get("must_exclude", []),
        (time.perf_counter() - started_at) * 1000,
    )

    return {
        "plan": [section.title() for section in sections],
        "tool_decision": list(DEFAULT_TOOLS),
        "planner_control": control,
        "queries": queries,
        "errors": errors,
    }
