from __future__ import annotations

import json
import re

from nodes.planner_agent.state import PlannerAgentState

ALLOWED_TOOLS = {"wikipedia", "arxiv"}


def _extract_json_object(text: str) -> dict | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = cleaned[start : end + 1]
    try:
        data = json.loads(snippet)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _coerce_list_of_strings(value: object, max_items: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def _infer_tools(prompt: str) -> list[str]:
    text = (prompt or "").strip().lower()
    if not text:
        return []

    if text in {"hi", "hello", "hey"}:
        return []

    wiki_markers = (
        "what is",
        "definition",
        "overview",
        "history",
        "background",
        "basics",
        "explain",
    )
    arxiv_markers = (
        "recent",
        "latest",
        "paper",
        "research",
        "study",
        "technical",
        "benchmark",
        "llm",
        "transformer",
        "algorithm",
        "model",
        "state of the art",
        "sota",
    )
    needs_wikipedia = any(marker in text for marker in wiki_markers)
    needs_arxiv = any(marker in text for marker in arxiv_markers)

    if needs_wikipedia and needs_arxiv:
        return ["wikipedia", "arxiv"]
    if needs_wikipedia:
        return ["wikipedia"]
    if needs_arxiv:
        return ["arxiv"]
    return ["wikipedia", "arxiv"]


def _fallback_plan_and_queries(prompt: str) -> tuple[list[str], list[str]]:
    compact = re.sub(r"\s+", " ", (prompt or "").strip())
    compact = compact[:160]
    plan = [
        f"Clarify the scope and intent of: {compact}",
        "Gather high-signal references from selected sources.",
        "Synthesize findings with explicit citations and caveats.",
    ]
    queries = [
        compact,
        f"{compact} overview",
        f"{compact} recent research",
    ]
    return plan, queries


def planner_node(state: PlannerAgentState) -> dict:
    """Create a plan, tool decision, and search queries from the user prompt."""
    prompt = state.get("prompt", "")
    errors = list(state.get("errors", []))

    plan: list[str] = []
    tool_decision: list[str] = []
    queries: list[str] = []

    try:
        from llm.factory import get_llm

        llm = get_llm()
        system = (
            "You are a research planner.\n"
            "Return strict JSON only with keys: plan, tool_decision, queries.\n"
            "Rules:\n"
            "- plan: 3-6 concise steps\n"
            "- tool_decision: subset of ['wikipedia','arxiv']\n"
            "- queries: 3-6 concise search queries\n"
            "Do not include any other text."
        )
        response = llm.generate(prompt, system=system)
        parsed = _extract_json_object(response.text or "")
        if parsed:
            plan = _coerce_list_of_strings(parsed.get("plan"), max_items=6)
            queries = _coerce_list_of_strings(parsed.get("queries"), max_items=6)
            raw_decision = _coerce_list_of_strings(parsed.get("tool_decision"), max_items=2)
            for item in raw_decision:
                normalized = item.strip().lower()
                if normalized in ALLOWED_TOOLS and normalized not in tool_decision:
                    tool_decision.append(normalized)
    except Exception as exc:
        errors.append(f"Planner agent LLM error: {exc}")

    fallback_plan, fallback_queries = _fallback_plan_and_queries(prompt)
    if len(plan) < 3:
        plan = (plan + fallback_plan)[:3]
    if len(queries) < 3:
        queries = (queries + fallback_queries)[:3]

    if not tool_decision:
        tool_decision = _infer_tools(prompt)
    else:
        tool_decision = [tool for tool in tool_decision if tool in ALLOWED_TOOLS]

    return {
        "plan": plan[:6],
        "tool_decision": tool_decision,
        "queries": queries[:6],
        "errors": errors,
    }
