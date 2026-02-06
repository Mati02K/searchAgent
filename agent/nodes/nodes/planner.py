from __future__ import annotations

import re

from nodes.state import AgentState


def _normalize_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if line:
            lines.append(line)
    return lines


def _extract_tagged(lines: list[str], tag: str) -> list[str]:
    prefix = f"{tag}:"
    values: list[str] = []
    for line in lines:
        if line.upper().startswith(prefix):
            value = line[len(prefix) :].strip()
            if value:
                values.append(value)
    return values


def _fallback_plan(prompt: str) -> tuple[list[str], list[str]]:
    base = " ".join(prompt.split())
    short = base[:100]
    plan = [
        f"Define scope and key terms for: {short}",
        "Collect credible evidence from multiple sources.",
        "Compare risks, benefits, and trade-offs before concluding.",
    ]
    queries = [
        short,
        f"{short} benefits",
        f"{short} risks",
    ]
    return plan, queries


def plan_research(state: AgentState) -> dict:
    """Create research angles and search queries."""
    if (state.get("report") or "").startswith("## Error"):
        return {}

    prompt = state.get("prompt", "")
    errors = list(state.get("errors", []))
    plan: list[str] = []
    queries: list[str] = []

    try:
        from llm.factory import get_llm

        llm = get_llm()
        system = (
            "You are a research planner. Return plain text only.\n"
            "Provide 3-6 lines prefixed with ANGLE:, then 3-6 lines prefixed with QUERY:.\n"
            "Keep each line concise and specific."
        )
        response = llm.generate(prompt, system=system)
        lines = _normalize_lines(response.text or "")
        plan = _extract_tagged(lines, "ANGLE")
        queries = _extract_tagged(lines, "QUERY")

        if not plan:
            plan = lines[:6]

    except Exception as exc:
        errors.append(f"Planner LLM error: {exc}")

    if not plan or not queries:
        fallback_plan, fallback_queries = _fallback_plan(prompt)
        if not plan:
            plan = fallback_plan
        if not queries:
            queries = fallback_queries

    plan = plan[:6]
    queries = queries[:6]
    if len(plan) < 3:
        fallback_plan, _ = _fallback_plan(prompt)
        plan = (plan + fallback_plan)[:3]
    if len(queries) < 3:
        _, fallback_queries = _fallback_plan(prompt)
        queries = (queries + fallback_queries)[:3]

    return {
        "plan": plan,
        "queries": queries,
        "errors": errors,
    }
