from __future__ import annotations

from nodes.planner_agent.state import PlannerAgentState


def _fallback_report(state: PlannerAgentState) -> str:
    prompt = (state.get("prompt") or "").strip()
    plan = state.get("plan", [])[:6]
    sources = state.get("sources", [])[:10]

    plan_md = "\n".join(f"- {step}" for step in plan) or "- No plan available."
    findings_md = "\n".join(
        f"- {item.get('summary', '')} ({item.get('source', '')}: {item.get('url', '')})"
        for item in sources
    ) or "- No source findings available."
    citations_md = "\n".join(
        f"- [{idx}] {item.get('title', '')} — {item.get('url', '')}"
        for idx, item in enumerate(sources, start=1)
    ) or "- No citations."

    return (
        "# Research Report\n\n"
        "## Problem Framing\n"
        f"{prompt or 'No prompt provided.'}\n\n"
        "## Research Plan\n"
        f"{plan_md}\n\n"
        "## Findings\n"
        f"{findings_md}\n\n"
        "_Note: arXiv sources are preprints and may be unreviewed._\n\n"
        "## Citations\n"
        f"{citations_md}\n"
    )


def synthesize_node(state: PlannerAgentState) -> dict:
    """Generate the final markdown report from plan and gathered sources."""
    prompt = state.get("prompt", "")
    plan = state.get("plan", [])[:6]
    sources = state.get("sources", [])[:12]
    errors = list(state.get("errors", []))

    plan_text = "\n".join(f"- {step}" for step in plan) or "- None"
    source_lines = []
    for idx, item in enumerate(sources, start=1):
        source_lines.append(
            f"[{idx}] title={item.get('title', '')} | source={item.get('source', '')} | "
            f"published={item.get('published', None)} | authors={item.get('authors', None)} | "
            f"url={item.get('url', '')} | summary={item.get('summary', '')}"
        )
    sources_text = "\n".join(source_lines) or "None"

    report = ""
    try:
        from llm.factory import get_llm

        llm = get_llm()
        system = (
            "You are a careful research synthesis agent.\n"
            "Return markdown only.\n"
            "Use sections exactly:\n"
            "## Title\n"
            "## Problem Framing\n"
            "## Research Plan\n"
            "## Findings\n"
            "## Citations\n"
            "Clearly attribute claims to citations.\n"
            "Explicitly note that arXiv papers are unreviewed preprints.\n"
            "Keep output concise and factual."
        )
        llm_prompt = (
            f"User prompt:\n{prompt}\n\n"
            f"Research plan:\n{plan_text}\n\n"
            f"Sources:\n{sources_text}\n"
        )
        response = llm.generate(llm_prompt, system=system)
        report = (response.text or "").strip()
    except Exception as exc:
        errors.append(f"Planner synthesis LLM error: {exc}")

    if not report:
        report = _fallback_report(state)

    return {
        "report": report,
        "errors": errors,
    }
