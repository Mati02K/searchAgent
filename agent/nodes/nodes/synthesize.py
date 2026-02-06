from __future__ import annotations

from nodes.state import AgentState


def _fallback_report(state: AgentState) -> str:
    prompt = state.get("prompt", "").strip()
    plan = state.get("plan", [])[:6]
    evidence = state.get("evidence", [])[:8]
    urls = sorted(
        {item.get("url", "") for item in state.get("sources", []) if item.get("url")}
    )

    plan_lines = "\n".join(f"- {item}" for item in plan) or "- No plan available."
    evidence_lines = "\n".join(
        f"- {item.get('statement', '')} ({item.get('url', '')})" for item in evidence
    ) or "- No evidence collected."
    source_lines = "\n".join(f"- {url}" for url in urls) or "- No sources available."

    return (
        "# Search Report\n\n"
        "## Problem Framing\n"
        f"{prompt or 'No prompt provided.'}\n\n"
        "## Research Angles\n"
        f"{plan_lines}\n\n"
        "## Evidence\n"
        f"{evidence_lines}\n\n"
        "## Claims vs Evidence\n"
        "### Claims\n"
        "- Claims should be validated against the evidence below.\n\n"
        "### Evidence\n"
        f"{evidence_lines}\n\n"
        "## Sources\n"
        f"{source_lines}\n"
    )


def synthesize_report(state: AgentState) -> dict:
    """Generate the final markdown report from gathered state."""
    if (state.get("report") or "").startswith("## Error"):
        return {}

    errors = list(state.get("errors", []))

    prompt = state.get("prompt", "")
    plan = state.get("plan", [])[:6]
    evidence = state.get("evidence", [])[:8]
    sources = state.get("sources", [])[:8]
    urls = sorted({item.get("url", "") for item in sources if item.get("url")})

    summary_plan = "\n".join(f"- {p}" for p in plan) or "- None"
    summary_evidence = "\n".join(
        f"- {item.get('statement', '')} | {item.get('source', '')} | {item.get('url', '')}"
        for item in evidence
    ) or "- None"
    summary_urls = "\n".join(f"- {u}" for u in urls) or "- None"

    report = ""
    try:
        from llm.factory import get_llm

        llm = get_llm()
        system = (
            "You are a concise research synthesis assistant. "
            "Return markdown only and keep the report under 450 words."
        )
        synthesis_prompt = (
            f"Prompt:\n{prompt}\n\n"
            "Use exactly these sections:\n"
            "## Problem Framing\n"
            "## Research Angles\n"
            "## Evidence\n"
            "## Claims vs Evidence\n"
            "### Claims\n"
            "### Evidence\n"
            "## Sources\n\n"
            f"Research Angles:\n{summary_plan}\n\n"
            f"Evidence Items:\n{summary_evidence}\n\n"
            f"Source URLs:\n{summary_urls}\n"
        )
        response = llm.generate(synthesis_prompt, system=system)
        report = (response.text or "").strip()
    except Exception as exc:
        errors.append(f"Synthesis LLM error: {exc}")

    if not report:
        report = _fallback_report(state)

    return {
        "report": report,
        "errors": errors,
    }
