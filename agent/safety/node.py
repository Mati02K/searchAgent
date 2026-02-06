from __future__ import annotations

from nodes.state import AgentState
from safety.service import get_safety_agent_service

MIN_PROMPT_LEN = 3
MAX_PROMPT_LEN = 10_000
SAFETY_BLOCK_REPORT = "## Unable to Answer\n\nI can't answer this question."


def safety_precheck(state: AgentState) -> dict:
    """Run prompt safety checks before other nodes."""
    prompt = (state.get("prompt") or "").strip()
    errors = list(state.get("errors", []))

    if len(prompt) < MIN_PROMPT_LEN:
        errors.append("Prompt is too short. Provide at least 3 characters.")
        return {
            "errors": errors,
            "report": SAFETY_BLOCK_REPORT,
        }

    if len(prompt) > MAX_PROMPT_LEN:
        errors.append(f"Prompt exceeds max length of {MAX_PROMPT_LEN} characters.")
        return {
            "errors": errors,
            "report": SAFETY_BLOCK_REPORT,
        }

    service = get_safety_agent_service()
    decision = service.evaluate(prompt)
    if not decision.allowed:
        reason = decision.reason
        if decision.matched_terms:
            reason = f"{reason} Matched terms: {', '.join(decision.matched_terms)}"
        errors.append(reason)
        return {
            "errors": errors,
            "report": SAFETY_BLOCK_REPORT,
        }

    return {}
