from __future__ import annotations

from nodes.prompt import SAFETY_BLOCK_REPORT
from nodes.safety.service import get_safety_agent_service
from nodes.state import AgentState

MIN_PROMPT_LEN = 3
MAX_PROMPT_LEN = 10_000


def safety_node(state: AgentState) -> dict:
    """Validate prompt safety before planner/search nodes execute."""
    errors = list(state.get("errors", []))
    try:
        prompt = (state.get("prompt") or "").strip()

        if len(prompt) < MIN_PROMPT_LEN:
            errors.append("Prompt is too short. Provide at least 3 characters.")
            return {"errors": errors, "report": SAFETY_BLOCK_REPORT}

        if len(prompt) > MAX_PROMPT_LEN:
            errors.append(f"Prompt exceeds max length of {MAX_PROMPT_LEN} characters.")
            return {"errors": errors, "report": SAFETY_BLOCK_REPORT}

        service = get_safety_agent_service()
        decision = service.evaluate(prompt)
        if not decision.allowed:
            errors.append(decision.reason)
            return {"errors": errors, "report": SAFETY_BLOCK_REPORT}

        return {}
    except Exception as exc:
        errors.append(f"Safety node failure: {exc}")
        return {"errors": errors, "report": SAFETY_BLOCK_REPORT}
