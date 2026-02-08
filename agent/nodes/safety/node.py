from __future__ import annotations

import time

from logging_utils import get_logger
from nodes.prompt import SAFETY_BLOCK_REPORT
from nodes.safety.service import get_safety_agent_service
from nodes.state import AgentState

MIN_PROMPT_LEN = 3
MAX_PROMPT_LEN = 10_000
logger = get_logger(__name__)


def safety_node(state: AgentState) -> dict:
    """Validate prompt safety before planner/search nodes execute."""
    started_at = time.perf_counter()
    errors = list(state.get("errors", []))
    prompt = (state.get("prompt") or "").strip()
    logger.info(
        "Safety start. trace_id=%s prompt_len=%d",
        state.get("trace_id", ""),
        len(prompt),
    )

    if len(prompt) < MIN_PROMPT_LEN:
        errors.append("Prompt is too short. Provide at least 3 characters.")
        logger.warning(
            "Safety blocked: prompt too short. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return {"errors": errors, "report": SAFETY_BLOCK_REPORT}

    if len(prompt) > MAX_PROMPT_LEN:
        errors.append(f"Prompt exceeds max length of {MAX_PROMPT_LEN} characters.")
        logger.warning(
            "Safety blocked: prompt too long. elapsed_ms=%.2f",
            (time.perf_counter() - started_at) * 1000,
        )
        return {"errors": errors, "report": SAFETY_BLOCK_REPORT}

    # Strict mode: classifier init/inference exceptions must propagate and stop graph execution.
    service = get_safety_agent_service()
    decision = service.evaluate(prompt)
    if not decision.allowed:
        errors.append(decision.reason)
        logger.warning(
            "Safety blocked by classifier. reason=%s elapsed_ms=%.2f",
            decision.reason,
            (time.perf_counter() - started_at) * 1000,
        )
        return {"errors": errors, "report": SAFETY_BLOCK_REPORT}

    logger.info("Safety passed. elapsed_ms=%.2f", (time.perf_counter() - started_at) * 1000)
    return {}
