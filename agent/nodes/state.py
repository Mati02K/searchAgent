from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4


class AgentState(TypedDict):
    prompt: str
    plan: list[str]
    tool_decision: list[str]
    planner_control: dict[str, Any]
    queries: list[str]
    candidates: list[dict]
    sources: list[dict]
    evidence: list[dict]
    report: str
    trace_id: str
    errors: list[str]


def init_state(prompt: str) -> AgentState:
    """Create a new agent state with defaults."""
    return {
        "prompt": prompt,
        "plan": [],
        "tool_decision": [],
        "planner_control": {},
        "queries": [],
        "candidates": [],
        "sources": [],
        "evidence": [],
        "report": "",
        "trace_id": str(uuid4()),
        "errors": [],
    }
