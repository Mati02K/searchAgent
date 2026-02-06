from __future__ import annotations

from typing import TypedDict
from uuid import uuid4


class AgentState(TypedDict):
    prompt: str
    plan: list[str]
    queries: list[str]
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
        "queries": [],
        "sources": [],
        "evidence": [],
        "report": "",
        "trace_id": str(uuid4()),
        "errors": [],
    }
