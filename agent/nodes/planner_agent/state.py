from __future__ import annotations

from typing import TypedDict


class PlannerAgentState(TypedDict):
    prompt: str
    plan: list[str]
    tool_decision: list[str]
    queries: list[str]
    sources: list[dict]
    report: str
    errors: list[str]


def init_planner_state(prompt: str) -> PlannerAgentState:
    return {
        "prompt": prompt,
        "plan": [],
        "tool_decision": [],
        "queries": [],
        "sources": [],
        "report": "",
        "errors": [],
    }
