from __future__ import annotations

from collections.abc import Callable

try:
    from langgraph.graph import END, StateGraph

    HAS_LANGGRAPH = True
except ModuleNotFoundError:
    END = "__end__"
    StateGraph = None
    HAS_LANGGRAPH = False

from nodes.planner_agent.graph import run_planner_agent
from nodes.state import AgentState, init_state
from safety.node import SAFETY_BLOCK_REPORT, safety_precheck


class _FallbackGraph:
    def __init__(
        self,
        node_order: list[Callable[[AgentState], dict]],
        route_after_safety: Callable[[AgentState], str],
    ):
        self.node_order = node_order
        self.route_after_safety = route_after_safety

    def invoke(self, state: AgentState) -> AgentState:
        current = dict(state)
        for idx, node in enumerate(self.node_order):
            update = node(current)
            if update:
                current.update(update)
            if idx == 0 and self.route_after_safety(current) == "stop":
                return current
        return current


def route_after_safety(state: AgentState) -> str:
    if state.get("report", "") == SAFETY_BLOCK_REPORT:
        return "stop"
    return "continue"


def planner_agent_node(state: AgentState) -> dict:
    """Run dedicated planner-agent graph and map output into main state."""
    prompt = state.get("prompt", "")
    existing_errors = list(state.get("errors", []))

    try:
        planner_state = run_planner_agent(prompt)
    except Exception as exc:
        existing_errors.append(f"Planner-agent execution error: {exc}")
        return {"errors": existing_errors}

    sources = list(planner_state.get("sources", []))
    evidence = [
        {
            "statement": item.get("summary", ""),
            "url": item.get("url", ""),
            "source": item.get("source", ""),
        }
        for item in sources
        if isinstance(item, dict)
    ]
    merged_errors = existing_errors + list(planner_state.get("errors", []))

    return {
        "plan": list(planner_state.get("plan", [])),
        "queries": list(planner_state.get("queries", [])),
        "sources": sources,
        "evidence": evidence,
        "report": str(planner_state.get("report", "")),
        "errors": merged_errors,
    }


def build_graph():
    """Build and compile the SearchAgent graph."""
    if not HAS_LANGGRAPH:
        return _FallbackGraph(
            [safety_precheck, planner_agent_node],
            route_after_safety,
        )

    builder = StateGraph(AgentState)
    builder.add_node("safety_precheck", safety_precheck)
    builder.add_node("planner_agent_node", planner_agent_node)

    builder.set_entry_point("safety_precheck")
    builder.add_conditional_edges(
        "safety_precheck",
        route_after_safety,
        {
            "continue": "planner_agent_node",
            "stop": END,
        },
    )
    builder.add_edge("planner_agent_node", END)

    return builder.compile()


def run_graph(prompt: str) -> AgentState:
    """Initialize state, run graph, and return final state."""
    graph = build_graph()
    state = init_state(prompt)
    result = graph.invoke(state)
    return result
