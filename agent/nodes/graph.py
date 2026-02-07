from __future__ import annotations

from langgraph.graph import END, StateGraph

from nodes.safety.node import SAFETY_BLOCK_REPORT, safety_node
from nodes.planner.node import planner_node
from nodes.search.node import search_node
from nodes.state import AgentState, init_state


def route_after_safety(state: AgentState) -> str:
    if state.get("report", "") == SAFETY_BLOCK_REPORT:
        return "stop"
    return "continue"


def build_graph():
    """Build and compile the main graph: safety -> planner -> search."""
    builder = StateGraph(AgentState)
    builder.add_node("safety_node", safety_node)
    builder.add_node("planner_node", planner_node)
    builder.add_node("search_node", search_node)

    builder.set_entry_point("safety_node")
    builder.add_conditional_edges(
        "safety_node",
        route_after_safety,
        {"continue": "planner_node", "stop": END},
    )
    builder.add_edge("planner_node", "search_node")
    builder.add_edge("search_node", END)
    return builder.compile()


def run_graph(prompt: str) -> AgentState:
    state = init_state(prompt)
    graph = build_graph()
    result = graph.invoke(state)
    return result
