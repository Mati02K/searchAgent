from __future__ import annotations

from langgraph.graph import END, StateGraph

from logging_utils import get_logger
from nodes.safety.node import SAFETY_BLOCK_REPORT, safety_node
from nodes.planner.node import planner_node
from nodes.reranker.node import reranker_node
from nodes.search.node import search_node
from nodes.state import AgentState, init_state

logger = get_logger(__name__)

def route_after_safety(state: AgentState) -> str:
    if state.get("report", "") == SAFETY_BLOCK_REPORT:
        return "stop"
    return "continue"


def build_graph():
    """Build and compile the main graph: safety -> planner -> reranker -> search."""
    builder = StateGraph(AgentState)
    builder.add_node("safety_node", safety_node)
    builder.add_node("planner_node", planner_node)
    builder.add_node("reranker_node", reranker_node)
    builder.add_node("search_node", search_node)

    builder.set_entry_point("safety_node")
    builder.add_conditional_edges(
        "safety_node",
        route_after_safety,
        {"continue": "planner_node", "stop": END},
    )
    builder.add_edge("planner_node", "reranker_node")
    builder.add_edge("reranker_node", "search_node")
    builder.add_edge("search_node", END)
    return builder.compile()


def run_graph(prompt: str) -> AgentState:
    try:
        state = init_state(prompt)
        logger.info("Graph run start. trace_id=%s prompt_len=%d", state["trace_id"], len(prompt or ""))
        graph = build_graph()
        result = graph.invoke(state)
        logger.info(
            "Graph run complete. trace_id=%s report_len=%d sources=%d errors=%d",
            result.get("trace_id", ""),
            len(result.get("report", "")),
            len(result.get("sources", [])),
            len(result.get("errors", [])),
        )
        return result
    except Exception as e:
        logger.exception("Graph execution failed with an exception.")
        return {
            "report": "",
            "sources": [],
            "errors": [f"Graph execution failed: {str(e)}"],
        }
