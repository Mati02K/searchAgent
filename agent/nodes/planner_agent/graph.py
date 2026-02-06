from __future__ import annotations

from collections.abc import Callable

try:
    from langgraph.graph import END, StateGraph

    HAS_LANGGRAPH = True
except ModuleNotFoundError:
    END = "__end__"
    StateGraph = None
    HAS_LANGGRAPH = False

from nodes.planner_agent.nodes.execute_tools import execute_tools_node
from nodes.planner_agent.nodes.plan import planner_node
from nodes.planner_agent.nodes.synthesize import synthesize_node
from nodes.planner_agent.state import PlannerAgentState, init_planner_state


class _FallbackPlannerGraph:
    def __init__(self, node_order: list[Callable[[PlannerAgentState], dict]]):
        self.node_order = node_order

    def invoke(self, state: PlannerAgentState) -> PlannerAgentState:
        current = dict(state)
        for node in self.node_order:
            update = node(current)
            if update:
                current.update(update)
        return current


def build_planner_graph():
    if not HAS_LANGGRAPH:
        return _FallbackPlannerGraph([planner_node, execute_tools_node, synthesize_node])

    builder = StateGraph(PlannerAgentState)
    builder.add_node("planner_node", planner_node)
    builder.add_node("execute_tools_node", execute_tools_node)
    builder.add_node("synthesize_node", synthesize_node)

    builder.set_entry_point("planner_node")
    builder.add_edge("planner_node", "execute_tools_node")
    builder.add_edge("execute_tools_node", "synthesize_node")
    builder.add_edge("synthesize_node", END)
    return builder.compile()


def run_planner_agent(prompt: str) -> PlannerAgentState:
    graph = build_planner_graph()
    state = init_planner_state(prompt)
    result = graph.invoke(state)
    return result
