from __future__ import annotations

from typing import Any

from nodes.planner_agent.state import PlannerAgentState
from tools.mcp.server import _handle_request


TOOL_NAME_BY_DECISION = {
    "wikipedia": "wikipedia_search",
    "arxiv": "arxiv_search",
}


def _mcp_tool_call(tool_name: str, query: str, limit: int = 5) -> list[dict]:
    request = {
        "jsonrpc": "2.0",
        "id": f"{tool_name}:{query}",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": {
                "query": query,
                "limit": limit,
            },
        },
    }
    response = _handle_request(request)
    if not isinstance(response, dict):
        return []

    result = response.get("result")
    if not isinstance(result, dict):
        return []

    values = result.get("results")
    if not isinstance(values, list):
        return []

    clean_results: list[dict] = []
    for item in values:
        if isinstance(item, dict):
            clean_results.append(item)
    return clean_results


def execute_tools_node(state: PlannerAgentState) -> dict:
    """Execute selected MCP tools for generated queries and collect sources."""
    errors = list(state.get("errors", []))
    decisions = list(state.get("tool_decision", []))
    queries = [q for q in state.get("queries", []) if str(q).strip()]

    if not decisions or not queries:
        return {"sources": []}

    sources: list[dict[str, Any]] = []
    for decision in decisions:
        tool_name = TOOL_NAME_BY_DECISION.get(decision)
        if not tool_name:
            continue
        for query in queries:
            try:
                results = _mcp_tool_call(tool_name=tool_name, query=query, limit=5)
                sources.extend(results)
            except Exception as exc:
                errors.append(f"MCP tool call failed ({tool_name}): {exc}")

    unique_sources: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in sources:
        key = (
            str(item.get("source", "")),
            str(item.get("url", "")),
            str(item.get("title", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(item)

    return {
        "sources": unique_sources,
        "errors": errors,
    }
