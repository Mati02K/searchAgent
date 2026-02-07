from __future__ import annotations

import json

from nodes.prompt import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt
from nodes.state import AgentState
from tools.mcp.server import _handle_request

ALLOWED_TOOLS = {"wikipedia", "arxiv"}
TOOL_NAME_BY_DECISION = {
    "wikipedia": "wikipedia_search",
    "arxiv": "arxiv_search",
}

def _extract_json(text: str) -> dict | None:
    content = (text or "").strip()
    if not content:
        return None
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        pass
    
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(content[start : end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        print("Failed to parse JSON from planner response.")
        return None
    


def _string_list(value: object, max_items: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = str(item).strip()
        if s:
            out.append(s)
        if len(out) >= max_items:
            break
    return out


def _call_mcp_tool(tool_name: str, query: str, limit: int = 5) -> list[dict]:
    try:
        request = {
            "jsonrpc": "2.0",
            "id": f"{tool_name}:{query}",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": {"query": query, "limit": limit},
            },
        }
        response = _handle_request(request)
        if not isinstance(response, dict):
            return []
        result = response.get("result")
        if not isinstance(result, dict):
            return []
        items = result.get("results")
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]
    except Exception:
        return []


def planner_node(state: AgentState) -> dict:
    """Build plan, choose tools, generate queries, and execute tool calls."""
    prompt = state.get("prompt", "")
    errors = list(state.get("errors", []))
    plan: list[str] = []
    queries: list[str] = []
    tool_decision: list[str] = []
    try:
        try:
            from llm.factory import get_llm

            llm = get_llm()
            response = llm.generate(
                build_planner_user_prompt(prompt),
                system=PLANNER_SYSTEM_PROMPT,
            )
            parsed = _extract_json(response.text or "")
            if not parsed:
                errors.append("Planner LLM returned non-JSON or empty output.")
                raise ValueError("Planner LLM returned non-JSON or empty output.")
            else:
                plan = _string_list(parsed.get("plan"), 6)
                queries = _string_list(parsed.get("queries"), 6)
                raw_tools = _string_list(parsed.get("tool_decision"), 2)
                for tool in raw_tools:
                    t = tool.lower().strip()
                    if t in ALLOWED_TOOLS and t not in tool_decision:
                        tool_decision.append(t)
        except Exception as exc:
            print(f"Error in planner_node: {exc}")
            errors.append(f"Planner LLM error: {exc}")

        if len(plan) < 3:
            print("Planner produced fewer than 3 plan items.")
            errors.append("Planner produced fewer than 3 plan items.")
        if len(queries) < 1:
            print("Planner produced no queries.")
            errors.append("Planner produced no queries.")
        if not tool_decision:
            print("Planner produced no valid tool decision.")
            errors.append("Planner produced no valid tool decision.")

        sources: list[dict] = []
        for decision in tool_decision:
            tool_name = TOOL_NAME_BY_DECISION.get(decision)
            if not tool_name:
                continue
            for query in queries:
                try:
                    sources.extend(_call_mcp_tool(tool_name, query, limit=5))
                except Exception as exc:
                    print("Tool execution failed ({tool_name}): {exc}")
                    errors.append(f"Tool execution failed ({tool_name}): {exc}")

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

        evidence = [
            {
                "statement": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
            }
            for item in unique_sources
        ]

        return {
            "plan": plan[:6],
            "queries": queries[:6],
            "tool_decision": tool_decision,
            "sources": unique_sources,
            "evidence": evidence,
            "errors": errors,
        }

    except Exception as exc:
        errors.append(f"Planner node failure: {exc}")
        return {
            "plan": [],
            "queries": [],
            "tool_decision": [],
            "sources": [],
            "evidence": [],
            "errors": errors,
        }
