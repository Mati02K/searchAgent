from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from logging_utils import get_logger

logger = get_logger(__name__)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8765/rpc")
MCP_CLIENT_TIMEOUT_SECONDS = float(os.getenv("MCP_CLIENT_TIMEOUT_SECONDS", "30.0"))


def call_mcp_tool(tool_name: str, query: str, limit: int = 5) -> list[dict]:
    """
    Call a tool through the MCP server using HTTP JSON-RPC.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return []

    request_id = f"{tool_name}:{cleaned_query}"
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": {"query": cleaned_query, "limit": int(limit)},
        },
    }
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        MCP_SERVER_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=MCP_CLIENT_TIMEOUT_SECONDS) as resp:
            response_text = resp.read().decode("utf-8", errors="replace")
        response = json.loads(response_text) if response_text else {}
        if not isinstance(response, dict):
            logger.warning("MCP HTTP response invalid shape. tool=%s", tool_name)
            return []
        if "error" in response:
            logger.warning("MCP tool returned error. tool=%s error=%s", tool_name, response.get("error"))
            return []
        result = response.get("result")
        if not isinstance(result, dict):
            logger.warning("MCP tool call returned invalid result shape. tool=%s", tool_name)
            return []
        rows = result.get("results")
        if not isinstance(rows, list):
            logger.warning("MCP tool call result missing results list. tool=%s", tool_name)
            return []
        return [row for row in rows if isinstance(row, dict)]
    except urllib.error.HTTPError as exc:
        logger.exception(
            "MCP HTTP error. tool=%s query=%s status=%s error=%s",
            tool_name,
            cleaned_query,
            getattr(exc, "code", "unknown"),
            exc,
        )
        return []
    except Exception as exc:
        logger.exception("MCP tool call failed. tool=%s query=%s error=%s", tool_name, cleaned_query, exc)
        return []
