from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    # Support direct execution: `python server.py` from tools/mcp.
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.arxiv.tool import search_arxiv
from tools.wikipedia.tool import search_wikipedia


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    handler: Callable[..., list[dict]]


SEARCH_RESULT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "url": {"type": "string"},
        "source": {"type": "string", "enum": ["wikipedia", "arxiv"]},
        "published": {"type": ["string", "null"]},
        "authors": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
    },
    "required": ["title", "summary", "url", "source", "published", "authors"],
    "additionalProperties": False,
}

SEARCH_RESULTS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": SEARCH_RESULT_ITEM_SCHEMA,
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

TOOLS: dict[str, ToolSpec] = {
    "wikipedia_search": ToolSpec(
        name="wikipedia_search",
        description="Search Wikipedia and return page summaries.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema=SEARCH_RESULTS_OUTPUT_SCHEMA,
        handler=search_wikipedia,
    ),
    "arxiv_search": ToolSpec(
        name="arxiv_search",
        description="Search arXiv papers and return structured metadata.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema=SEARCH_RESULTS_OUTPUT_SCHEMA,
        handler=search_arxiv,
    ),
}


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        text = line.decode("utf-8", errors="replace").strip()
        if text == "":
            break
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None

    body = sys.stdin.buffer.read(content_length)
    if not body:
        return None
    return json.loads(body.decode("utf-8", errors="replace"))


def _write_message(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    header = f"Content-Length: {len(serialized)}\r\n\r\n".encode("utf-8")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(serialized)
    sys.stdout.buffer.flush()


def _tool_to_mcp_spec(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "inputSchema": spec.input_schema,
        "outputSchema": spec.output_schema,
    }


def _build_error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    spec = TOOLS.get(name)
    if spec is None:
        print(f"[mcp] unknown tool: {name}")
        return {"results": []}

    query = str(arguments.get("query") or "").strip()
    try:
        limit = int(arguments.get("limit", 5))
    except Exception:
        limit = 5

    try:
        results = spec.handler(query=query, limit=limit)
    except Exception as exc:
        print(f"[mcp] tool execution failed for '{name}': {exc}")
        results = []
    return {"results": results}


def _handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    try:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        if method == "notifications/initialized":
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "search-tools-mcp", "version": "0.1.0"},
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [_tool_to_mcp_spec(spec) for spec in TOOLS.values()],
                },
            }

        if method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": _run_tool(name, arguments),
            }

        return _build_error_response(request_id, -32601, f"Method not found: {method}")
    except Exception as exc:
        print(f"[mcp] error handling request: {exc}")
        return _build_error_response(None, -32603, f"Internal error: {exc}")


def serve_stdio() -> None:
    while True:
        try:
            message = _read_message()
            if message is None:
                break
            response = _handle_request(message)
            if response is not None:
                _write_message(response)
        except Exception as exc:
            print(f"[mcp] server error: {exc}")
            break


class _MCPHTTPHandler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
            return
        self._write_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/rpc":
            self._write_json(404, {"error": "not_found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            content_length = 0
        if content_length <= 0:
            self._write_json(400, _build_error_response(None, -32600, "Empty request body"))
            return

        raw = self.rfile.read(content_length)
        try:
            message = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as exc:
            self._write_json(400, _build_error_response(None, -32700, f"Parse error: {exc}"))
            return

        if not isinstance(message, dict):
            self._write_json(400, _build_error_response(None, -32600, "Invalid Request"))
            return

        response = _handle_request(message)
        if response is None:
            self.send_response(204)
            self.end_headers()
            return
        self._write_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[mcp-http] {self.address_string()} - {format % args}")


def serve_http(host: str = "0.0.0.0", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), _MCPHTTPHandler)
    print(f"[mcp-http] listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SearchAgent MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Server transport mode.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.transport == "http":
        serve_http(host=args.host, port=args.port)
    else:
        serve_stdio()
