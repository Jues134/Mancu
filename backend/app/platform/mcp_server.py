"""Minimal JSON-RPC MCP server for the pipeline's read-only tools.

The implementation supports initialize, tools/list, tools/call and notifications
without adding a separate protocol SDK dependency.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    #: JSON Schema for arguments (kept minimal but valid).
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any]]


class MCPServer:
    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> Callable[[Callable[[dict[str, Any]], Awaitable[Any]]], Callable[[dict[str, Any]], Awaitable[Any]]]:
        def decorator(fn: Callable[[dict[str, Any]], Awaitable[Any]]) -> Callable[[dict[str, Any]], Awaitable[Any]]:
            self._tools[name] = ToolSpec(name, description, input_schema, fn)
            return fn

        return decorator

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke one tool in-process, through the same registry the protocol uses.

        Exists so the pipeline can use its own tools without serialising JSON-RPC
        into itself: same handler, same argument validation by the schema, same
        error surface — one implementation, two callers (`tools/call` for external
        clients, this for in-process stages).
        """
        spec = self._tools.get(str(name))
        if spec is None:
            raise KeyError(f"未知工具：{name}")
        return await spec.handler(arguments or {})

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    # ---------------------------------------------------------------- protocol

    async def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC message. Returns None for notifications."""
        method = request.get("method")
        req_id = request.get("id")

        if method == "initialize":
            return _ok(
                req_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.name, "version": self.version},
                },
            )
        if method in ("notifications/initialized", "notifications/cancelled"):
            return None
        if method == "tools/list":
            return _ok(
                req_id,
                {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.input_schema,
                        }
                        for t in self._tools.values()
                    ]
                },
            )
        if method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            spec = self._tools.get(str(name))
            if spec is None:
                return _err(req_id, -32602, f"未知工具：{name}")
            try:
                result = await spec.handler(args)
            except Exception as exc:  # noqa: BLE001 - reported as a tool error
                return _ok(
                    req_id,
                    {
                        "content": [{"type": "text", "text": f"工具执行失败：{exc}"}],
                        "isError": True,
                    },
                )
            return _ok(req_id, _content(result))
        return _err(req_id, -32601, f"不支持的方法：{method}")

    async def serve_stdio(self) -> None:
        """Run the JSON-RPC loop over stdin/stdout (the standard MCP transport)."""
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                _write(_err(None, -32700, "JSON 解析失败"))
                continue
            response = await self.handle(request)
            if response is not None:
                _write(response)


def _ok(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _content(value: Any) -> dict[str, Any]:
    """Wrap a tool result as MCP content blocks."""
    if isinstance(value, str):
        return {"content": [{"type": "text", "text": value}]}
    return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, indent=2)}]}


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
