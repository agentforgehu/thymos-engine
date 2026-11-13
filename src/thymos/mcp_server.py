"""A tiny, dependency-free MCP server over stdio.

This exposes the most useful AgentMemory operations to any MCP-capable host
(Claude Desktop, Cursor, custom agents).  It is intentionally small: the
server owns one state file and serialises requests, so it is suitable for a
personal bot rather than a multi-tenant service.

Run:

    python -m thymos.mcp_server --state .thymos/mcp.json --offline
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .agent import AgentMemory
from .config import Profile, ThymosConfig

PROTOCOL_VERSION = "2024-11-05"


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
    }


TOOLS = [
    _tool(
        "memory_context",
        "Recall a compact, trust-labelled memory block for the current user turn.",
        {
            "user_text": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            "max_chars": {"type": "integer", "minimum": 100, "maximum": 4000},
            "max_tokens": {"type": "integer", "minimum": 50, "maximum": 4000},
        },
        ["user_text"],
    ),
    _tool(
        "memory_plan",
        "Explain the control-plane plan: intent, affect gate, operators, budget, selected ids.",
        {
            "user_text": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            "max_chars": {"type": "integer", "minimum": 100, "maximum": 4000},
            "max_tokens": {"type": "integer", "minimum": 50, "maximum": 4000},
        },
        ["user_text"],
    ),
    _tool(
        "memory_observe",
        "Store a user turn if it clears the importance threshold.",
        {
            "user_text": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        ["user_text"],
    ),
    _tool(
        "memory_recall",
        "Return ranked memory hits with retrievability and score components.",
        {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        ["query"],
    ),
    _tool(
        "memory_correct_fact",
        "Correct a wrong or outdated structured fact and persist the state.",
        {
            "predicate": {"type": "string"},
            "value": {"type": "string"},
            "old_value": {"type": "string"},
            "subject": {"type": "string", "default": "user"},
        },
        ["predicate", "value"],
    ),
    _tool(
        "memory_delete_fact",
        "Retire a structured fact while keeping its history.",
        {
            "predicate": {"type": "string"},
            "value": {"type": "string"},
            "fact_id": {"type": "string"},
            "subject": {"type": "string", "default": "user"},
        },
    ),
    _tool("memory_snapshot", "Return the inspectable memory/affect snapshot.", {}),
]


class MCPServer:
    def __init__(self, memory: AgentMemory) -> None:
        self.memory = memory

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        if method == "initialize":
            return self._result(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "thymos-memory", "version": "0.3.4"},
                },
            )
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            return self._result(request_id, {"tools": TOOLS})
        if method == "tools/call":
            return self._call_tool(request_id, request.get("params") or {})
        return self._error(request_id, -32601, f"method not found: {method}")

    def _call_tool(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            text = self._dispatch(name, args)
        except Exception as exc:  # MCP tool errors are data, not transport errors
            return self._result(
                request_id,
                {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True},
            )
        return self._result(request_id, {"content": [{"type": "text", "text": text}], "isError": False})

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        if name == "memory_context":
            return self.memory.context(
                str(args.get("user_text", "")),
                top_k=int(args.get("top_k", 5)),
                max_chars=args.get("max_chars"),
                max_tokens=args.get("max_tokens"),
            )
        if name == "memory_plan":
            return json.dumps(
                self.memory.plan(
                    str(args.get("user_text", "")),
                    top_k=int(args.get("top_k", 5)),
                    max_chars=args.get("max_chars"),
                    max_tokens=args.get("max_tokens"),
                ),
                ensure_ascii=False,
                indent=2,
            )
        if name == "memory_observe":
            item = self.memory.observe(
                str(args.get("user_text", "")),
                importance=args.get("importance"),
                tags=args.get("tags") or None,
            )
            return json.dumps({"stored": bool(item), "id": item.id if item else None}, ensure_ascii=False)
        if name == "memory_recall":
            hits = self.memory.recall(str(args.get("query", "")), top_k=int(args.get("top_k", 5)), review=False)
            return json.dumps(
                [
                    {
                        "id": hit.memory.id,
                        "content": hit.memory.content,
                        "score": round(hit.score, 4),
                        "retrievability": round(hit.retrievability, 4),
                        "components": {k: round(v, 4) for k, v in hit.components.items()},
                    }
                    for hit in hits
                ],
                ensure_ascii=False,
            )
        if name == "memory_correct_fact":
            fact = self.memory.correct_fact(
                str(args["predicate"]),
                str(args["value"]),
                old_value=args.get("old_value"),
                subject=args.get("subject", "user"),
            )
            return json.dumps(fact.to_dict(), ensure_ascii=False)
        if name == "memory_delete_fact":
            count = self.memory.delete_fact(
                predicate=args.get("predicate"),
                value=args.get("value"),
                fact_id=args.get("fact_id"),
                subject=args.get("subject", "user"),
            )
            return json.dumps({"deleted": count}, ensure_ascii=False)
        if name == "memory_snapshot":
            return json.dumps(self.memory.snapshot(), ensure_ascii=False, indent=2)
        raise KeyError(f"unknown tool: {name}")

    @staticmethod
    def _result(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def run_stdio(memory: AgentMemory, *, stdin=None, stdout=None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = MCPServer(memory)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n")
            stdout.flush()
            continue
        response = server.handle(request)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thymos-mcp", description="Thymos MCP stdio server")
    parser.add_argument("--state", default=".thymos/mcp.json")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--name", default="用户")
    parser.add_argument("--city", default="上海")
    args = parser.parse_args(argv)
    memory = AgentMemory(
        args.state,
        profile=Profile(name=args.name, location_name=args.city),
        config=ThymosConfig(offline=True),
    )
    return run_stdio(memory)


if __name__ == "__main__":
    raise SystemExit(main())
