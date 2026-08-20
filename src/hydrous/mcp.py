"""Native Model Context Protocol (MCP) server for Hydrous.

Runs over stdio (JSON-RPC 2.0) with zero external dependencies. Allows IDEs and
agents (Cursor, Claude Desktop, Windsurf, Cline, Antigravity) to seamlessly read,
write, time-travel, inspect supersessions, and perform cross-agent handoffs.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, BinaryIO, TextIO

from hydrous.mem0 import AsyncMemory, Memory
from hydrous.schema import ABSTAIN_STRING

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"


def encode_rpc_message(payload: dict[str, Any]) -> bytes:
    """Encode one JSON-RPC message with MCP/LSP Content-Length framing."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def decode_rpc_messages(data: bytes | str) -> list[dict[str, Any]]:
    """Parse a stdout dump that may mix Content-Length frames and NDJSON."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    messages: list[dict[str, Any]] = []
    i = 0
    while i < len(raw):
        if raw[i : i + 1] in (b" ", b"\t", b"\r", b"\n"):
            i += 1
            continue
        if raw[i : i + 1] == b"{":
            nl = raw.find(b"\n", i)
            chunk = raw[i:] if nl < 0 else raw[i:nl]
            messages.append(json.loads(chunk.decode("utf-8")))
            i = len(raw) if nl < 0 else nl + 1
            continue
        header_end = raw.find(b"\r\n\r\n", i)
        if header_end < 0:
            break
        headers = raw[i:header_end].decode("ascii", errors="replace")
        length = 0
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1].strip())
        start = header_end + 4
        end = start + length
        messages.append(json.loads(raw[start:end].decode("utf-8")))
        i = end
    return messages


def _read_line(buf: BinaryIO) -> bytes | None:
    line = bytearray()
    while True:
        ch = buf.read(1)
        if not ch:
            return None if not line else bytes(line)
        if ch == b"\n":
            return bytes(line)
        line.extend(ch)


def _read_exact(buf: BinaryIO, n: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < n:
        piece = buf.read(n - len(chunks))
        if not piece:
            return None
        chunks.extend(piece)
    return bytes(chunks)


def read_stdio_rpc(buf: BinaryIO) -> dict[str, Any] | None:
    """Read one MCP stdio message (Content-Length, or a JSON line)."""
    peek = getattr(buf, "peek", None)
    if callable(peek):
        while True:
            preview = peek(1)
            if not preview:
                return None
            if preview[:1] in (b" ", b"\t", b"\r", b"\n"):
                if not buf.read(1):
                    return None
                continue
            break
        preview = peek(1)
        if preview[:1] == b"{":
            line = _read_line(buf)
            if line is None:
                return None
            return json.loads(line.decode("utf-8").lstrip("\r"))
    headers: dict[str, str] = {}
    while True:
        line = _read_line(buf)
        if line is None:
            return None
        if line in (b"", b"\r"):
            break
        decoded = line.decode("ascii", errors="replace").rstrip("\r")
        if decoded.lstrip().startswith("{"):
            return json.loads(decoded)
        if ":" not in decoded:
            continue
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length") or 0)
    if length <= 0:
        return None
    body = _read_exact(buf, length)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))

TOOLS = [
    {
        "name": "memory_add",
        "description": (
            "Add conversational memory or facts into Hydrous. Automatically extracts "
            "atomic bi-temporal facts, scores confidence through the quality gate, and "
            "links supersessions if existing facts are updated/overwritten."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The message or fact text to store.",
                },
                "user_id": {
                    "type": "string",
                    "description": "User or project scope ID (defaults to active scope).",
                },
                "infer": {
                    "type": "boolean",
                    "description": (
                        "If True, runs LLM fact extraction + quality gate. If False, stores raw message."
                    ),
                    "default": True,
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Search agent memory using hybrid retrieval (dense + BM25) and graph paths. "
            "Supports time-travel queries via `reference_date` to see what was true at a historical date."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or concept to look up.",
                },
                "user_id": {
                    "type": "string",
                    "description": "User or project scope ID.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of memory results to return (default 5).",
                    "default": 5,
                },
                "reference_date": {
                    "type": "string",
                    "description": "Optional ISO date (e.g. '2024-01-01') for as-of time-travel retrieval.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_get_timeline",
        "description": (
            "Inspect how a decision, preference, or fact evolved over time. Returns the full "
            "supersession chain (e.g., Dell XPS -> MacBook Pro) with validity intervals and reasons."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Memory ID or fact ID to inspect history for.",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory_ask",
        "description": (
            "Ask a question against agent memory. Retrieves chronological evidence, resolves "
            "knowledge updates, and guarantees verbatim abstention if the answer is missing "
            "(never hallucinates unrecorded facts)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The natural language question to ask.",
                },
                "user_id": {
                    "type": "string",
                    "description": "User or project scope ID.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "memory_handoff",
        "description": (
            "Generate or retrieve a cross-agent handoff context packet. Summarizes active decisions, "
            "recent revisions, superseded choices, and open tasks so the next agent "
            "(e.g. Cursor after Claude) resumes without amnesia or repeated explanations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User or project scope ID.",
                },
            },
        },
    },
]

PROMPTS = [
    {
        "name": "agent_handoff",
        "description": "Resume working on a codebase/task with full cross-agent memory and decisions.",
        "arguments": [
            {
                "name": "task_summary",
                "description": "Brief description of current task or goal.",
                "required": False,
            }
        ],
    }
]


class MCPServer:
    """Stdio JSON-RPC 2.0 Model Context Protocol Server."""

    def __init__(
        self,
        memory: Memory | AsyncMemory,
        *,
        default_user_id: str = "default",
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.memory = memory
        self.default_user_id = default_user_id
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self._running = True

    def _write_message(self, payload: dict[str, Any]) -> None:
        framed = encode_rpc_message(payload)
        buf = getattr(self.stdout, "buffer", None)
        if buf is not None:
            buf.write(framed)
            buf.flush()
            return
        self.stdout.write(framed.decode("utf-8"))
        self.stdout.flush()

    def _send_response(self, request_id: Any, result: Any) -> None:
        self._write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
        )

    def _send_error(self, request_id: Any, code: int, message: str, data: Any = None) -> None:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        if data is not None:
            payload["error"]["data"] = data
        self._write_message(payload)

    def handle_message(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            self._send_error(None, -32700, f"Parse error: {exc}")
            return
        self._dispatch(msg)

    def _dispatch(self, msg: Any) -> None:
        if not msg:
            return
        if not isinstance(msg, dict):
            self._send_error(None, -32600, "Invalid Request: expected JSON object")
            return

        request_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {}) or {}

        # Notifications (no id)
        if request_id is None:
            if method == "notifications/initialized":
                log.info("MCP client initialized")
            return

        # Methods
        if method == "initialize":
            self._send_response(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"listChanged": False, "subscribe": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "hydrous-mcp",
                        "version": "0.1.0",
                    },
                },
            )
        elif method == "ping":
            self._send_response(request_id, {})
        elif method == "tools/list":
            self._send_response(request_id, {"tools": TOOLS})
        elif method == "tools/call":
            self._handle_tool_call(request_id, params.get("name", ""), params.get("arguments", {}) or {})
        elif method == "resources/list":
            self._send_response(
                request_id,
                {
                    "resources": [
                        {
                            "uri": f"hydrous://memory/{self.default_user_id}",
                            "name": f"Hydrous Active Memories ({self.default_user_id})",
                            "mimeType": "application/json",
                        },
                        {
                            "uri": f"hydrous://handoff/{self.default_user_id}",
                            "name": f"Hydrous Handoff Packet ({self.default_user_id})",
                            "mimeType": "text/markdown",
                        },
                    ]
                },
            )
        elif method == "resources/read":
            self._handle_resource_read(request_id, params.get("uri", ""))
        elif method == "prompts/list":
            self._send_response(request_id, {"prompts": PROMPTS})
        elif method == "prompts/get":
            self._handle_prompt_get(request_id, params.get("name", ""), params.get("arguments", {}) or {})
        else:
            self._send_error(request_id, -32601, f"Method not found: {method}")

    def _handle_tool_call(self, request_id: Any, name: str, args: dict[str, Any]) -> None:
        user_id = args.get("user_id") or self.default_user_id
        try:
            if name == "memory_add":
                text = args["text"]
                infer = args.get("infer", True)
                res = self.memory.add(text, user_id=user_id, infer=infer)
                result_text = json.dumps(res, indent=2, default=str)
                self._send_response(
                    request_id,
                    {"content": [{"type": "text", "text": result_text}], "isError": False},
                )
            elif name == "memory_search":
                query = args["query"]
                top_k = args.get("top_k", 5)
                ref_date = args.get("reference_date")
                res = self.memory.search(
                    query,
                    filters={"user_id": user_id},
                    top_k=top_k,
                    reference_date=ref_date,
                )
                result_text = json.dumps(res, indent=2, default=str)
                self._send_response(
                    request_id,
                    {"content": [{"type": "text", "text": result_text}], "isError": False},
                )
            elif name == "memory_get_timeline":
                mid = args["memory_id"]
                history = self.memory.history(mid)
                result_text = json.dumps(history, indent=2, default=str)
                self._send_response(
                    request_id,
                    {"content": [{"type": "text", "text": result_text}], "isError": False},
                )
            elif name == "memory_ask":
                question = args["question"]
                answer = self.memory.chat(question, filters={"user_id": user_id})
                is_abstain = answer.strip() == ABSTAIN_STRING
                status_note = (
                    "Verified Absent in Memory (Zero-Hallucination)"
                    if is_abstain
                    else "Grounded in Stored Evidence"
                )
                out = f"**Answer**: {answer}\n\n> *[Abstention Status: {status_note}]*"
                self._send_response(
                    request_id,
                    {"content": [{"type": "text", "text": out}], "isError": False},
                )
            elif name == "memory_handoff":
                all_mem = self.memory.get_all(filters={"user_id": user_id})
                results = all_mem.get("results", [])
                lines = [
                    f"# Hydrous Cross-Agent Handoff (Scope: {user_id})",
                    "",
                    f"**Total Registered Memories/Facts**: {len(results)}",
                    "",
                    "## Active Knowledge & Decisions",
                ]
                if not results:
                    lines.append("*No memories currently recorded in this scope.*")
                else:
                    for idx, item in enumerate(results, 1):
                        lines.append(f"{idx}. {item.get('memory', '')}")
                lines.extend(
                    [
                        "",
                        "## Handoff Guidance for the Resuming Agent",
                        "- Decisions recorded above have superseded past conflicting statements.",
                        "- Do not re-ask the user about preferences or decisions already resolved.",
                        "- Query `memory_search` or `memory_ask` before making assumptions on architecture.",
                    ]
                )
                self._send_response(
                    request_id,
                    {"content": [{"type": "text", "text": "\n".join(lines)}], "isError": False},
                )
            else:
                self._send_error(request_id, -32601, f"Unknown tool: {name}")
        except Exception as exc:  # noqa: BLE001
            log.exception("Tool %s execution failed", name)
            self._send_response(
                request_id,
                {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True},
            )

    def _handle_resource_read(self, request_id: Any, uri: str) -> None:
        try:
            if uri.startswith("hydrous://memory/"):
                uid = uri[len("hydrous://memory/") :] or self.default_user_id
                memories = self.memory.get_all(filters={"user_id": uid})
                self._send_response(
                    request_id,
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(memories, indent=2, default=str),
                            }
                        ]
                    },
                )
            elif uri.startswith("hydrous://handoff/"):
                uid = uri[len("hydrous://handoff/") :] or self.default_user_id
                all_mem = self.memory.get_all(filters={"user_id": uid})
                results = all_mem.get("results", [])
                text = f"# Handoff for {uid}\n\n" + "\n".join(
                    f"- {m.get('memory', '')}" for m in results
                )
                self._send_response(
                    request_id,
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/markdown",
                                "text": text,
                            }
                        ]
                    },
                )
            else:
                self._send_error(request_id, -32602, f"Resource not found: {uri}")
        except Exception as exc:  # noqa: BLE001
            self._send_error(request_id, -32603, f"Internal error reading resource: {exc}")

    def _handle_prompt_get(self, request_id: Any, name: str, args: dict[str, Any]) -> None:
        if name == "agent_handoff":
            task_summary = args.get("task_summary", "Continue current development task")
            handoff = self.memory.get_all(filters={"user_id": self.default_user_id})
            items = handoff.get("results", [])
            mem_summary = "\n".join(f"- {m.get('memory')}" for m in items) if items else "None"
            prompt_text = (
                f"You are resuming a multi-agent engineering task.\n"
                f"Task Goal: {task_summary}\n\n"
                f"Relevant Prior Decisions & Memory:\n{mem_summary}\n\n"
                f"Please inspect the current repository and continue from where the previous agent left off."
            )
            self._send_response(
                request_id,
                {
                    "description": "Resume task with cross-agent memory context",
                    "messages": [
                        {
                            "role": "user",
                            "content": {"type": "text", "text": prompt_text},
                        }
                    ],
                },
            )
        else:
            self._send_error(request_id, -32601, f"Prompt not found: {name}")

    def serve_forever(self) -> None:
        """Read MCP stdio messages until EOF (Content-Length framing, JSON lines accepted)."""
        raw = getattr(self.stdin, "buffer", None)
        while True:
            try:
                if raw is not None:
                    msg = read_stdio_rpc(raw)
                else:
                    line = self.stdin.readline()
                    if line == "":
                        break
                    line = line.strip()
                    if not line:
                        continue
                    msg = json.loads(line)
            except json.JSONDecodeError as exc:
                self._send_error(None, -32700, f"Parse error: {exc}")
                continue
            if msg is None:
                break
            self._dispatch(msg)
