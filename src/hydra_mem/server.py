"""Minimal HTTP bridge for the mem0-compatible surface (stdlib, no new deps).

Two jobs in one process:
1. Serve the mem0 API over HTTP — the backend the TypeScript SDK (and any
   language) talks to: the same method surface as `hydra_mem.Memory`.
2. Receive HydraDB `indexing.status_changed` webhooks at /webhooks/hydradb/indexing
   and mark the shared IndexingTracker, so a `--store hydradb` server's ingests
   stop polling and wake on the event instead. Register the webhook with
   HydraDB at startup when --callback-url is a PUBLIC URL.

Endpoints (JSON in / JSON out):
    POST /add /search /get_all /update /delete /delete_all /reset /chat
         /add_knowledge /import-mem0
    GET  /get/{memory_id}  /history/{memory_id}
    POST /webhooks/hydradb/indexing
    GET  /health
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from hydra_mem import Memory
from hydra_mem.indexing import IndexingTracker
from hydrous.schema import ABSTAIN_STRING

log = logging.getLogger(__name__)

POST_ROUTES = {
    "/add": "add",
    "/search": "search",
    "/get_all": "get_all",
    "/update": "update",
    "/delete": "delete",
    "/delete_all": "delete_all",
    "/reset": "reset",
    "/chat": "chat",
    "/add_knowledge": "add_knowledge",
    "/import-mem0": "import_mem0",
}


def _webhook_ids(payload: dict) -> list[str]:
    """Tolerate the plausible shapes of an indexing.status_changed payload:
    {"ids": [...]}, {"id": ...}, {"memory_id": ...}, {"source_id": ...}, or a
    list of item objects under "data" each with an id."""
    candidates: list[str] = []
    for key in ("ids", "memory_ids", "source_ids"):
        v = payload.get(key)
        if isinstance(v, list):
            candidates.extend(str(x) for x in v if x)
        elif v:
            candidates.append(str(v))
    for key in ("id", "memory_id", "source_id"):
        v = payload.get(key)
        if v:
            candidates.append(str(v))
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key in ("id", "memory_id", "source_id"):
                    if item.get(key):
                        candidates.append(str(item[key]))
                        break
    return list(dict.fromkeys(candidates))


class MemoryHandler(BaseHTTPRequestHandler):
    server: MemHTTPServer  # type: ignore[assignment]

    # ------------------------------------------------------------------ io

    def _send(self, code: int, body: Any) -> None:
        data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return body if isinstance(body, dict) else {}

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter access log
        log.info("http: %s %s", self.command, self.path)

    # ------------------------------------------------------------------ verbs

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"status": "ok", "abstain": ABSTAIN_STRING})
            return
        if path.startswith("/get/"):
            memory_id = path[len("/get/") :]
            item = self.server.memory.get(memory_id)
            if item is None:
                self._send(404, {"error": f"Memory with id {memory_id} not found"})
                return
            self._send(200, item)
            return
        if path.startswith("/history/"):
            memory_id = path[len("/history/") :]
            self._send(200, self.server.memory.history(memory_id))
            return
        self._send(404, {"error": f"unknown route: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/webhooks/hydradb/indexing":
            self._handle_webhook()
            return
        method = POST_ROUTES.get(path)
        if method is None:
            self._send(404, {"error": f"unknown route: {path}"})
            return
        body = self._read_json()
        try:
            result = getattr(self.server.memory, method)(**body)
        except TypeError as e:  # mem0-style signature mismatch
            self._send(400, {"error": str(e)})
            return
        except ValueError as e:  # mem0-parity validation error
            self._send(422, {"error": str(e)})
            return
        except NotImplementedError as e:
            self._send(501, {"error": str(e)})
            return
        except Exception as e:  # noqa: BLE001 — boundary: report, don't crash the server
            log.exception("route %s failed", path)
            self._send(500, {"error": str(e)})
            return
        self._send(200, result)

    def _handle_webhook(self) -> None:
        payload = self._read_json()
        db = payload.get("database") or self.server.database
        collection = payload.get("collection") or ""
        status = payload.get("indexing_status") or payload.get("status") or "completed"
        ids = _webhook_ids(payload)
        if not ids:
            self._send(400, {"error": "webhook payload carried no ids"})
            return
        self.server.tracker.mark(db, collection, ids, status)
        self._send(200, {"received": True, "ids": ids})


class MemHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        addr: tuple[str, int],
        memory: Memory,
        tracker: IndexingTracker,
        database: str,
    ):
        super().__init__(addr, MemoryHandler)
        self.memory = memory
        self.tracker = tracker
        self.database = database


def build_server(
    memory: Memory,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    tracker: IndexingTracker | None = None,
    database: str = "hydrous",
) -> MemHTTPServer:
    return MemHTTPServer((host, port), memory, tracker or IndexingTracker(), database)
