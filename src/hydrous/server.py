"""Minimal HTTP bridge for the mem0-compatible surface + Interactive Playground.

Two jobs in one process:
1. Serve the mem0 API over HTTP — the backend the TypeScript SDK (and any
   language) talks to: the same method surface as `hydrous.Memory`.
2. Receive HydraDB `indexing.status_changed` webhooks at /webhooks/hydradb/indexing
   and mark the shared IndexingTracker, so a `--store hydradb` server's ingests
   stop polling and wake on the event instead.
3. Serve the interactive Visual Playground at / and /playground with live graph
   state, OpenCypher inspector, as-of time-travel slider, and 1-click demos.

Endpoints (JSON in / JSON out):
    POST /add /search /get_all /update /delete /delete_all /reset /chat
         /add_knowledge /import-mem0
    GET  /get/{memory_id}  /history/{memory_id}
    POST /webhooks/hydradb/indexing
    GET  /health  /api/graph  /api/status
    POST /api/demos/run
    GET  /  /playground
"""

from __future__ import annotations

import asyncio
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from hydrous import Memory
from hydrous.indexing import IndexingTracker
from hydrous.mem0 import _Scope
from hydrous.schema import ABSTAIN_STRING, Fact

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
    """Tolerate the plausible shapes of an indexing.status_changed payload."""
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
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, code: int, html_content: str) -> None:
        data = html_content.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
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

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("http: %s %s", self.command, self.path)

    # ------------------------------------------------------------------ verbs

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query_params = parse_qs(parsed.query)

        if path in ("/", "/playground", "/index.html", "/playground.html"):
            self._serve_playground()
            return
        if path == "/health":
            self._send(200, {"status": "ok", "abstain": ABSTAIN_STRING})
            return
        if path == "/api/status":
            mem = self.server.memory
            backend = getattr(getattr(mem, "_async", mem), "_store_backend", "local")
            engines = {
                "hydradb-oss": "HydraDB OSS graph",
                "hydradb": "HydraDB cloud",
                "local": "local SQLite",
            }
            self._send(
                200,
                {
                    "status": "online",
                    "store": backend,
                    "engine": engines.get(str(backend), str(backend)),
                    "database": self.server.database,
                    "abstain_string": ABSTAIN_STRING,
                },
            )
            return
        if path == "/api/graph":
            user_id = query_params.get("user_id", ["demo"])[0]
            graph_data = self._get_graph_data(user_id)
            self._send(200, graph_data)
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
        if path == "/api/demos/run":
            self._handle_demo_run()
            return
        method = POST_ROUTES.get(path)
        if method is None:
            self._send(404, {"error": f"unknown route: {path}"})
            return
        body = self._read_json()
        try:
            result = getattr(self.server.memory, method)(**body)
        except TypeError as e:
            self._send(400, {"error": str(e)})
            return
        except ValueError as e:
            self._send(422, {"error": str(e)})
            return
        except NotImplementedError as e:
            self._send(501, {"error": str(e)})
            return
        except Exception as e:  # noqa: BLE001
            log.exception("route %s failed", path)
            self._send(500, {"error": str(e)})
            return
        self._send(200, result)

    def _serve_playground(self) -> None:
        candidates = [
            Path(__file__).parent.parent.parent / "playground" / "index.html",
            Path(__file__).parent.parent / "playground" / "index.html",
            Path("playground/index.html"),
        ]
        for p in candidates:
            if p.is_file():
                try:
                    content = p.read_text(encoding="utf-8")
                    self._send_html(200, content)
                    return
                except OSError:
                    pass
        # Fallback minimal HTML
        fallback = (
            "<!DOCTYPE html><html><head><title>Hydrous Playground</title></head>"
            "<body style='font-family:sans-serif;padding:2rem;background:#0d1117;color:#e6edf3'>"
            "<h1>Hydrous Bi-Temporal Agent Memory Playground</h1>"
            "<p>API is active. Edit <code>playground/index.html</code> to view the visual graph.</p>"
            "<p><a href='/health' style='color:#38bdf8'>/health</a> | "
            "<a href='/api/graph?user_id=demo' style='color:#38bdf8'>/api/graph</a></p>"
            "</body></html>"
        )
        self._send_html(200, fallback)

    def _get_graph_data(self, user_id: str) -> dict[str, Any]:
        """Extract graph entities, facts, and edges for the given user_id."""
        mem = self.server.memory
        qid = _Scope(user_id=user_id).question_id

        async def _collect() -> dict[str, Any]:
            store = await mem._async._ensure_store()
            facts = await store.get_facts(qid)
            entities: dict[str, dict[str, Any]] = {}
            fact_nodes: list[dict[str, Any]] = []
            edges: list[dict[str, Any]] = []

            for f in facts:
                s_id = f"entity:{f.subject}"
                v_id = f"entity:{f.value}"
                if f.subject not in entities:
                    entities[f.subject] = {"id": s_id, "name": f.subject, "type": "entity"}
                if f.value not in entities:
                    entities[f.value] = {"id": v_id, "name": f.value, "type": "entity"}

                is_active = f.valid_to is None
                fact_nodes.append(
                    {
                        "id": f.id,
                        "subject": f.subject,
                        "predicate": f.predicate,
                        "value": f.value,
                        "valid_from": f.valid_from,
                        "valid_to": f.valid_to,
                        "recorded_at": f.recorded_at,
                        "superseded_by": f.superseded_by,
                        "active": is_active,
                        "quality_score": f.quality_score,
                    }
                )

                edges.append(
                    {
                        "source": s_id,
                        "target": v_id,
                        "label": f.predicate,
                        "type": "predicate",
                        "fact_id": f.id,
                        "valid_from": f.valid_from,
                        "valid_to": f.valid_to,
                        "active": is_active,
                    }
                )

            by_id = {f.id: f for f in facts}
            for f in facts:
                nxt = by_id.get(f.superseded_by or "")
                if nxt is None:
                    continue
                edges.append(
                    {
                        "source": f"entity:{f.value}",
                        "target": f"entity:{nxt.value}",
                        "label": "SUPERSEDED_BY",
                        "type": "supersession",
                        "active": False,
                        "valid_from": f.valid_to or nxt.valid_from,
                        "valid_to": None,
                    }
                )

            return {
                "user_id": user_id,
                "question_id": qid,
                "entities": list(entities.values()),
                "facts": fact_nodes,
                "edges": edges,
                "total_facts": len(facts),
            }

        fut = asyncio.run_coroutine_threadsafe(_collect(), mem._loop)
        try:
            return fut.result(timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to collect graph data")
            return {"user_id": user_id, "error": str(exc), "entities": [], "facts": [], "edges": []}

    def _handle_demo_run(self) -> None:
        body = self._read_json()
        demo_id = int(body.get("demo_id", 1))
        user_id = str(body.get("user_id", "demo"))
        mem = self.server.memory
        qid = _Scope(user_id=user_id).question_id

        async def _seed_and_execute() -> dict[str, Any]:
            # Clean and seed demo data
            store = await mem._async._ensure_store()
            await store.delete_question(qid)
            f_dell = Fact(
                id="demo_dell_xps",
                question_id=qid,
                subject="user",
                predicate="uses_laptop",
                value="Dell XPS 13",
                fact_date="2023-01-01",
                session_id="demo_s1",
                valid_from="2023-01-01",
                valid_to="2024-03-15",
                superseded_by="demo_macbook_pro",
                quality_score=0.95,
            )
            f_mac = Fact(
                id="demo_macbook_pro",
                question_id=qid,
                subject="user",
                predicate="uses_laptop",
                value="MacBook Pro",
                fact_date="2024-03-15",
                session_id="demo_s2",
                valid_from="2024-03-15",
                valid_to=None,
                quality_score=0.98,
            )
            f_sister = Fact(
                id="demo_sister_anna",
                question_id=qid,
                subject="user",
                predicate="has_sister",
                value="Anna",
                fact_date="2023-01-05",
                session_id="demo_s1",
                valid_from="2023-01-05",
                valid_to=None,
                quality_score=0.92,
            )
            f_city = Fact(
                id="demo_anna_city",
                question_id=qid,
                subject="Anna",
                predicate="works_in_city",
                value="London",
                fact_date="2023-02-10",
                session_id="demo_s2",
                valid_from="2023-02-10",
                valid_to=None,
                quality_score=0.94,
            )
            await store.add_facts(qid, [f_dell, f_mac, f_sister, f_city])

            if demo_id == 1:
                # Demo 1: Laptop supersession
                q = "what laptop did I use before the MacBook?"
                search_res = await mem._async.search(q, filters={"user_id": user_id}, top_k=3)
                answer = await mem._async.chat(q, filters={"user_id": user_id})
                cypher = (
                    "MATCH (start:Fact {id: 'demo_dell_xps'})-[:SUPERSEDED_BY*1..4]->(curr:Fact)\n"
                    f"WHERE curr.question_id = '{qid}'\n"
                    "RETURN start.value AS prior, curr.value AS current, start.valid_to AS switch_date"
                )
                return {
                    "demo_id": 1,
                    "title": "Laptop Supersession & Evolution",
                    "query": q,
                    "cypher": cypher,
                    "search_results": search_res.get("results", []),
                    "chat_answer": answer,
                    "is_abstention": answer.strip() == ABSTAIN_STRING,
                    "explanation": (
                        "Dell XPS 13 validity closed on 2024-03-15 when replaced by MacBook Pro. "
                        "Old version remains retrievable via supersession chain."
                    ),
                }
            elif demo_id == 2:
                # Demo 2: As-of Time-travel
                q = "laptop"
                ref_date = "2024-01-01"
                search_res = await mem._async.search(
                    q, filters={"user_id": user_id}, top_k=3, reference_date=ref_date
                )
                cypher = (
                    f"MATCH (e:Entity)-[r:PREDICATE]->(v:Entity)\n"
                    f"WHERE r.question_id = '{qid}'\n"
                    f"  AND r.valid_from <= '{ref_date}'\n"
                    f"  AND (r.valid_to IS NULL OR r.valid_to > '{ref_date}')\n"
                    "RETURN e.name, r.name, v.name"
                )
                return {
                    "demo_id": 2,
                    "title": f"As-Of Time-Travel (as_of='{ref_date}')",
                    "query": f"search('{q}', reference_date='{ref_date}')",
                    "cypher": cypher,
                    "search_results": search_res.get("results", []),
                    "chat_answer": (
                        "Dell XPS 13 was active as of 2024-01-01 "
                        "(before MacBook Pro was acquired on 2024-03-15)."
                    ),
                    "is_abstention": False,
                    "explanation": (
                        "Valid interval filtering accurately returns what was true then "
                        "without destructive overwrites."
                    ),
                }
            elif demo_id == 3:
                # Demo 3: Safe Abstention
                q = "what did I eat for breakfast on 3 March 1999?"
                answer = await mem._async.chat(q, filters={"user_id": user_id})
                search_res = await mem._async.search(q, filters={"user_id": user_id}, top_k=3)
                cypher = (
                    f"MATCH (e:Entity)-[r:PREDICATE]->(v:Entity)\n"
                    f"WHERE r.question_id = '{qid}' AND r.name CONTAINS 'breakfast'\n"
                    "RETURN count(r) AS matches // returns 0 -> ABSTAIN"
                )
                return {
                    "demo_id": 3,
                    "title": "Zero-Hallucination Safe Abstention",
                    "query": q,
                    "cypher": cypher,
                    "search_results": search_res.get("results", []),
                    "chat_answer": answer,
                    "is_abstention": answer.strip() == ABSTAIN_STRING,
                    "explanation": (
                        "Evidence is absent. System returns verbatim official abstention string "
                        "rather than hallucinating plausible food items."
                    ),
                }
            else:
                # Demo 4: Multi-hop graph path
                q = "what city does my sister work in?"
                search_res = await mem._async.search(q, filters={"user_id": user_id}, top_k=3)
                answer = await mem._async.chat(q, filters={"user_id": user_id})
                cypher = (
                    f"MATCH (u:Entity {{name: 'user'}})-[r1:PREDICATE {{name: 'has_sister'}}]->(a:Entity)\n"
                    f"      -[r2:PREDICATE {{name: 'works_in_city'}}]->(c:Entity)\n"
                    f"WHERE r1.question_id = '{qid}' AND r2.question_id = '{qid}'\n"
                    "RETURN u.name, a.name, c.name // user -> Anna -> London"
                )
                return {
                    "demo_id": 4,
                    "title": "Multi-Hop Entity Graph Traversal",
                    "query": q,
                    "cypher": cypher,
                    "search_results": search_res.get("results", []),
                    "chat_answer": answer,
                    "is_abstention": False,
                    "explanation": (
                        "Traverses 2 hops in the HydraDB graph (user -> Anna -> London) "
                        "providing structured path evidence to the reader."
                    ),
                }

        fut = asyncio.run_coroutine_threadsafe(_seed_and_execute(), mem._loop)
        try:
            result = fut.result(timeout=10.0)
            # Attach updated graph
            result["graph"] = self._get_graph_data(user_id)
            self._send(200, result)
        except Exception as exc:  # noqa: BLE001
            log.exception("Demo run failed")
            self._send(500, {"error": str(exc)})

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
