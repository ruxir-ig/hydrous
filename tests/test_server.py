"""HTTP bridge smoke: mem0 surface over HTTP + HydraDB indexing webhook endpoint."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from hydrous import Memory
from hydrous.embeddings import TfidfEmbedder
from hydrous.indexing import IndexingTracker
from hydrous.llm import MockLLM
from hydrous.schema import ABSTAIN_STRING
from hydrous.server import build_server


@pytest.fixture
def server(tmp_path):
    tracker = IndexingTracker()
    memory = Memory(
        store="local",
        store_path=str(tmp_path / "srv.sqlite3"),
        llm=MockLLM(),
        embedder=TfidfEmbedder(),
        webhook_tracker=tracker,
    )
    srv = build_server(memory, host="127.0.0.1", port=0, tracker=tracker, database="hydrous")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    memory.close()
    thread.join(timeout=5)


def _call(server, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    port = server.server_address[1]
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode()
            try:
                return resp.status, json.loads(text)
            except json.JSONDecodeError:
                return resp.status, text
    except urllib.error.HTTPError as e:
        text = e.read().decode()
        try:
            return e.code, json.loads(text)
        except json.JSONDecodeError:
            return e.code, text


def test_health(server):
    code, body = _call(server, "GET", "/health")
    assert code == 200
    assert body["status"] == "ok"
    assert body["abstain"] == ABSTAIN_STRING


def test_add_get_search_history_delete_roundtrip(server):
    code, added = _call(server, "POST", "/add", {"messages": "hello from the server", "user_id": "alice", "infer": False})
    assert code == 200
    mid = added["results"][0]["id"]
    assert added["results"][0]["event"] == "ADD"

    code, item = _call(server, "GET", f"/get/{mid}")
    assert code == 200 and item["memory"] == "hello from the server"

    code, found = _call(server, "POST", "/search", {"query": "hello server", "filters": {"user_id": "alice"}})
    assert code == 200 and found["results"] and found["results"][0]["id"] == mid

    code, history = _call(server, "GET", f"/history/{mid}")
    assert code == 200 and history[0]["event"] == "ADD"

    code, deleted = _call(server, "POST", "/delete", {"memory_id": mid})
    assert code == 200 and deleted == {"message": "Memory deleted successfully!"}

    code, _ = _call(server, "GET", f"/get/{mid}")
    assert code == 404  # mem0 parity: get of a deleted id is a 404 over HTTP


def test_chat_abstains_and_answers(server):
    code, abstained = _call(server, "POST", "/chat", {"query": "who am I?", "filters": {"user_id": "nobody"}})
    assert code == 200 and abstained == ABSTAIN_STRING

    _call(server, "POST", "/add", {"messages": "I love hiking", "user_id": "hiker", "infer": False})
    code, answer = _call(server, "POST", "/chat", {"query": "I love hiking", "filters": {"user_id": "hiker"}})
    assert code == 200 and "hiking" in answer


def test_infer_true_extracts_facts_over_http(server):
    code, added = _call(server, "POST", "/add", {"messages": "my favorite color is teal", "user_id": "d", "infer": True})
    assert code == 200 and added["results"]
    code, all_items = _call(server, "POST", "/get_all", {"filters": {"user_id": "d"}})
    assert code == 200
    assert all(i["metadata"]["kind"] == "fact" for i in all_items["results"])


def test_validation_errors_are_http_statuses(server):
    code, body = _call(server, "POST", "/search", {"query": "x"})  # missing entity filter
    assert code == 422 and "user_id" in body["error"]
    code, body = _call(server, "POST", "/delete", {"memory_id": "nope"})
    assert code == 422 and "not found" in body["error"]


def test_webhook_endpoint_marks_tracker(server):
    payload = {
        "database": "hydrous",
        "collection": "qid_q",
        "ids": ["src_1", "src_2"],
        "indexing_status": "completed",
    }
    code, body = _call(server, "POST", "/webhooks/hydradb/indexing", payload)
    assert code == 200 and body["received"] is True
    assert server.tracker.done("hydrous", "qid_q", {"src_1", "src_2"}) == {"src_1", "src_2"}

    # tolerant shapes: dict under "data" with per-item ids
    payload = {"database": "hydrous", "collection": "qid_y", "indexing_status": "failed",
               "data": [{"memory_id": "a"}, {"id": "b"}]}
    code, _ = _call(server, "POST", "/webhooks/hydradb/indexing", payload)
    assert code == 200
    assert server.tracker.failed("hydrous", "qid_y", {"a", "b"}) == {"a", "b"}

    # no ids -> 400
    code, _ = _call(server, "POST", "/webhooks/hydradb/indexing", {"status": "completed"})
    assert code == 400


def test_unknown_route_404(server):
    assert _call(server, "GET", "/nope")[0] == 404
    assert _call(server, "POST", "/nope")[0] == 404


def test_playground_and_status_api(server):
    code, _ = _call(server, "GET", "/playground")
    assert code == 200

    code, status = _call(server, "GET", "/api/status")
    assert code == 200 and status["status"] == "online"
    assert status["store"] == "local"
    assert "SQLite" in status["engine"]


def test_graph_and_demos_run_api(server):
    # Run Demo 1 (Laptop supersession)
    code, d1 = _call(server, "POST", "/api/demos/run", {"demo_id": 1, "user_id": "demo"})
    assert code == 200
    assert d1["demo_id"] == 1
    assert "Laptop Supersession" in d1["title"]
    assert "SUPERSEDED_BY" in d1["cypher"]
    assert len(d1["graph"]["facts"]) >= 4

    # Run Demo 3 (Abstention)
    code, d3 = _call(server, "POST", "/api/demos/run", {"demo_id": 3, "user_id": "demo"})
    assert code == 200
    assert d3["is_abstention"] is True
    assert d3["chat_answer"] == ABSTAIN_STRING

    # GET /api/graph
    code, g = _call(server, "GET", "/api/graph?user_id=demo")
    assert code == 200
    assert len(g["entities"]) >= 4
    assert len(g["edges"]) >= 4
    supers = [e for e in g["edges"] if e["type"] == "supersession"]
    assert any(
        e["source"] == "entity:Dell XPS 13" and e["target"] == "entity:MacBook Pro" for e in supers
    )
