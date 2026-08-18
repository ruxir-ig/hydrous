"""Unified RAG + memory in one database: multi-corpus type=knowledge|memory with
per-scope collections weighting (hydrous extension; mem0 has no knowledge corpus)."""

from types import SimpleNamespace

import pytest

from hydra_mem import Memory
from hydra_mem.api import _Scope
from hydrous.embeddings import TfidfEmbedder
from hydrous.llm import MockLLM
from hydrous.schema import Fact
from hydrous.store.local import LocalMemoryStore

from .conftest import run


@pytest.fixture
def mem(tmp_path) -> Memory:
    m = Memory(
        store="local",
        store_path=str(tmp_path / "mc.sqlite3"),
        llm=MockLLM(),
        embedder=TfidfEmbedder(),
    )
    yield m
    m.close()


def test_collection_bucket_mapping():
    from hydra_mem.api import AsyncMemory

    qid = "mem0_u_alice"
    assert AsyncMemory._collection_bucket(qid, "memory") == qid
    assert AsyncMemory._collection_bucket(qid, "base") == qid
    assert AsyncMemory._collection_bucket(qid, "knowledge") == f"{qid}_knowledge"
    assert AsyncMemory._collection_bucket(qid, "qid_other") == "qid_other"  # raw passes through


def test_add_knowledge_and_weighted_search(mem):
    r = mem.add_knowledge("Hydrous supports multi-corpus retrieval", user_id="alice")
    assert r["results"][0]["event"] == "ADD"
    kid = r["results"][0]["id"]

    found = mem.search(
        "multi-corpus retrieval", filters={"user_id": "alice"},
        collections=[{"name": "knowledge", "weight": 1.0}],
    )["results"]
    assert found and found[0]["id"] == kid

    # memory bucket alone knows nothing about it (knowledge is a separate corpus)
    assert mem.search("multi-corpus retrieval", filters={"user_id": "alice"})["results"] == []

    # weight 0 excludes the corpus entirely
    assert mem.search(
        "multi-corpus retrieval", filters={"user_id": "alice"},
        collections=[{"name": "knowledge", "weight": 0.0}],
    )["results"] == []


def test_weighted_merge_ranks_corpora(mem, tmp_path):
    """Both corpora have candidates; per-scope weights decide the winner."""
    qid = _Scope(user_id="mixed").question_id
    seeder = LocalMemoryStore(tmp_path / "mc.sqlite3", TfidfEmbedder())
    run(seeder.delete_question(qid))
    run(seeder.delete_question(f"{qid}_knowledge"))
    run(seeder.add_facts(qid, [
        Fact(id="mem_fact", question_id=qid, subject="user", predicate="knows",
             value="the kernel boots in two seconds", fact_date="2023-01-01",
             session_id="s1", valid_from="2023-01-01"),
    ]))
    run(seeder.add_knowledge(qid, [{"text": "kernel boots in two seconds flat"}]))
    seeder.close()

    # knowledge weight > memory weight -> the knowledge doc wins the tie
    results = mem.search(
        "kernel boots", filters={"user_id": "mixed"},
        collections=[{"name": "knowledge", "weight": 0.9}, {"name": "memory", "weight": 0.4}],
    )["results"]
    assert results and "two seconds flat" in results[0]["memory"]

    # flip the weights -> the memory fact wins
    results = mem.search(
        "kernel boots", filters={"user_id": "mixed"},
        collections=[{"name": "memory", "weight": 0.9}, {"name": "knowledge", "weight": 0.2}],
    )["results"]
    assert results and results[0]["id"] == "mem_fact"


def test_hydradb_knowledge_collection_shape():
    """Unit-level (no cloud): knowledge ingests to the _knowledge collection with
    type="knowledge"; weighted search sends the collections dict and drops the
    single-collection param (SDK constraint)."""
    from hydrous.store.hydradb import HydraDBStore

    captured: dict = {}

    class _FakeIngest:
        def __init__(self, _, **kw):
            captured["ingest_kwargs"] = kw

    class _FakeDb:
        create = None
        status = None

    class _FakeClient:
        def __init__(self):
            self.context = SimpleNamespace(ingest=_FakeIngest, status=None)

    store = HydraDBStore("fake", base_url="http://127.0.0.1:1")
    store._ready = True
    store._client = _FakeClient()

    async def fake_ingest_items(self, qid, items, *, item_type="memory", collection=None):
        captured["ingest"] = (item_type, collection)

    store._ingest_items = fake_ingest_items.__get__(store)

    run(store.add_knowledge("q1", [{"text": "docs"}]))
    assert captured["ingest"][0] == "knowledge"
    assert captured["ingest"][1] == "qid_q1_knowledge"


def test_hydradb_search_collections_passthrough():
    from hydrous.store.hydradb import HydraDBStore

    captured: dict = {}

    class _FakeQuery:
        def __init__(self, data):
            self.data = data

    class _FakeClient:
        def __init__(self):
            self.context = SimpleNamespace(status=None)

        async def query(self, **kwargs):
            captured["query_kwargs"] = kwargs
            return _FakeQuery(SimpleNamespace(chunks=[], graph_context=None))

    store = HydraDBStore("fake", base_url="http://127.0.0.1:1")
    store._ready = True
    store._client = _FakeClient()

    run(store.search("q1", "hello", collections=[{"name": "qid_q1", "weight": 0.7},
                                                 {"name": "qid_q1_knowledge", "weight": 0.3}]))
    kw = captured["query_kwargs"]
    assert "collection" not in kw  # SDK: don't send collection + collections together
    assert kw["collections"] == {"qid_q1": 0.7, "qid_q1_knowledge": 0.3}

    # no collections -> the plain single-collection path stays intact
    run(store.search("q1", "hello"))
    assert captured["query_kwargs"]["collection"] == "qid_q1"
    assert "collections" not in captured["query_kwargs"]
