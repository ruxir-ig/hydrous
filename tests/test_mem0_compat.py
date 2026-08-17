"""mem0 drop-in contract tests, lifted from mem0's own docs (mem0 v1.1+):

- README usage:  `memory.search(query=message, filters={"user_id": user_id}, top_k=3)`
                 `memory.add(messages, user_id=user_id)`
- add:           `add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None,
                 timestamp=None, expiration_date=None, infer=True, memory_type=None, prompt=None)`
- search:        `search(query, *, top_k=20, filters=None, threshold=0.1, rerank=False,
                 explain=False, reference_date=None, show_expired=False, **kwargs)`
                 -> {"results": [{"id", "memory", "score", ...}]}; ValueError when
                 filters lack user_id/agent_id/run_id; mem0 OSS raises on reference_date
                 (we IMPLEMENT it — see test_reference_date_time_travel).
- get_all:       `get_all(*, filters=None, top_k=20, ...)` -> {"results": [...]}
- update:        `update(memory_id, text=None, metadata=None, data=None)`
                 -> {"message": "Memory updated successfully!"}
- delete:        `delete(memory_id)` -> {"message": ...}; ValueError for unknown ids.

These tests run offline (MockLLM + tfidf) and pass with NO API key.
"""

import json
import sqlite3

import pytest

from hydra_mem import AsyncMemory, Memory
from hydra_mem.api import _Scope
from hydrous.embeddings import TfidfEmbedder
from hydrous.llm import MockLLM
from hydrous.schema import ABSTAIN_STRING, Fact
from hydrous.store.local import LocalMemoryStore

from .conftest import run


@pytest.fixture
def mem(tmp_path) -> Memory:
    m = Memory(
        store="local",
        store_path=str(tmp_path / "mem.sqlite3"),
        llm=MockLLM(),
        embedder=TfidfEmbedder(),
    )
    yield m
    m.close()


@pytest.fixture
def async_mem(tmp_path) -> AsyncMemory:
    return AsyncMemory(
        store="local",
        store_path=str(tmp_path / "amem.sqlite3"),
        llm=MockLLM(),
        embedder=TfidfEmbedder(),
    )


# ------------------------------------------------------------------ add / search


def test_mem0_readme_example(mem):
    """The exact mem0 README pattern: add(messages, user_id=...) then
    search(query, filters={'user_id': ...}, top_k=3)."""
    mem.add("I am planning to go to the market tomorrow", user_id="alice")  # infer=True default
    results = mem.search(
        "what is alice planning to do tomorrow?", filters={"user_id": "alice"}, top_k=3
    )["results"]
    assert results
    assert set(("id", "memory", "score")) <= set(results[0])
    assert "market" in results[0]["memory"]


def test_add_raw_string_roundtrip(mem):
    r = mem.add("I love hiking on weekends", user_id="alice", infer=False)
    assert r["results"][0]["event"] == "ADD"
    added = r["results"][0]
    assert added["memory"] == "I love hiking on weekends"
    assert added["role"] == "user"

    found = mem.search("what does alice like?", filters={"user_id": "alice"}, top_k=3)["results"]
    assert found
    # search returns the bare content (role prefixes stripped), like mem0
    assert found[0]["memory"] == "I love hiking on weekends"
    assert found[0]["id"] == added["id"]


def test_add_messages_list_roles_and_system_skip(mem):
    r = mem.add(
        [
            {"role": "user", "content": "remind me to call mom"},
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "assistant", "content": "I will remind you."},
        ],
        user_id="bob",
        infer=False,
    )["results"]
    assert [x["role"] for x in r] == ["user", "assistant"]
    assert len(r) == 2
    all_results = mem.get_all(filters={"user_id": "bob"})["results"]
    assert len(all_results) == 2  # system message never stored


def test_add_requires_entity_scope(mem):
    with pytest.raises(ValueError, match="user_id, agent_id, run_id"):
        mem.add("hello", infer=False)


def test_search_requires_entity_filter(mem):
    with pytest.raises(ValueError, match="filters={{'user_id': 'u1'}}") as excinfo:
        mem.search("hello")
    # mem0 v1.1+ exact error family:
    assert "at least one of" in str(excinfo.value)


def test_search_accepts_rerank_categories_explain_and_threshold(mem):
    mem.add("loves espresso and cold brew", user_id="caf", infer=False)
    res = mem.search(
        "coffee preferences",
        filters={"user_id": "caf"},
        rerank=True,
        categories=["general", "preferences"],
        explain=True,
        threshold=0.0,
    )["results"]
    assert res
    assert "score_details" in res[0]  # raw RRF exposed when explain=True
    assert "rrf" in res[0]["score_details"]

    # threshold floors the score: nothing below 1.0 with a single best hit exists
    empty = mem.search("zzz nowhere", filters={"user_id": "caf"}, threshold=1.01)["results"]
    assert empty == []


def test_legacy_top_level_entity_kwargs_accepted(mem):
    # mem0 <=1.0 code style: search(query, user_id=...) — accepted (logged), not raised
    mem.add("likes Rust over Go", user_id="dev", infer=False)
    res = mem.search("programming language", user_id="dev", top_k=3)["results"]
    assert res and "Rust" in res[0]["memory"]


def test_get_all_shape_and_scope_isolation(mem):
    mem.add("one", user_id="u1", infer=False)
    mem.add("two", user_id="u1", infer=False)
    mem.add("three", user_id="u2", infer=False)
    items = mem.get_all(filters={"user_id": "u1"})["results"]
    assert len(items) == 2
    assert set(("id", "memory", "hash", "metadata", "created_at")) <= set(items[0])
    assert all(i["user_id"] == "u1" for i in items)
    assert len(mem.get_all(filters={"user_id": "u2"})["results"]) == 1


def test_scope_sanitizes_entity_ids(tmp_path):
    scope = _Scope(user_id="a b/c:d")
    assert scope.question_id == scope.question_id.replace("/", "_").replace(":", "_")
    assert "/" not in scope.question_id and ":" not in scope.question_id


# ------------------------------------------------------------ feedback loop


def test_feedback_loop_readd_after_search_is_noop(mem):
    mem.add("my favorite movie is Inception", user_id="film", infer=False)
    hit = mem.search("favorite movie", filters={"user_id": "film"})["results"][0]
    # re-adding exactly what search returned must NOT duplicate it
    r = mem.add(hit["memory"], user_id="film", infer=False)["results"][0]
    assert r["event"] == "NOOP"
    assert r["id"] == hit["id"]
    assert len(mem.get_all(filters={"user_id": "film"})["results"]) == 1


def test_infer_true_extracts_facts_only(mem):
    r = mem.add("I have an interview at Google next Monday", user_id="jobs", infer=True)
    assert r["results"]
    for item in r["results"]:
        assert item["event"] == "ADD"
    items = mem.get_all(filters={"user_id": "jobs"})["results"]
    # mem0 parity: raw conversation is NOT stored, only extracted (quality-gated) facts
    assert items and all(i["metadata"]["kind"] == "fact" for i in items)


def test_infer_true_duplicate_fact_is_noop(mem):
    first = mem.add("I drive a Toyota Corolla", user_id="car", infer=True)["results"]
    second = mem.add("I drive a Toyota Corolla", user_id="car", infer=True)["results"]
    assert [x["event"] for x in first] == ["ADD"]
    assert [x["event"] for x in second] == ["NOOP"]
    assert len(mem.get_all(filters={"user_id": "car"})["results"]) == 1


# ------------------------------------------------------------ update / delete


def test_update_replaces_raw_memory_text(mem):
    mem.add("I love hiking", user_id="alice", infer=False)
    mem_id = mem.get_all(filters={"user_id": "alice"})["results"][0]["id"]
    resp = mem.update(mem_id, text="I love bouldering")
    assert resp == {"message": "Memory updated successfully!"}
    updated = mem.get_all(filters={"user_id": "alice"})["results"][0]
    assert updated["memory"] == "I love bouldering"
    assert updated["id"] == mem_id  # id stable across update (mem0 contract)


def test_update_data_alias_is_deprecated_but_works(mem):
    mem.add("a", user_id="alice", infer=False)
    mem_id = mem.get_all(filters={"user_id": "alice"})["results"][0]["id"]
    mem.update(mem_id, data="b")
    assert mem.get_all(filters={"user_id": "alice"})["results"][0]["memory"] == "b"


def test_update_requires_something(mem):
    with pytest.raises(ValueError, match="At least one of text, metadata"):
        mem.update("whatever")


def test_update_unknown_id_raises(mem):
    with pytest.raises(ValueError, match="not found"):
        mem.update("nope", text="x")


def test_update_extracted_fact_not_supported(mem):
    fact_id = mem.add("I use arch btw", user_id="linux", infer=True)["results"][0]["id"]
    with pytest.raises(NotImplementedError):
        mem.update(fact_id, text="I use nix btw")


def test_delete_and_unknown_delete(mem):
    mem.add("secret", user_id="alice", infer=False)
    mem_id = mem.get_all(filters={"user_id": "alice"})["results"][0]["id"]
    assert mem.delete(mem_id) == {"message": "Memory deleted successfully!"}
    assert mem.get_all(filters={"user_id": "alice"})["results"] == []
    with pytest.raises(ValueError, match="not found"):
        mem.delete(mem_id)


def test_delete_all_and_reset(mem):
    mem.add("x", user_id="alice", infer=False)
    mem.add("y", user_id="bob", infer=False)
    assert mem.delete_all(user_id="alice") == {"message": "Memories deleted successfully!"}
    assert mem.get_all(filters={"user_id": "alice"})["results"] == []
    assert len(mem.get_all(filters={"user_id": "bob"})["results"]) == 1
    mem.reset()
    assert mem.get_all(filters={"user_id": "bob"})["results"] == []


# ------------------------------------------------------------ temporal demos


def test_reference_date_time_travel(mem, tmp_path):
    """as-of search: mem0 OSS raises ValueError for reference_date; we implement it
    over bi-temporal facts. Superseded facts are hidden from current search."""
    qid = _Scope(user_id="alice").question_id
    seeder = LocalMemoryStore(tmp_path / "mem.sqlite3", TfidfEmbedder())
    run(seeder.delete_question(qid))  # idempotent
    dell = Fact(
        id="laptop_dell", question_id=qid, subject="user", predicate="uses_laptop",
        value="Dell XPS 13", fact_date="2023-06-01", session_id="s1",
        valid_from="2023-06-01", valid_to="2024-03-15", superseded_by="laptop_macbook",
    )
    macbook = Fact(
        id="laptop_macbook", question_id=qid, subject="user", predicate="uses_laptop",
        value="MacBook Pro", fact_date="2024-03-15", session_id="s2", valid_from="2024-03-15",
    )
    run(seeder.add_facts(qid, [dell, macbook]))
    seeder.close()

    current = mem.search("laptop", filters={"user_id": "alice"})["results"]
    assert len(current) == 1 and current[0]["id"] == "laptop_macbook"  # superseded hidden

    past = mem.search("laptop", filters={"user_id": "alice"}, reference_date="2024-01-01")["results"]
    assert len(past) == 1 and past[0]["id"] == "laptop_dell"  # time travel to before the switch

    # full history visible via get_all (mem0 cannot show old versions at all)
    ids = {i["id"] for i in mem.get_all(filters={"user_id": "alice"})["results"]}
    assert ids == {"laptop_dell", "laptop_macbook"}


def test_chat_abstains_with_no_evidence(mem):
    answer = mem.chat("what did I eat for breakfast on 3 March 1999?", filters={"user_id": "empty"})
    assert answer == ABSTAIN_STRING


def test_chat_answers_from_memory(mem):
    mem.add("I love hiking on weekends", user_id="alice", infer=False)
    answer = mem.chat("I love hiking", filters={"user_id": "alice"})
    assert "hiking" in answer
    assert answer != ABSTAIN_STRING


def test_expiration_date_stored_in_metadata(mem):
    mem.add("ephemeral note", user_id="alice", infer=False, expiration_date="2020-12-31")
    meta = mem.get_all(filters={"user_id": "alice"})["results"][0]["metadata"]
    assert meta["expiration_date"] == "2020-12-31"


# ------------------------------------------- get / history (Phase 2.1 patch)


def test_get_returns_mem0_shape_and_none_for_unknown(mem):
    mem.add("loves espresso", user_id="caf", infer=False)
    mid = mem.get_all(filters={"user_id": "caf"})["results"][0]["id"]
    item = mem.get(mid)
    assert item["id"] == mid
    assert item["memory"] == "loves espresso"
    assert set(("hash", "created_at", "updated_at", "metadata", "user_id")) <= set(item)
    assert item["user_id"] == "caf"
    assert mem.get("nope") is None  # mem0 parity: get returns None, does not raise


def test_history_add_record_for_raw_memory(mem):
    mem.add("remember to water the plants", user_id="g", infer=False)
    mid = mem.get_all(filters={"user_id": "g"})["results"][0]["id"]
    history = mem.history(mid)
    assert len(history) == 1
    rec = history[0]
    # exact mem0 history-table field names (old_memory/new_memory/memory_id/event)
    assert set(("id", "memory_id", "old_memory", "new_memory", "event", "created_at",
                "updated_at", "is_deleted")) <= set(rec)
    assert rec["memory_id"] == mid
    assert rec["event"] == "ADD"
    assert rec["old_memory"] is None and rec["new_memory"] == "remember to water the plants"
    assert mem.history("nope") == []  # unknown -> empty list (mem0 parity)


def test_history_walks_supersession_chain(mem, tmp_path):
    """history(fact_id) returns the full version chain of the fact's slot:
    every version's ADD plus the supersession UPDATE — richer than mem0's flat
    previous_value/new_value log."""
    qid = _Scope(user_id="laptop").question_id
    seeder = LocalMemoryStore(tmp_path / "mem.sqlite3", TfidfEmbedder())
    run(seeder.delete_question(qid))
    dell = Fact(
        id="laptop_dell", question_id=qid, subject="user", predicate="uses_laptop",
        value="Dell XPS 13", fact_date="2023-06-01", session_id="s1",
        valid_from="2023-06-01", valid_to="2024-03-15", superseded_by="laptop_macbook",
    )
    macbook = Fact(
        id="laptop_macbook", question_id=qid, subject="user", predicate="uses_laptop",
        value="MacBook Pro", fact_date="2024-03-15", session_id="s2", valid_from="2024-03-15",
    )
    run(seeder.add_facts(qid, [dell, macbook]))
    seeder.close()

    # register ids (get_all sees superseded facts too; search would only see the current one)
    mem.get_all(filters={"user_id": "laptop"})

    history = mem.history("laptop_dell")
    events = [(r["memory_id"], r["event"], r["old_memory"], r["new_memory"]) for r in history]
    assert ("laptop_dell", "ADD", None, "user uses_laptop: Dell XPS 13") in events
    assert ("laptop_dell", "UPDATE", "user uses_laptop: Dell XPS 13", "user uses_laptop: MacBook Pro") in events
    assert ("laptop_macbook", "ADD", None, "user uses_laptop: MacBook Pro") in events
    # chronological: Dell's ADD before the supersession UPDATE/MacBook's ADD
    add_dell = next(i for i, r in enumerate(history) if r["event"] == "ADD" and r["memory_id"] == "laptop_dell")
    assert add_dell < next(i for i, r in enumerate(history) if r["event"] == "UPDATE")


# -------------------------------------------- durable id resolution (2.1)


def test_ids_resolve_after_process_restart(tmp_path):
    """delete()/update()/get() work on a fresh instance over the same store —
    the id -> scope registry is persisted, like mem0 resolving ids from storage."""
    db = tmp_path / "mem.sqlite3"
    with Memory(store="local", store_path=str(db), llm=MockLLM(), embedder=TfidfEmbedder()) as m1:
        m1.add("durable memory", user_id="alice", infer=False)
        mid = m1.get_all(filters={"user_id": "alice"})["results"][0]["id"]
    # registry sidecar persisted next to the sqlite
    assert (str(db) + ".ids.json") in [str(p) for p in tmp_path.iterdir()]

    with Memory(store="local", store_path=str(db), llm=MockLLM(), embedder=TfidfEmbedder()) as m2:
        assert m2.get(mid)["memory"] == "durable memory"
        assert m2.delete(mid) == {"message": "Memory deleted successfully!"}
        with pytest.raises(ValueError, match="not found"):
            m2.delete(mid)


def test_reset_wipes_other_instances_scopes(tmp_path):
    db = tmp_path / "mem.sqlite3"
    with Memory(store="local", store_path=str(db), llm=MockLLM(), embedder=TfidfEmbedder()) as m1:
        m1.add("alice data", user_id="alice", infer=False)
    with Memory(store="local", store_path=str(db), llm=MockLLM(), embedder=TfidfEmbedder()) as m2:
        m2.add("bob data", user_id="bob", infer=False)
        m2.reset()  # wipe-everything semantics, including alice's persisted scope
        assert m2.get_all(filters={"user_id": "bob"})["results"] == []
        assert m2.get_all(filters={"user_id": "alice"})["results"] == []
    with Memory(store="local", store_path=str(db), llm=MockLLM(), embedder=TfidfEmbedder()) as m3:
        assert m3.get_all(filters={"user_id": "alice"})["results"] == []  # still gone


def test_close_releases_store(tmp_path):
    m = Memory(store="local", store_path=str(tmp_path / "c.sqlite3"), llm=MockLLM(), embedder=TfidfEmbedder())
    m.add("x", user_id="u", infer=False)
    conn = m._async._store._conn
    m.close()
    with pytest.raises(sqlite3.ProgrammingError):  # closed conn
        conn.execute("SELECT 1")


def test_delete_all_clears_registry(tmp_path):
    db = tmp_path / "mem.sqlite3"
    with Memory(store="local", store_path=str(db), llm=MockLLM(), embedder=TfidfEmbedder()) as m:
        m.add("gone soon", user_id="alice", infer=False)
        mid = m.get_all(filters={"user_id": "alice"})["results"][0]["id"]
        m.delete_all(user_id="alice")
        assert m.get(mid) is None  # registry entry dropped
    with Memory(store="local", store_path=str(db), llm=MockLLM(), embedder=TfidfEmbedder()) as m2:
        assert m2.get_all(filters={"user_id": "alice"})["results"] == []


def test_search_keeps_all_hits_at_default_threshold(mem, tmp_path):
    """Rank-mapped scores floor at 0.1, so the default threshold 0.1 keeps every
    hit (a 2-result search returns 2); raising it trims the weak tail."""
    qid = _Scope(user_id="lang").question_id
    seeder = LocalMemoryStore(tmp_path / "mem.sqlite3", TfidfEmbedder())
    run(seeder.delete_question(qid))
    run(seeder.add_facts(qid, [
        Fact(id="f_rust", question_id=qid, subject="user", predicate="likes_language",
             value="Rust", fact_date="2023-01-01", session_id="s1", valid_from="2023-01-01"),
        Fact(id="f_go", question_id=qid, subject="user", predicate="likes_language",
             value="Go", fact_date="2023-02-01", session_id="s2", valid_from="2023-02-01"),
    ]))
    seeder.close()

    res = mem.search("likes_language", filters={"user_id": "lang"})["results"]
    assert {r["id"] for r in res} == {"f_rust", "f_go"}
    scores = [r["score"] for r in res]
    assert max(scores) == 1.0 and min(scores) == 0.1
    assert len(mem.search("likes_language", filters={"user_id": "lang"}, threshold=0.5)["results"]) == 1


def test_async_get_and_history(async_mem):
    async def go():
        await async_mem.add("prefers async apis", user_id="dev", infer=False)
        all_items = (await async_mem.get_all(filters={"user_id": "dev"}))["results"]
        mid = all_items[0]["id"]
        assert (await async_mem.get(mid))["memory"] == "prefers async apis"
        hist = await async_mem.history(mid)
        assert hist and hist[0]["event"] == "ADD"
        await async_mem.delete(mid)
        assert await async_mem.get(mid) is None

    run(go())


# ------------------------------------------------------------ async surface


def test_async_memory_mirrors_sync(async_mem):
    async def go():
        r = await async_mem.add("prefers async apis", user_id="dev", infer=False)
        assert r["results"][0]["event"] == "ADD"
        results = (await async_mem.search("api style", filters={"user_id": "dev"}))["results"]
        assert results and "async" in results[0]["memory"]
        all_items = (await async_mem.get_all(filters={"user_id": "dev"}))["results"]
        assert len(all_items) == 1
        await async_mem.delete(all_items[0]["id"])
        assert (await async_mem.get_all(filters={"user_id": "dev"}))["results"] == []

    run(go())


def test_memory_thread_safety_many_ops(mem):
    """The sync facade routes every call through one event-loop thread."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(mem.add, f"fact number {i}", user_id="par", infer=False) for i in range(12)
        ]
        for f in futures:
            assert f.result()["results"][0]["event"] == "ADD"
    assert len(mem.get_all(filters={"user_id": "par"})["results"]) == 12


def test_results_are_json_serializable(mem):
    mem.add("json safe", user_id="j", infer=False)
    search = mem.search("json safe", filters={"user_id": "j"}, explain=True)
    dumped = json.dumps(search, default=str)  # no np.float64 surprises
    assert "json safe" in dumped
    assert json.loads(dumped)["results"][0]["score"] == 1.0


def test_add_timestamp_raises_like_mem0(mem):
    with pytest.raises(ValueError, match="platform-only"):
        mem.add("x", user_id="alice", timestamp="2024-01-01", infer=False)


def test_empty_scope_two_instances_do_not_collide(tmp_path):
    m1 = Memory(store="local", store_path=str(tmp_path / "a.sqlite3"), llm=MockLLM(), embedder=TfidfEmbedder())
    m2 = Memory(store="local", store_path=str(tmp_path / "b.sqlite3"), llm=MockLLM(), embedder=TfidfEmbedder())
    try:
        m1.add("shared name", user_id="alice", infer=False)
        assert m2.get_all(filters={"user_id": "alice"})["results"] == []
    finally:
        m1.close()
        m2.close()
