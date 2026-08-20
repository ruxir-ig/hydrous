from __future__ import annotations

import asyncio

import pytest

from hydrous.embeddings import TfidfEmbedder
from hydrous.graph import InMemoryGraph
from hydrous.graph.hydradb_oss import HydraDBOSSGraph
from hydrous.graph.projector import GraphBackedStore
from hydrous.graph.queries import as_of, multi_hop, supersession_chain
from hydrous.schema import Evidence, Fact
from hydrous.store.local import LocalMemoryStore


def _fact(fid: str, subject: str, predicate: str, value: str, start: str, end: str | None = None) -> Fact:
    return Fact(
        id=fid,
        question_id="q1",
        subject=subject,
        predicate=predicate,
        value=value,
        fact_date=start,
        session_id="s1",
        valid_from=start,
        valid_to=end,
    )


def test_in_memory_graph_time_travel_and_supersession() -> None:
    async def run() -> None:
        graph = InMemoryGraph()
        old = _fact("old", "user", "uses_laptop", "Dell", "2023-01-01", "2024-03-15")
        new = _fact("new", "user", "uses_laptop", "MacBook", "2024-03-15")
        old = old.model_copy(update={"superseded_by": "new"})
        sister = _fact("sister", "user", "has_sister", "Anna", "2023-01-01")
        city = _fact("city", "Anna", "works_in", "London", "2023-01-01")
        await graph.put_facts("q1", [old, new, sister, city])

        assert [f.value for f in await graph.as_of("q1", "2024-01-01")] == ["Dell", "Anna", "London"]
        assert {f.value for f in await graph.as_of("q1")} == {"MacBook", "Anna", "London"}
        assert [f.id for f in await graph.supersession_chain("q1", "user", "uses_laptop")] == ["old", "new"]
        paths = await graph.multi_hop("q1", "user", ["has_sister", "works_in"])
        assert paths[0].text == "user -[:has_sister]-> Anna -[:works_in]-> London"

    asyncio.run(run())


def test_oss_wire_decodes_typed_rows_without_sdk() -> None:
    assert HydraDBOSSGraph._decode({"type": "string", "value": "London"}) == "London"
    assert HydraDBOSSGraph._decode({"type": "integer", "value": 3}) == 3
    assert HydraDBOSSGraph._decode({"type": "null", "value": None}) is None
    # HydraDB OSS currently rejects the interval predicate shape, so the
    # adapter fetches the scoped edges and applies Fact.valid_as_of locally.
    assert "r.question_id = $question_id" in as_of()
    assert "r0.name = $predicate_0" in multi_hop(hops=2)
    assert "[:SUPERSEDED_BY*1..4]" in supersession_chain()


def test_graph_evidence_key_includes_path_identity() -> None:
    first = Evidence(
        source="graph", text="path one", date="2023-01-01", session_id="s1", fact_id="graph:f1:f2"
    )
    second = Evidence(
        source="graph", text="path two", date="2023-01-01", session_id="s1", fact_id="graph:f3:f4"
    )

    assert first.key != second.key


def test_graph_backed_search_returns_graph_paths_to_the_reader(tmp_path) -> None:
    async def run() -> None:
        primary = LocalMemoryStore(tmp_path / "search.sqlite3", TfidfEmbedder())
        graph = InMemoryGraph()
        store = GraphBackedStore(primary, graph)
        facts = [
            _fact("sister", "user", "has_sister", "Anna", "2023-01-01"),
            _fact("city", "Anna", "works_in_city", "London", "2023-01-02"),
        ]
        await store.add_facts("q1", facts)

        results = await store.search("q1", "what city does my sister work in?", top_k=5)

        assert any(
            evidence.source == "graph"
            and "Anna" in evidence.text
            and "London" in evidence.text
            for evidence in results
        )
        store.close()

    asyncio.run(run())


def test_graph_backed_delete_memory_is_targeted(tmp_path) -> None:
    class NoRebuildGraph(InMemoryGraph):
        def __init__(self) -> None:
            super().__init__()
            self.deleted: list[tuple[str, str]] = []

        async def put_facts(self, question_id: str, facts: list[Fact]) -> None:
            if self._facts[question_id]:
                raise AssertionError("delete_memory must not rebuild the whole graph")
            await super().put_facts(question_id, facts)

        async def delete_memory(self, question_id: str, memory_id: str, *, kind: str) -> None:
            self.deleted.append((question_id, memory_id))
            await super().delete_memory(question_id, memory_id, kind=kind)

    async def run() -> None:
        primary = LocalMemoryStore(tmp_path / "delete.sqlite3", TfidfEmbedder())
        graph = NoRebuildGraph()
        store = GraphBackedStore(primary, graph)
        facts = [_fact("old", "user", "uses_laptop", "Dell", "2023-01-01"),
                 _fact("new", "user", "likes", "tea", "2023-01-02")]
        await store.add_facts("q1", facts)

        await store.delete_memory("q1", "old")

        assert graph.deleted == [("q1", "old")]
        assert [fact.id for fact in await graph.as_of("q1")] == ["new"]
        assert [fact.id for fact in await primary.get_facts("q1")] == ["new"]
        store.close()

    asyncio.run(run())


def test_graph_backed_delete_restores_after_partial_graph_failure(tmp_path) -> None:
    class FailingDeleteGraph(InMemoryGraph):
        async def delete_memory(self, question_id: str, memory_id: str, *, kind: str) -> None:
            await super().delete_memory(question_id, memory_id, kind=kind)
            raise RuntimeError("simulated graph delete failure")

    async def run() -> None:
        primary = LocalMemoryStore(tmp_path / "restore.sqlite3", TfidfEmbedder())
        graph = FailingDeleteGraph()
        store = GraphBackedStore(primary, graph)
        fact = _fact("f1", "user", "likes", "tea", "2023-01-01")
        await store.add_facts("q1", [fact])

        with pytest.raises(RuntimeError, match="simulated graph delete failure"):
            await store.delete_memory("q1", "f1")

        assert [item.id for item in await primary.get_facts("q1")] == ["f1"]
        assert [item.id for item in await graph.as_of("q1")] == ["f1"]
        store.close()

    asyncio.run(run())
