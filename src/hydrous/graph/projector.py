"""Keep the existing hybrid MemoryStore and the explicit graph in sync."""

from __future__ import annotations

from pathlib import Path

from ..embeddings import Embedder
from ..schema import Evidence, Fact, Session
from ..store.base import MemoryStore
from ..store.local import LocalMemoryStore
from ..text_utils import word_overlap
from .base import MemoryGraph


class GraphBackedStore(MemoryStore):
    """Composite store: SQLite serves hybrid retrieval; MemoryGraph serves graph queries."""

    def __init__(self, primary: MemoryStore, graph: MemoryGraph):
        self.primary = primary
        self.graph = graph

    @classmethod
    def local(cls, path: str | Path, embedder: Embedder, graph: MemoryGraph) -> GraphBackedStore:
        return cls(LocalMemoryStore(path, embedder), graph)

    async def add_session(self, question_id: str, session: Session, facts: list[Fact] | None = None) -> None:
        await self.primary.add_session(question_id, session, facts=facts)
        await self.graph.put_session(question_id, session)
        if facts:
            await self.graph.put_facts(question_id, facts)

    async def add_facts(self, question_id: str, facts: list[Fact]) -> None:
        await self.primary.add_facts(question_id, facts)
        await self.graph.put_facts(question_id, facts)

    async def add_knowledge(self, question_id: str, documents: list[dict]) -> list[str]:
        return await self.primary.add_knowledge(question_id, documents)

    async def update_fact(self, fact: Fact) -> None:
        await self.primary.update_fact(fact)
        await self.graph.update_fact(fact)

    async def delete_memory(self, question_id: str, memory_id: str) -> None:
        facts = await self.primary.get_facts(question_id)
        sessions = await self.primary.get_sessions(question_id)
        fact = next((fact for fact in facts if fact.id == memory_id), None)
        session = next((session for session in sessions if session.session_id == memory_id), None)
        target = fact or session
        if target is None:
            await self.primary.delete_memory(question_id, memory_id)
            return

        kind = "fact" if fact is not None else "session"
        try:
            # Delete from the projection first. If either side fails, restore
            # the one graph object instead of rebuilding or leaving a half-empty scope.
            await self.graph.delete_memory(question_id, memory_id, kind=kind)
            await self.primary.delete_memory(question_id, memory_id)
        except Exception:
            try:
                if kind == "fact":
                    await self.graph.put_facts(question_id, [target])
                else:
                    await self.graph.put_session(question_id, target)
                    linked_facts = [item for item in facts if item.session_id == memory_id]
                    if linked_facts:
                        await self.graph.put_facts(question_id, linked_facts)
            except Exception as restore_error:
                raise RuntimeError(
                    f"memory delete failed and graph restore also failed for {memory_id}"
                ) from restore_error
            raise

    async def search(
        self,
        question_id: str,
        query: str,
        *,
        as_of_date: str | None = None,
        top_k: int = 10,
        include_facts: bool = True,
        include_sessions: bool = True,
        include_superseded: bool = False,
        collections: list[dict] | None = None,
    ) -> list[Evidence]:
        results = await self.primary.search(
            question_id,
            query,
            as_of_date=as_of_date,
            top_k=top_k,
            include_facts=include_facts,
            include_sessions=include_sessions,
            include_superseded=include_superseded,
            collections=collections,
        )
        # Graph paths are first-class retrieval evidence. This is what makes
        # AsyncMemory.chat() graph-aware: its reader already consumes store.search().
        paths = await self.graph.search_paths(
            question_id,
            query,
            reference_date=as_of_date,
            top_k=max(3, min(top_k, 5)),
            include_superseded=include_superseded,
        )
        graph_facts = {fact.id: fact for fact in await self.graph.as_of(question_id, as_of_date)}
        if include_superseded:
            for fact in list(graph_facts.values()):
                for historical in await self.graph.supersession_chain(
                    question_id, fact.subject, fact.predicate
                ):
                    graph_facts[historical.id] = historical
        for path in paths:
            path_facts = [graph_facts[fact_id] for fact_id in path.facts if fact_id in graph_facts]
            if not path_facts:
                continue
            path_id = "graph:" + ":".join(path.facts)
            if all(predicate == "SUPERSEDED_BY" for predicate in path.predicates):
                text = (
                    f"Supersession path: {path_facts[0].subject} {path_facts[0].predicate}: "
                    f"{path_facts[0].value} -[:SUPERSEDED_BY]-> {path_facts[-1].value}"
                )
            else:
                text = f"Graph path: {path.text}"
            results.append(
                Evidence(
                    source="graph",
                    text=text,
                    date=min(fact.fact_date for fact in path_facts),
                    session_id=path_facts[0].session_id,
                    score=0.7 + 0.3 * word_overlap(query, path.text),
                    fact_id=path_id,
                )
            )
        deduped: dict[str, Evidence] = {}
        for evidence in results:
            if evidence.key not in deduped or evidence.score > deduped[evidence.key].score:
                deduped[evidence.key] = evidence
        return sorted(deduped.values(), key=lambda evidence: evidence.score, reverse=True)[:top_k]

    async def get_facts(self, question_id: str) -> list[Fact]:
        return await self.primary.get_facts(question_id)

    async def get_sessions(self, question_id: str) -> list[Session]:
        return await self.primary.get_sessions(question_id)

    async def get_timeline(self, question_id: str, subject: str, predicate: str) -> list[Fact]:
        return await self.primary.get_timeline(question_id, subject, predicate)

    async def find_candidates(
        self, question_id: str, subject: str, predicate: str, text: str, threshold: float = 0.75
    ):
        return await self.primary.find_candidates(question_id, subject, predicate, text, threshold)

    async def has_question(self, question_id: str) -> bool:
        return await self.primary.has_question(question_id)

    async def delete_question(self, question_id: str) -> None:
        await self.primary.delete_question(question_id)
        await self.graph.delete_question(question_id)

    def close(self) -> None:
        self.primary.close()
        self.graph.close()
