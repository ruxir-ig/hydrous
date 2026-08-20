"""Deterministic in-memory MemoryGraph used by unit tests and offline demos."""

from __future__ import annotations

from collections import defaultdict

from ..schema import Fact, Session
from .base import GraphPath, MemoryGraph


class InMemoryGraph(MemoryGraph):
    def __init__(self) -> None:
        self._facts: dict[str, dict[str, Fact]] = defaultdict(dict)
        self._sessions: dict[str, dict[str, Session]] = defaultdict(dict)

    async def put_session(self, question_id: str, session: Session) -> None:
        self._sessions[question_id][session.session_id] = session

    async def put_facts(self, question_id: str, facts: list[Fact]) -> None:
        for fact in facts:
            self._facts[question_id][fact.id] = fact.model_copy(deep=True)

    async def update_fact(self, fact: Fact) -> None:
        if fact.id in self._facts[fact.question_id]:
            self._facts[fact.question_id][fact.id] = fact.model_copy(deep=True)

    async def as_of(self, question_id: str, reference_date: str | None = None) -> list[Fact]:
        return [
            f.model_copy(deep=True)
            for f in sorted(self._facts[question_id].values(), key=lambda f: f.valid_from)
            if f.valid_as_of(reference_date)
        ]

    async def supersession_chain(self, question_id: str, subject: str, predicate: str) -> list[Fact]:
        facts = {
            f.id: f
            for f in self._facts[question_id].values()
            if f.subject == subject and f.predicate == predicate
        }
        replaced = {f.superseded_by for f in facts.values() if f.superseded_by in facts}
        roots = [f for f in facts.values() if f.id not in replaced]
        if not roots:
            roots = list(facts.values())
        root = min(roots, key=lambda f: f.valid_from)
        chain: list[Fact] = []
        seen: set[str] = set()
        current: Fact | None = root
        while current is not None and current.id not in seen:
            chain.append(current.model_copy(deep=True))
            seen.add(current.id)
            current = facts.get(current.superseded_by or "")
        return chain

    async def multi_hop(
        self, question_id: str, start_subject: str, predicates: list[str], reference_date: str | None = None
    ) -> list[GraphPath]:
        current = [(start_subject, (), ())]
        facts = await self.as_of(question_id, reference_date)
        for predicate in predicates:
            next_paths = []
            for node, nodes, fact_ids in current:
                for fact in facts:
                    if fact.subject == node and fact.predicate == predicate:
                        next_paths.append((fact.value, nodes + (node,), fact_ids + (fact.id,)))
            current = next_paths
        return [
            GraphPath(nodes=nodes + (node,), predicates=tuple(predicates), facts=fact_ids)
            for node, nodes, fact_ids in current
        ]

    async def delete_memory(self, question_id: str, memory_id: str, *, kind: str) -> None:
        if kind == "fact":
            self._facts[question_id].pop(memory_id, None)
        elif kind == "session":
            self._sessions[question_id].pop(memory_id, None)
        else:
            raise ValueError(f"unknown graph memory kind: {kind}")

    async def delete_question(self, question_id: str) -> None:
        self._facts.pop(question_id, None)
        self._sessions.pop(question_id, None)

    def close(self) -> None:
        self._facts.clear()
        self._sessions.clear()
