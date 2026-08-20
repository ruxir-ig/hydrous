"""Deep interface for the explicit memory graph.

The rest of Hydrous asks named graph questions. It does not know whether the
answers come from HydraDB OSS, the test fake, or a future graph implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..schema import Fact, Session
from ..text_utils import content_words, word_overlap


@dataclass(frozen=True)
class GraphPath:
    """A named traversal result suitable for the demo and reader evidence."""

    nodes: tuple[str, ...]
    predicates: tuple[str, ...]
    facts: tuple[str, ...] = field(default_factory=tuple)
    cypher: str = ""

    @property
    def text(self) -> str:
        if not self.nodes:
            return ""
        out = [self.nodes[0]]
        for predicate, node in zip(self.predicates, self.nodes[1:], strict=False):
            out.append(f"-[:{predicate}]-> {node}")
        return " ".join(out)


class MemoryGraph(ABC):
    """Graph-native write and query contract for Hydrous memory."""

    @abstractmethod
    async def put_session(self, question_id: str, session: Session) -> None:
        """Create or update a Session vertex and its chronological relation."""

    @abstractmethod
    async def put_facts(self, question_id: str, facts: list[Fact]) -> None:
        """Project facts into Entity, Fact, Session, and PREDICATE edges."""

    @abstractmethod
    async def update_fact(self, fact: Fact) -> None:
        """Persist interval changes and the SUPERSEDED_BY edge."""

    @abstractmethod
    async def as_of(self, question_id: str, reference_date: str | None = None) -> list[Fact]:
        """Return graph facts valid at a date, or only current facts when omitted."""

    @abstractmethod
    async def supersession_chain(self, question_id: str, subject: str, predicate: str) -> list[Fact]:
        """Return all versions of one fact slot in chronological order."""

    @abstractmethod
    async def multi_hop(
        self,
        question_id: str,
        start_subject: str,
        predicates: list[str],
        reference_date: str | None = None,
    ) -> list[GraphPath]:
        """Follow a bounded named predicate chain from an entity."""

    async def search_paths(
        self,
        question_id: str,
        query: str,
        reference_date: str | None = None,
        top_k: int = 3,
        include_superseded: bool = False,
    ) -> list[GraphPath]:
        """Find relevant one- and two-hop paths and execute them in the graph.

        The candidate expansion is deliberately small and deterministic. The
        final path comes from ``multi_hop`` so an OSS implementation still
        executes the relationship traversal in HydraDB rather than rebuilding
        it from SQLite rows.
        """
        facts = await self.as_of(question_id, reference_date)
        if not facts:
            return []
        superseded_slots: set[tuple[str, str]] = set()
        history_paths: list[GraphPath] = []
        if include_superseded:
            current_slots = {(fact.subject, fact.predicate) for fact in facts}
            historical: dict[str, Fact] = {fact.id: fact for fact in facts}
            for subject, predicate in current_slots:
                chain = await self.supersession_chain(question_id, subject, predicate)
                if len(chain) < 2:
                    continue
                superseded_slots.add((subject, predicate))
                historical.update({fact.id: fact for fact in chain})
                history_paths.append(
                    GraphPath(
                        nodes=tuple(fact.value for fact in chain),
                        predicates=("SUPERSEDED_BY",) * (len(chain) - 1),
                        facts=tuple(fact.id for fact in chain),
                    )
                )
            facts = list(historical.values())
        query_words = content_words(query)
        candidates: dict[tuple[str, tuple[str, ...]], float] = {}
        for first in facts:
            if include_superseded and (first.subject, first.predicate) in superseded_slots:
                continue
            one_hop = (first.subject, (first.predicate,))
            one_text = f"{first.subject} -[:{first.predicate}]-> {first.value}"
            score = word_overlap(query, one_text)
            if score > 0:
                candidates[one_hop] = max(candidates.get(one_hop, 0.0), score)
            for second in facts:
                if second.subject != first.value:
                    continue
                predicates = (first.predicate, second.predicate)
                two_text = (
                    f"{first.subject} -[:{first.predicate}]-> {first.value} "
                    f"-[:{second.predicate}]-> {second.value}"
                )
                score = word_overlap(query, two_text)
                if score > 0 or not query_words:
                    candidates[(first.subject, predicates)] = max(
                        candidates.get((first.subject, predicates), 0.0), score
                    )

        paths: dict[tuple[str, ...], tuple[float, GraphPath]] = {
            path.facts: (word_overlap(query, path.text), path) for path in history_paths
        }
        ranked_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        for (start_subject, predicates), _score in ranked_candidates[: max(top_k * 4, 4)]:
            for path in await self.multi_hop(
                question_id, start_subject, list(predicates), reference_date
            ):
                if path.facts:
                    paths[path.facts] = (word_overlap(query, path.text), path)
        ranked_paths = sorted(paths.values(), key=lambda item: item[0], reverse=True)
        return [path for _score, path in ranked_paths[:top_k]]

    @abstractmethod
    async def delete_memory(self, question_id: str, memory_id: str, *, kind: str) -> None:
        """Delete one Fact or Session projection without rebuilding the scope."""

    @abstractmethod
    async def delete_question(self, question_id: str) -> None:
        """Remove the graph projection for one Hydrous scope."""

    @abstractmethod
    def close(self) -> None:
        """Release network or local graph resources."""
