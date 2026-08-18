"""MemoryStore interface — implemented by the local store and the HydraDB adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import Evidence, Fact, Session


class MemoryStore(ABC):
    @abstractmethod
    async def add_session(
        self, question_id: str, session: Session, facts: list[Fact] | None = None
    ) -> None:
        """Store a session (raw turns + summary). `facts` enables fact-augmented indexing:
        implementations may index the session text together with its facts' text."""

    @abstractmethod
    async def add_facts(self, question_id: str, facts: list[Fact]) -> None: ...

    @abstractmethod
    async def add_knowledge(self, question_id: str, documents: list[dict]) -> list[str]:
        """Add knowledge documents — the RAG segment of the unified multi-corpus
        story (memory lives in the base bucket, knowledge in the
        `{question_id}_knowledge` bucket). Each document: {"text": str, ...}.
        Returns the stored document ids."""

    @abstractmethod
    async def update_fact(self, fact: Fact) -> None:
        """Persist validity-interval changes (valid_to / superseded_by). Never deletes."""

    @abstractmethod
    async def delete_memory(self, question_id: str, memory_id: str) -> None:
        """Delete ONE memory by its id (fact id or session_id). Unknown id is a no-op."""

    @abstractmethod
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
        """Hybrid search scoped to one question. Facts must satisfy
        `valid_as_of(as_of_date)` unless include_superseded is set (used for
        'what did I previously think' questions). Sessions are always eligible.

        `collections`: optional multi-corpus weighting — e.g.
        [{"name": "qid_x", "weight": 0.7}, {"name": "qid_x_knowledge", "weight": 0.3}].
        Each named bucket is searched with its own lane ranks; the weight scales
        its scores before the final merge (RAG + memory in one database)."""

    @abstractmethod
    async def get_facts(self, question_id: str) -> list[Fact]: ...

    @abstractmethod
    async def get_sessions(self, question_id: str) -> list[Session]: ...

    @abstractmethod
    async def get_timeline(self, question_id: str, subject: str, predicate: str) -> list[Fact]:
        """All versions of a (subject, predicate), oldest first — including superseded."""

    @abstractmethod
    async def find_candidates(
        self, question_id: str, subject: str, predicate: str, text: str, threshold: float = 0.75
    ) -> list[Fact]:
        """Currently-valid facts that might conflict with a new fact: same normalized
        (subject, predicate), or same subject with embedding similarity >= threshold."""

    @abstractmethod
    async def has_question(self, question_id: str) -> bool: ...

    @abstractmethod
    async def delete_question(self, question_id: str) -> None: ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
