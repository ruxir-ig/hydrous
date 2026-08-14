"""HydraDB cloud adapter implementing MemoryStore (optional extra: `hydra`).

Mapping:
- One collection per LongMemEval instance: `qid_{question_id}`.
- Facts ingest with `id` as upsert key; validity interval lives in `additional_metadata`
  (valid_from / valid_to / superseded_by). Supersession = re-ingest same id with updated
  metadata — never delete.
- Query: query_by="hybrid", mode="thinking", recency_bias=0.4, and the question date is
  passed via additional_context (HydraDB has no native bi-temporal filter).
- Ingestion is asynchronous: the adapter polls databases.status / context.status.

Integration tests run only when HYDRA_DB_API_KEY is set. Without the SDK installed,
constructing HydraDBStore raises a clear error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from ..schema import Evidence, Fact, MemoryItem, Session
from .base import MemoryStore

log = logging.getLogger(__name__)

_SDK_HELP = "HydraDB SDK not installed. Run: uv sync --extra hydra  (or pip install 'hydradb-sdk>=2,<3')"


def fact_to_memory_item(fact: Fact) -> MemoryItem:
    return MemoryItem(
        id=fact.id,
        title=f"{fact.subject} {fact.predicate}",
        text=fact.text,
        infer=True,
        metadata=json.dumps({"question_id": fact.question_id, "kind": "fact"}),
        additional_metadata={
            "question_id": fact.question_id,
            "session_id": fact.session_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "fact_date": fact.fact_date,
            "valid_from": fact.valid_from,
            "valid_to": fact.valid_to,
            "superseded_by": fact.superseded_by,
        },
    )


def session_to_memory_item(question_id: str, session: Session) -> MemoryItem:
    pairs: list[dict[str, str]] = []
    pending_user: str | None = None
    for turn in session.turns:
        if turn.role == "user":
            pending_user = turn.content
        elif turn.role == "assistant" and pending_user is not None:
            pairs.append({"user": pending_user, "assistant": turn.content})
            pending_user = None
    item = MemoryItem(
        id=f"{question_id}:{session.session_id}",
        title=f"session {session.session_id} ({session.date})",
        infer=True,
        metadata=json.dumps({"question_id": question_id, "kind": "session"}),
        additional_metadata={
            "question_id": question_id,
            "session_id": session.session_id,
            "date": session.date,
            "summary": session.summary,
        },
    )
    if pairs:
        item.user_assistant_pairs = pairs
    else:
        item.text = session.text
    return item


class HydraDBStore(MemoryStore):
    def __init__(
        self,
        api_key: str,
        *,
        database: str = "hydra_mem",
        poll_interval: float = 5.0,
        poll_timeout: float = 600.0,
    ):
        try:
            from hydra_db import HydraDB
        except ImportError as e:
            raise RuntimeError(_SDK_HELP) from e
        self._client = HydraDB(token=api_key)
        self._database = database
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._ensure_database()

    @staticmethod
    def _collection(question_id: str) -> str:
        return f"qid_{question_id}"

    # ------------------------------------------------------------------ infra

    def _ensure_database(self) -> None:
        self._client.databases.create(database=self._database)
        deadline = time.monotonic() + self._poll_timeout
        while time.monotonic() < deadline:
            status = self._client.databases.status(self._database)
            if status.data.infra.ready_for_ingestion:
                return
            time.sleep(self._poll_interval)
        raise TimeoutError(f"HydraDB database {self._database!r} not ready within {self._poll_timeout}s")

    def _wait_indexed(self, collection: str) -> None:
        deadline = time.monotonic() + self._poll_timeout
        while time.monotonic() < deadline:
            status = self._client.context.status(database=self._database, collection=collection)
            statuses = status.data.statuses
            if statuses and statuses[0].indexing_status == "completed":
                return
            time.sleep(self._poll_interval)
        raise TimeoutError(f"HydraDB collection {collection!r} not indexed within {self._poll_timeout}s")

    async def _ingest_items(self, question_id: str, items: list[MemoryItem]) -> None:
        if not items:
            return
        collection = self._collection(question_id)
        memories = json.dumps([item.model_dump(exclude_none=True) for item in items])
        await asyncio.to_thread(
            self._client.context.ingest,
            type="memory",
            database=self._database,
            collection=collection,
            memories=memories,
        )
        await asyncio.to_thread(self._wait_indexed, collection)

    # ------------------------------------------------------------------ interface

    async def add_session(
        self, question_id: str, session: Session, facts: list[Fact] | None = None
    ) -> None:
        items = [session_to_memory_item(question_id, session)]
        if facts:  # fact-augmented indexing: ingest facts alongside their session
            items.extend(fact_to_memory_item(f) for f in facts)
        await self._ingest_items(question_id, items)

    async def add_facts(self, question_id: str, facts: list[Fact]) -> None:
        if facts and await self.has_question(question_id):
            # facts were already ingested with their session (fact-augmented); skip re-ingest
            return
        await self._ingest_items(question_id, [fact_to_memory_item(f) for f in facts])

    async def update_fact(self, fact: Fact) -> None:
        # upsert by id with updated validity metadata — never delete
        await self._ingest_items(fact.question_id, [fact_to_memory_item(fact)])

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
    ) -> list[Evidence]:
        from hydra_db.helpers import build_string

        result = await asyncio.to_thread(
            self._client.query,
            database=self._database,
            collection=self._collection(question_id),
            type="memory",
            query=query,
            query_by="hybrid",
            mode="thinking",
            max_results=top_k,
            recency_bias=0.4,
            additional_context=f"Question date: {as_of_date}" if as_of_date else None,
        )
        text = build_string(result)
        if not text or not text.strip():
            return []
        # HydraDB returns a synthesized context string rather than discrete rows;
        # we surface it as a single evidence item.
        return [
            Evidence(
                source="session",
                text=text,
                date=as_of_date or "",
                session_id="hydradb",
                score=1.0,
            )
        ]

    async def get_facts(self, question_id: str) -> list[Fact]:
        log.warning("HydraDBStore.get_facts: no native list API — returning []")
        return []

    async def get_sessions(self, question_id: str) -> list[Session]:
        log.warning("HydraDBStore.get_sessions: no native list API — returning []")
        return []

    async def get_timeline(self, question_id: str, subject: str, predicate: str) -> list[Fact]:
        log.warning(
            "HydraDBStore.get_timeline: no native bi-temporal API; validity is encoded in "
            "additional_metadata and resolved at query time — returning []"
        )
        return []

    async def find_candidates(
        self, question_id: str, subject: str, predicate: str, text: str, threshold: float = 0.75
    ) -> list[Fact]:
        # Conflict detection is delegated to HydraDB's `infer` pipeline; supersession is
        # applied via update_fact upserts. Returning [] = no local pre-emption.
        return []

    async def has_question(self, question_id: str) -> bool:
        # No native existence check; ingestion is idempotent (upsert by id), so returning
        # False is safe but may re-ingest. The extraction cache keeps this cheap.
        return False

    async def delete_question(self, question_id: str) -> None:
        log.warning("HydraDBStore.delete_question: append-only store — no delete (by design)")

    def close(self) -> None:
        pass
