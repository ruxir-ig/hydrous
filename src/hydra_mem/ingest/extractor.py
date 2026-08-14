"""Per-session fact extraction, disk caching, and bi-temporal supersession.

Extraction is order-independent → sessions are extracted concurrently (semaphore-limited).
Versioning is order-dependent → results are applied sequentially in chronological order.
Extraction results are cached to disk per (question_id, session_id); the cache is
invalidated when the extraction model changes, so reruns with the same model are free.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

from .. import prompts
from ..llm import LLMClient
from ..schema import Fact, Session
from ..store.base import MemoryStore
from ..text_utils import compare_dates, date_key

log = logging.getLogger(__name__)


def parse_json_array(text: str) -> list[dict]:
    """Robustly extract a JSON array of objects from an LLM response."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        log.warning("extraction returned unparseable JSON")
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


def make_fact_id(question_id: str, session_id: str, raw: dict) -> str:
    """Deterministic id from content — cache reloads reproduce identical facts."""
    key = "|".join(
        [
            question_id,
            session_id,
            str(raw.get("subject", "")),
            str(raw.get("predicate", "")),
            str(raw.get("value", "")),
            str(raw.get("fact_date", "")),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class ExtractionCache:
    def __init__(self, cache_dir: str | Path):
        self._dir = Path(cache_dir) / "extractions"

    def _path(self, question_id: str, session_id: str) -> Path:
        safe = f"{question_id}__{session_id}".replace("/", "_")
        return self._dir / f"{safe}.json"

    def load(self, question_id: str, session_id: str, model: str) -> tuple[list[dict], str] | None:
        path = self._path(question_id, session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("model") != model:
            return None
        return data.get("facts", []), data.get("summary", "")

    def save(self, question_id: str, session_id: str, model: str, facts: list[dict], summary: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(question_id, session_id)
        payload = {"model": model, "facts": facts, "summary": summary}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FactExtractor:
    def __init__(
        self,
        llm: LLMClient,
        store: MemoryStore,
        cache_dir: str | Path,
        *,
        model_name: str = "unknown",
        max_concurrency: int = 8,
        conflict_threshold: float = 0.75,
    ):
        self._llm = llm
        self._store = store
        self._cache = ExtractionCache(cache_dir)
        self._model = model_name
        self._sem = asyncio.Semaphore(max_concurrency)
        self._conflict_threshold = conflict_threshold

    @property
    def store(self) -> MemoryStore:
        return self._store

    async def extract_session(self, question_id: str, session: Session) -> tuple[list[Fact], str]:
        """Extract facts + summary for one session (cache-first)."""
        cached = self._cache.load(question_id, session.session_id, self._model)
        if cached is not None:
            raw_facts, summary = cached
        else:
            async with self._sem:
                facts_resp, summary_resp = await asyncio.gather(
                    self._llm.chat(prompts.extraction_messages(session)),
                    self._llm.chat(prompts.summary_messages(session)),
                )
            raw_facts, summary = parse_json_array(facts_resp), summary_resp.strip()
            self._cache.save(question_id, session.session_id, self._model, raw_facts, summary)
        facts = [f for f in (self._build_fact(question_id, session, d) for d in raw_facts) if f]
        return facts, summary

    def _build_fact(self, question_id: str, session: Session, raw: dict) -> Fact | None:
        subject = str(raw.get("subject", "")).strip()
        predicate = str(raw.get("predicate", "")).strip()
        value = str(raw.get("value", "")).strip()
        if not (subject and predicate and value):
            return None
        fact_date = str(raw.get("fact_date", "")).strip()
        if not date_key(fact_date):
            fact_date = session.date
        return Fact(
            id=make_fact_id(question_id, session.session_id, raw),
            question_id=question_id,
            subject=subject,
            predicate=predicate,
            value=value,
            fact_date=fact_date,
            session_id=session.session_id,
            valid_from=fact_date,
        )

    async def ingest_question(
        self, question_id: str, sessions: list[Session], *, skip_if_ingested: bool = True
    ) -> None:
        """Full ingestion for one LongMemEval instance (sessions must be date-sorted)."""
        if skip_if_ingested and await self._store.has_question(question_id):
            log.info("question %s already ingested — skipping", question_id)
            return
        results = await asyncio.gather(
            *[self.extract_session(question_id, s) for s in sessions]
        )
        for session, (facts, summary) in zip(sessions, results, strict=True):
            session.summary = summary
            await self._apply_versioning(question_id, facts)
            await self._store.add_session(question_id, session, facts=facts)
            await self._store.add_facts(question_id, facts)

    async def _apply_versioning(self, question_id: str, new_facts: list[Fact]) -> None:
        """Close validity intervals of currently-valid facts contradicted by new ones."""
        for fact in new_facts:
            candidates = await self._store.find_candidates(
                question_id, fact.subject, fact.predicate, fact.text,
                threshold=self._conflict_threshold,
            )
            for cand in candidates:
                if cand.id == fact.id:
                    continue
                # sessions arrive chronologically; never close a fact with an older one
                if compare_dates(fact.fact_date, cand.fact_date) < 0:
                    continue
                async with self._sem:
                    verdict = await self._llm.chat(
                        prompts.conflict_messages(
                            cand.subject, cand.predicate, cand.value, cand.fact_date,
                            fact.value, fact.fact_date,
                        )
                    )
                if "yes" in verdict.lower():
                    cand.valid_to = fact.fact_date
                    cand.superseded_by = fact.id
                    await self._store.update_fact(cand)
                    log.info("superseded fact %s -> %s", cand.id, fact.id)
