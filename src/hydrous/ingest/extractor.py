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


def llm_confidence(raw: dict) -> float | None:
    """Extraction LLM confidence (0..1) -> float; None when the model didn't score."""
    c = raw.get("confidence")
    if c is None:
        return None
    try:
        score = float(c)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


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
    """Disk cache per (question_id, session_id), keyed on model + extraction-prompt
    version. Old entries (missing or different prompt_version) are cache misses, so
    editing EXTRACTION_SYSTEM invalidates extractions without manual cache wipes."""

    def __init__(self, cache_dir: str | Path, prompt_version: int):
        self._dir = Path(cache_dir) / "extractions"
        self._prompt_version = prompt_version

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
        if data.get("model") != model or data.get("prompt_version") != self._prompt_version:
            return None
        return data.get("facts", []), data.get("summary", "")

    def save(self, question_id: str, session_id: str, model: str, facts: list[dict], summary: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(question_id, session_id)
        payload = {
            "model": model,
            "prompt_version": self._prompt_version,
            "facts": facts,
            "summary": summary,
        }
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
        quality_min_score: float = 0.4,
    ):
        self._llm = llm
        self._store = store
        self._cache = ExtractionCache(cache_dir, prompts.EXTRACTION_PROMPT_VERSION)
        self._model = model_name
        self._sem = asyncio.Semaphore(max_concurrency)
        self._ingest_sem = asyncio.Semaphore(max_concurrency)
        self._conflict_threshold = conflict_threshold
        # Quality gate: facts below this confidence are dropped before storage
        # (the fix for mem0-style junk extraction). The LLM's confidence is the
        # score; it is kept on the stored Fact (quality_score) for audit/reuse.
        self._quality_min_score = quality_min_score

    @property
    def store(self) -> MemoryStore:
        return self._store

    async def extract_session(self, question_id: str, session: Session) -> tuple[list[Fact], str]:
        """Extract facts + summary for one session (cache-first).

        Concurrency is bounded by the caller (`_extract` holds `self._sem`);
        never re-acquire the same semaphore here — nested acquisition deadlocks
        (RUN_NOTES: "nested semaphore")."""
        cached = self._cache.load(question_id, session.session_id, self._model)
        if cached is not None:
            raw_facts, summary = cached
        else:
            facts_resp, summary_resp = await asyncio.gather(
                self._llm.chat(prompts.extraction_messages(session)),
                self._llm.chat(prompts.summary_messages(session)),
            )
            raw_facts, summary = parse_json_array(facts_resp), summary_resp.strip()
            self._cache.save(question_id, session.session_id, self._model, raw_facts, summary)
        facts = [f for f in (self._build_fact(question_id, session, d) for d in raw_facts) if f]
        return facts, summary

    def _build_fact(self, question_id: str, session: Session, raw: dict) -> Fact | None:
        """Build a Fact, applying the quality gate. Returns None for junk/rejects."""
        reject = raw.get("reject") in (True, "true", "yes", "1")
        score = llm_confidence(raw)
        if reject or (score is not None and score < self._quality_min_score):
            log.debug(
                "quality gate dropped fact %r (score=%s, reject=%s)",
                {k: raw.get(k) for k in ("subject", "predicate", "value")},
                score,
                reject,
            )
            return None
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
            quality_score=score,
        )

    async def ingest_question(
        self, question_id: str, sessions: list[Session], *, skip_if_ingested: bool = True
    ) -> None:
        """Full ingestion for one LongMemEval instance (sessions must be date-sorted)."""
        if skip_if_ingested and await self._store.has_question(question_id):
            log.info("question %s already ingested — skipping", question_id)
            return
        async def _extract(session: Session):
            async with self._sem:
                return await self.extract_session(question_id, session)
        results = await asyncio.gather(*[_extract(s) for s in sessions])
        for session, (_facts, summary) in zip(sessions, results, strict=True):
            session.summary = summary

        async def _process(session: Session, facts: list[Fact]) -> None:
            async with self._ingest_sem:
                await self._apply_versioning(question_id, facts)
                await self._store.add_session(question_id, session, facts=facts)
                await self._store.add_facts(question_id, facts)

        # Sessions are versioned+ingested in bounded parallel: the dominant cost
        # is server-side indexing, which pipelines across concurrent ingests.
        await asyncio.gather(*(_process(s, f) for s, (f, _) in zip(sessions, results, strict=True)))

    async def store_facts(
        self, question_id: str, facts: list[Fact], *, session: Session | None = None
    ) -> None:
        """Apply supersession + persist facts (optionally alongside their session).
        `session=None` stores facts only — used by the mem0-compat infer=True path,
        which mirrors mem0 (extracted memories only, no raw conversation)."""
        if not facts:
            return
        await self._apply_versioning(question_id, facts)
        if session is not None:
            await self._store.add_session(question_id, session, facts=facts)
        await self._store.add_facts(question_id, facts)

    async def ingest_session(self, question_id: str, session: Session) -> tuple[list[Fact], str]:
        """Extract (cache-first) + apply supersession + store ONE session with its
        facts. Public single-session entry point (used by the mem0-compat API)."""
        facts, summary = await self.extract_session(question_id, session)
        session.summary = summary
        await self.store_facts(question_id, facts, session=session)
        return facts, summary

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
                        ),
                    )
                if "yes" in verdict.lower():
                    cand.valid_to = fact.fact_date
                    cand.superseded_by = fact.id
                    await self._store.update_fact(cand)
                    log.info("superseded fact %s -> %s", cand.id, fact.id)
