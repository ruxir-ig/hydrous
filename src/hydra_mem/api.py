"""mem0-compatible `Memory` / `AsyncMemory` backed by the hydrous stores.

Contract follows mem0 v1.1+ (`mem0.memory.main`): same method surface, same
argument names, same `{"results": [...]}` / `{"message": ...}` result shapes,
same validation errors (missing entity scope, unknown memory id on delete).
Contract tests in tests/test_mem0_compat.py are lifted from mem0's own README
examples and docstrings — "drop-in" is a tested claim, not a promise.

Deliberate differences / supersets (all documented in README):
- `search(reference_date=...)` WORKS (mem0 OSS raises "platform-only"): it is
  `as_of` time-travel over bi-temporal facts.
- `search` / `get_all` also accept legacy top-level `user_id=` kwargs (mem0 1.1+
  rejects them; we accept with a log note, for old code).
- `chat()` is implemented (mem0 raises NotImplementedError): retrieval + reader,
  exact abstention string when the evidence is empty.
- `rerank`, `categories`, `expiration_date` (infer=True), `memory_type`,
  `prompt`, non-entity metadata filters: accepted and logged, not implemented.
- Scores are rank-normalized to [0, 1] (best hit = 1.0) — hydrous uses RRF, not
  cosine similarity; the raw RRF score is exposed via `score_details` (explain).
- Memory ids resolve within the process that created or searched them (they map
  back to the owning scope bucket).

Scope mapping: mem0 entities -> a hydrous question_id bucket
(`mem0_u_alice_a_bob`), so each user/agent/run owns an isolated collection
(local sqlite rows, or one HydraDB collection).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from hydrous.answer.reader import Answerer
from hydrous.config import Settings, build_llm
from hydrous.embeddings import TfidfEmbedder
from hydrous.ingest.extractor import FactExtractor
from hydrous.llm import LLMClient
from hydrous.retrieve.pipeline import RetrievalPipeline
from hydrous.schema import Session, Turn
from hydrous.store.base import MemoryStore
from hydrous.text_utils import tokenize

log = logging.getLogger(__name__)

_ENTITY_KEYS = ("user_id", "agent_id", "run_id")
_UNSET = object()

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _content_hash(text: str) -> str:
    """Feedback-loop tag: hash of normalized content (= the "tag" search puts on
    results, compared against on re-add)."""
    norm = " ".join(str(text).split()).casefold()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _norm_key(s: str) -> str:
    return " ".join(tokenize(s))


def _date_part(value: Any) -> str:
    """Any date-ish value -> 'YYYY-MM-DD' ('' when unparseable)."""
    s = str(value or "").strip()
    if not s:
        return ""
    d = s[:10]
    try:
        datetime.strptime(d, "%Y-%m-%d")
        return d
    except ValueError:
        return ""


def _to_iso(value: Any) -> str:
    """mem0 created_at -> ISO-ish string (best effort; never None)."""
    s = str(value or "").strip()
    if not s:
        return datetime.now(UTC).isoformat()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return s


_ROLE_PREFIX_RE = re.compile(r"^(USER|ASSISTANT|SYSTEM):\s*", re.IGNORECASE)


def _strip_role_prefix(text: str) -> str:
    """Session.text renders turns as 'USER: content' lines; mem0 memories are the
    bare content, so strip the role prefixes when returning "memory" text."""
    out = []
    for line in str(text).splitlines():
        m = _ROLE_PREFIX_RE.match(line)
        out.append(line[m.end() :] if m else line)
    return "\n".join(out)


def _messages_to_turns(messages: Any) -> list[Turn]:
    """Normalize mem0 `messages` (str or [{'role','content'}, ...]) to turns.
    Skips malformed entries and system messages, mirroring mem0's raw path."""
    if isinstance(messages, str):
        return [Turn(role="user", content=messages)]
    if not isinstance(messages, list):
        raise ValueError("messages must be a string or a list of {'role': ..., 'content': ...}")
    turns: list[Turn] = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") is None or m.get("content") is None:
            log.warning("skipping invalid message format: %s", m)
            continue
        if m["role"] == "system":
            continue
        turns.append(Turn(role=str(m["role"]), content=str(m["content"])))
    return turns


class _Scope:
    """Entity scope -> hydrous question_id bucket. At least one entity required
    (mem0 raises the same ValueError otherwise)."""

    def __init__(
        self,
        *,
        user_id: Any = None,
        agent_id: Any = None,
        run_id: Any = None,
        filters: dict | None = None,
        kwargs: dict | None = None,
        what: str = "operation",
    ):
        provided: dict[str, Any] = {}
        entity_defaults = {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}
        for src in (filters or {}, kwargs or {}, entity_defaults):
            provided.update({k: v for k, v in src.items() if k in _ENTITY_KEYS and v is not None})
        ids = {k: str(v).strip() for k, v in provided.items() if str(v).strip()}
        if not ids:
            raise ValueError(
                f"{what} requires at least one of user_id, agent_id, run_id "
                "(e.g. filters={{'user_id': 'u1'}})"
            )
        self.ids = ids
        raw = "mem0_" + "_".join(f"{k[:1]}_{v}" for k, v in sorted(ids.items()))
        # sanitized so the bucket is a safe collection/id key everywhere
        self.question_id = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
        self.extra_filters = {k: v for k, v in (filters or {}).items() if k not in _ENTITY_KEYS}
        self.top_level_kwargs = {k: v for k, v in (kwargs or {}).items() if k not in _ENTITY_KEYS}


class _IdRegistry:
    """Persistent memory-id -> scope map (a JSON sidecar file).

    mem0 resolves ids from storage, so `delete()`/`update()`/`get()`/`history()`
    keep working across process restarts here too: every scope-bounded operation
    records the ids it created/returned, persisted atomically on mutation, and
    instances pointing at the same store share the file. Scope entity ids are
    kept alongside so `get()` can reconstruct mem0's user_id/agent_id/run_id
    item fields without a scope-building call.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._ids: dict[str, str] = {}
        self._scopes: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        raw_ids = data.get("ids")
        if isinstance(raw_ids, dict):
            self._ids = {str(k): str(v) for k, v in raw_ids.items()}
        raw_scopes = data.get("scopes")
        if isinstance(raw_scopes, dict):
            self._scopes = {str(k): v for k, v in raw_scopes.items() if isinstance(v, dict)}

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps({"ids": self._ids, "scopes": self._scopes}, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError:
            log.warning("failed to persist memory-id registry at %s", self._path, exc_info=True)

    def set(self, memory_id: str, scope: str, entities: dict[str, str] | None = None) -> None:
        changed = self._ids.get(memory_id) != scope
        self._ids[memory_id] = scope
        if entities:
            if self._scopes.get(scope) != entities:
                self._scopes[scope] = dict(entities)
                changed = True
        if changed:
            self.save()

    def get(self, memory_id: str) -> str | None:
        return self._ids.get(memory_id)

    def pop(self, memory_id: str) -> None:
        if memory_id in self._ids:
            del self._ids[memory_id]
            self.save()

    def drop_scope(self, scope: str) -> set[str]:
        removed = {k for k, v in self._ids.items() if v == scope}
        if not removed:
            return set()
        for k in removed:
            del self._ids[k]
        self._scopes.pop(scope, None)
        self.save()
        return removed

    def scopes(self) -> set[str]:
        return set(self._ids.values()) | set(self._scopes)

    def entity_ids(self, scope: str) -> dict[str, str]:
        return dict(self._scopes.get(scope) or {})

    def clear(self) -> None:
        if not self._ids and not self._scopes:
            return
        self._ids.clear()
        self._scopes.clear()
        self.save()


# ---------------------------------------------------------------------------
# AsyncMemory
# ---------------------------------------------------------------------------


class AsyncMemory:
    """mem0-compatible async memory. Same surface as mem0.AsyncMemory."""

    def __init__(
        self,
        *,
        store: str = "local",
        store_path: str | None = None,
        api_key: str | None = None,
        database: str | None = None,
        base_url: str | None = None,
        llm: LLMClient | None = None,
        embedder: Any | None = None,
        cache_dir: str | None = None,
        model_name: str | None = None,
        quality_min_score: float | None = None,
        top_k: int = 20,
        use_gate: bool = False,
        ids_path: str | None = None,
    ):
        self._store_backend = store
        self._store_path = store_path
        self._api_key = api_key
        self._database = database
        self._base_url = base_url
        self._llm = llm
        self._embedder = embedder
        self._cache_dir = cache_dir
        self._model_name = model_name
        self._quality_min_score = quality_min_score
        self._top_k = top_k
        self._use_gate = use_gate
        self._settings = Settings()

        self._store: MemoryStore | None = None
        self._extractor: FactExtractor | None = None
        self._answerer: Answerer | None = None
        # feedback-loop prevention: content hash -> memory id ("tag" from search)
        self._seen: dict[str, str] = {}
        # memory id -> scope bucket, PERSISTED so update/delete/get/history work
        # across process restarts (mem0 resolves ids from storage, not memory).
        self._registry = _IdRegistry(ids_path or self._default_ids_path())
        self._scopes: set[str] = set()

    def _default_ids_path(self) -> str:
        if self._store_backend == "hydradb":
            return str(self._settings.data_dir / "hydra_mem_ids.json")
        return str(Path(self._store_path or self._settings.store_path)) + ".ids.json"

    # ------------------------------------------------------------- internals

    async def _ensure_store(self) -> MemoryStore:
        if self._store is None:
            s = self._settings
            if self._store_backend == "hydradb":
                from hydrous.store.hydradb import HydraDBStore

                key = self._api_key or s.hydra_db_api_key
                if not key:
                    raise RuntimeError(
                        "hydradb store requires api_key=... or HYDRA_DB_API_KEY in .env"
                    )
                self._store = HydraDBStore(
                    key,
                    database=self._database or s.hydra_db_database,
                    base_url=self._base_url or s.hydra_db_base_url,
                )
            else:
                from hydrous.store.local import LocalMemoryStore

                embedder = self._embedder
                if embedder is None:
                    try:
                        from hydrous.config import build_default_embedder

                        embedder = build_default_embedder(s)
                    except Exception:
                        embedder = TfidfEmbedder()
                        log.warning("default embedder unavailable; using tfidf", exc_info=True)
                self._store = LocalMemoryStore(self._store_path or s.store_path, embedder)
        return self._store

    def _get_llm(self) -> LLMClient:
        if self._llm is not None:
            return self._llm
        return build_llm(self._settings)  # raises a clear error naming the env var

    async def _get_extractor(self) -> FactExtractor:
        if self._extractor is None:
            llm = self._get_llm()
            self._extractor = FactExtractor(
                llm,
                await self._ensure_store(),
                self._cache_dir or self._settings.cache_dir,
                model_name=self._model_name or getattr(llm, "model", "unknown"),
                quality_min_score=(
                    self._quality_min_score
                    if self._quality_min_score is not None
                    else self._settings.quality_min_score
                ),
            )
        return self._extractor

    @staticmethod
    def _log_ignored(**params: Any) -> None:
        for name, value in params.items():
            if value not in (None, False, {}, []):
                log.info("%s: %r accepted — not implemented yet (fed back to hydrous)", name, value)

    def _find_scope(self, memory_id: str) -> str | None:
        return self._registry.get(memory_id)

    def _scope_of(self, memory_id: str) -> str:
        qid = self._find_scope(memory_id)
        if qid is None:
            raise ValueError(f"Memory with id {memory_id} not found")
        return qid

    # --------------------------------------------------------------- add

    async def add(
        self,
        messages: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict | None = None,
        timestamp: Any = None,
        expiration_date: Any = None,
        infer: bool = True,
        memory_type: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        """Create memory(ies). `infer=False` stores raw messages with NO LLM call
        (the cheap/fast path, one memory per non-system message). `infer=True`
        extracts quality-gated facts and stores those (mem0 parity: the raw
        conversation is not stored). Returns mem0's `{"results": [...]}` shape
        with `event` ADD or NOOP (feedback-loop / duplicate)."""
        if timestamp is not None:
            raise ValueError(
                "timestamp is a platform-only parameter and is not supported in OSS (same as mem0)."
            )
        self._log_ignored(memory_type=memory_type, prompt=prompt)
        scope = _Scope(user_id=user_id, agent_id=agent_id, run_id=run_id, what="add")
        self._scopes.add(scope.question_id)
        meta = dict(metadata or {})
        if expiration_date is not None:
            exp = _date_part(expiration_date)
            if exp:
                meta["expiration_date"] = exp  # stored in metadata; not yet enforced

        turns = _messages_to_turns(messages)
        if not turns:
            return {"results": []}

        store = await self._ensure_store()

        if not infer:
            results = []
            for t in turns:
                h = _content_hash(t.content)
                if h in self._seen:  # re-adding something search() returned -> no-op
                    results.append(self._noop(self._seen[h], t.content))
                    continue
                session = Session(
                    session_id="raw_" + uuid4().hex[:16],
                    date=date.today().isoformat(),
                    turns=[t],
                    metadata={**meta, "role": t.role},
                )
                await store.add_session(scope.question_id, session)
                self._registry.set(session.session_id, scope.question_id, scope.ids)
                self._seen[h] = session.session_id
                results.append(
                    {
                        "id": session.session_id,
                        "memory": t.content,
                        "event": "ADD",
                        "actor_id": None,
                        "role": t.role,
                        "metadata": session.metadata,
                    }
                )
            return {"results": results}

        # infer=True: extraction (LLM) -> quality gate -> versioning -> facts only
        extractor = await self._get_extractor()
        session = Session(
            session_id="conv_" + uuid4().hex[:16],
            date=date.today().isoformat(),
            turns=turns,
        )
        try:
            facts, _summary = await extractor.extract_session(scope.question_id, session)
        except Exception:
            log.exception("add(infer=True): extraction failed; nothing stored")
            return {"results": []}
        if expiration_date is not None:
            self._log_ignored(expiration_date=expiration_date)  # infer path: stored nowhere
        if not facts:
            return {"results": []}

        existing: dict[tuple[str, str, str], str] = {}
        for f in await store.get_facts(scope.question_id):
            existing[(_norm_key(f.subject), _norm_key(f.predicate), _norm_key(f.value))] = f.id
        fresh: list = []
        results = []
        for fact in facts:
            key = (_norm_key(fact.subject), _norm_key(fact.predicate), _norm_key(fact.value))
            old = existing.get(key) or self._seen.get(_content_hash(fact.text))
            if old is not None:
                results.append(self._noop(old, fact.text))
                continue
            fresh.append(fact)
            existing[key] = fact.id
            results.append(
                {
                    "id": fact.id,
                    "memory": fact.text,
                    "event": "ADD",
                    "actor_id": None,
                    "role": None,
                    "metadata": {
                        "kind": "fact",
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "quality_score": fact.quality_score,
                    },
                }
            )
        if fresh:
            await extractor.store_facts(scope.question_id, fresh, session=None)
            for f in fresh:
                self._registry.set(f.id, scope.question_id, scope.ids)
        return {"results": results}

    @staticmethod
    def _noop(memory_id: str, memory: str) -> dict:
        return {
            "id": memory_id or "",
            "memory": memory,
            "event": "NOOP",
            "actor_id": None,
            "role": None,
            "metadata": {"from_search": True},
        }

    # ------------------------------------------------------------ search

    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict | None = None,
        threshold: float = 0.1,
        rerank: bool = False,
        explain: bool = False,
        reference_date: Any = None,
        show_expired: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Hybrid search. `reference_date` is implemented time-travel (as-of);
        `rerank`/`categories` (kwargs) are accepted and logged; `explain` adds
        `score_details` with the raw RRF score. Scores are rank-normalized to
        [0, 1]; `threshold` floors them. Results are tagged so re-`add`ing them
        is a NOOP."""
        if not query or not str(query).strip():
            raise ValueError("query must be a non-empty string")
        if int(top_k) <= 0:
            raise ValueError("top_k must be a positive integer")
        scope = _Scope(filters=filters, kwargs=kwargs, what="search")
        self._scopes.add(scope.question_id)
        if scope.top_level_kwargs:
            log.info(
                "search: top-level entity kwargs %s accepted (legacy form); filters= is mem0 1.1+'s way",
                sorted(scope.top_level_kwargs),
            )
        self._log_ignored(
            rerank=rerank,
            categories=kwargs.get("categories"),
            show_expired=show_expired,
            metadata_filters=scope.extra_filters,
        )

        store = await self._ensure_store()
        as_of = _date_part(reference_date) or None  # reference_date = as-of time travel
        evs = await store.search(
            scope.question_id,
            str(query).strip(),
            as_of_date=as_of,
            top_k=max(int(top_k), 1),
            include_superseded=False,
        )
        if not evs:
            return {"results": []}

        ranked = [
            (
                ev,
                ev.fact_id if ev.source == "fact" else ev.session_id,
                _strip_role_prefix(ev.text),
            )
            for ev in evs
        ]
        raw = [ev.score for ev, _, _ in ranked]
        lo = min(raw)
        hi = max(raw)

        out = []
        for ev, mem_id, text in ranked:
            if not mem_id:
                continue
            # Rank-map to [0.1, 1.0] (best hit = 1.0): with the default threshold
            # 0.1 every returned hit survives — mem0's default rarely trims either.
            # The LOWEST hit maps to 0.1 (not 0.0), so a 2-result search returns 2;
            # raising the threshold trims the weak tail. min-max is RELATIVE, not a
            # cosine similarity: scores are rank-normalized RRF, and explain=True
            # exposes the raw RRF value in score_details.
            norm = 1.0 if hi == lo else 0.1 + 0.9 * (ev.score - lo) / (hi - lo)
            if threshold and threshold > 0 and norm < threshold:
                continue
            item: dict[str, Any] = {
                "id": mem_id,
                "memory": text,
                "hash": _content_hash(text),
                "metadata": {"kind": "fact" if ev.source == "fact" else "session", "from_search": True},
                "score": round(norm, 4),
                "created_at": ev.date,
                "updated_at": ev.date,
                **scope.ids,
            }
            if explain:
                item["score_details"] = {"rrf": float(ev.score)}
            self._seen[_content_hash(text)] = mem_id  # the feedback-loop tag
            self._registry.set(mem_id, scope.question_id, scope.ids)
            out.append(item)
        return {"results": out}

    # ------------------------------------------------------------ get_all

    async def get_all(
        self, *, filters: dict | None = None, top_k: int = 20, show_expired: bool = False, **kwargs: Any
    ) -> dict:
        """List all memories in a scope (facts + raw sessions), newest first.
        Superseded facts are included, marked via metadata (full history —
        mem0 cannot show old versions at all)."""
        scope = _Scope(filters=filters, kwargs=kwargs, what="get_all")
        self._log_ignored(show_expired=show_expired)
        store = await self._ensure_store()
        items: list[dict] = []
        for f in await store.get_facts(scope.question_id):
            items.append(
                {
                    "id": f.id,
                    "memory": f.text,
                    "hash": _content_hash(f.text),
                    "metadata": {
                        "kind": "fact",
                        "subject": f.subject,
                        "predicate": f.predicate,
                        "value": f.value,
                        "session_id": f.session_id,
                        "quality_score": f.quality_score,
                        "valid_from": f.valid_from,
                        "valid_to": f.valid_to,
                        "superseded_by": f.superseded_by,
                    },
                    "created_at": f.recorded_at,
                    "updated_at": f.recorded_at,
                    **scope.ids,
                }
            )
        for s in await store.get_sessions(scope.question_id):
            items.append(
                {
                    "id": s.session_id,
                    "memory": _strip_role_prefix(s.text),
                    "hash": _content_hash(_strip_role_prefix(s.text)),
                    "metadata": {**s.metadata, "kind": "session"},
                    "created_at": s.recorded_at,
                    "updated_at": s.recorded_at,
                    **scope.ids,
                }
            )
        items.sort(key=lambda i: str(i.get("created_at") or ""), reverse=True)
        for item in items:  # refuse to lose track of ids -> delete/update work later
            self._registry.set(item["id"], scope.question_id, scope.ids)
        return {"results": items[: max(int(top_k), 0)]}

    # ------------------------------------------------------------ get

    async def get(self, memory_id: str) -> dict | None:
        """Retrieve one memory by id. mem0 parity: returns None for unknown ids
        (delete/update are the ones that raise)."""
        qid = self._find_scope(memory_id)
        if qid is None:
            return None
        store = await self._ensure_store()
        entities = self._registry.entity_ids(qid)
        for f in await store.get_facts(qid):
            if f.id == memory_id:
                return {
                    "id": f.id,
                    "memory": f.text,
                    "hash": _content_hash(f.text),
                    "created_at": f.recorded_at,
                    "updated_at": f.recorded_at,
                    **entities,
                    "metadata": {
                        "kind": "fact",
                        "subject": f.subject,
                        "predicate": f.predicate,
                        "value": f.value,
                        "session_id": f.session_id,
                        "quality_score": f.quality_score,
                        "valid_from": f.valid_from,
                        "valid_to": f.valid_to,
                        "superseded_by": f.superseded_by,
                    },
                }
        for s in await store.get_sessions(qid):
            if s.session_id == memory_id:
                return {
                    "id": s.session_id,
                    "memory": _strip_role_prefix(s.text),
                    "hash": _content_hash(_strip_role_prefix(s.text)),
                    "created_at": s.recorded_at,
                    "updated_at": s.recorded_at,
                    **entities,
                    "metadata": {**s.metadata, "kind": "session"},
                }
        # registered but gone (deleted from another instance) — drop the stale entry
        self._registry.pop(memory_id)
        return None

    @staticmethod
    def _history_record(
        memory_id: str,
        old_memory: str | None,
        new_memory: str | None,
        event: str,
        *,
        created_at: str,
        updated_at: str,
    ) -> dict:
        """One mem0 `history()` record — same field names as mem0's history table."""
        return {
            "id": uuid4().hex,
            "memory_id": memory_id,
            "old_memory": old_memory,
            "new_memory": new_memory,
            "event": event,
            "created_at": created_at,
            "updated_at": updated_at,
            "is_deleted": False,
            "actor_id": None,
            "role": None,
        }

    async def history(self, memory_id: str) -> list[dict]:
        """mem0 `history(memory_id)`: changes for a memory, newest-last (empty
        list when unknown — mem0 parity). For facts this walks the FULL
        supersession chain of the fact's (subject, predicate) slot in both
        directions: every version gets an ADD record, every superseded version
        an UPDATE record pointing at its successor — strictly richer than
        mem0's flat previous_value/new_value log."""
        qid = self._find_scope(memory_id)
        if qid is None:
            return []
        store = await self._ensure_store()
        sessions = {s.session_id: s for s in await store.get_sessions(qid)}
        if memory_id in sessions:
            sess = sessions[memory_id]
            return [
                self._history_record(
                    memory_id, None, _strip_role_prefix(sess.text), "ADD",
                    created_at=sess.recorded_at, updated_at=sess.recorded_at,
                )
            ]
        fact = next((f for f in await store.get_facts(qid) if f.id == memory_id), None)
        if fact is None:
            return []
        timeline = await store.get_timeline(qid, fact.subject, fact.predicate)
        by_id = {f.id: f for f in timeline}
        records = []
        for ver in timeline:
            records.append(
                self._history_record(
                    ver.id, None, ver.text, "ADD",
                    created_at=ver.recorded_at, updated_at=ver.recorded_at,
                )
            )
            succ = by_id.get(ver.superseded_by) if ver.superseded_by else None
            if succ is not None:
                records.append(
                    self._history_record(
                        ver.id, ver.text, succ.text, "UPDATE",
                        created_at=succ.recorded_at, updated_at=succ.recorded_at,
                    )
                )
        records.sort(key=lambda r: (str(r["created_at"] or ""), r["event"]))
        return records

    # ------------------------------------------------------------ update

    async def update(
        self,
        memory_id: str,
        text: str | None = None,
        metadata: dict | None = None,
        expiration_date: Any = _UNSET,
        data: str | None = None,
    ) -> dict:
        """Replace a raw memory's text / merge metadata. `data` is the deprecated
        mem0 alias for `text`. Extracted-fact updates raise NotImplementedError
        (delete + re-add instead)."""
        if data is not None:
            log.warning("update(data=...) is deprecated — use text=... (same as mem0)")
            if text is None:
                text = data
        if text is None and metadata is None and expiration_date is _UNSET:
            raise ValueError("At least one of text, metadata, or expiration_date must be provided.")

        qid = self._scope_of(memory_id)
        store = await self._ensure_store()

        sessions = {s.session_id: s for s in await store.get_sessions(qid)}
        if memory_id in sessions:
            sess = sessions[memory_id]
            new_meta = dict(sess.metadata)
            if metadata:
                new_meta.update(metadata)
            if expiration_date is not _UNSET:
                exp = _date_part(expiration_date)
                if exp:
                    new_meta["expiration_date"] = exp
                else:
                    new_meta.pop("expiration_date", None)
            role = sess.turns[0].role if sess.turns else "user"
            new_turns = sess.turns if text is None else [Turn(role=role, content=text)]
            updated = Session(
                session_id=sess.session_id,
                date=sess.date,
                summary=sess.summary,
                recorded_at=sess.recorded_at,
                metadata=new_meta,
                turns=new_turns,
            )
            await store.add_session(qid, updated)
            return {"message": "Memory updated successfully!"}

        for f in await store.get_facts(qid):
            if f.id == memory_id:
                raise NotImplementedError(
                    "updating an extracted fact is not supported yet — delete it and re-add "
                    "(bi-temporal supersession already handles 'replaced' facts)"
                )
        self._registry.pop(memory_id)  # stale entry: registered but no longer stored
        raise ValueError(f"Memory with id {memory_id} not found")

    # ------------------------------------------------------------ delete

    async def delete(self, memory_id: str) -> dict:
        """Delete one memory by id; raises ValueError (mem0 parity) if unknown.
        NOTE: this is an EXPLICIT user-initiated delete and is honored as a hard
        delete. Append-only only means supersession never SILENTLY deletes."""
        qid = self._scope_of(memory_id)
        store = await self._ensure_store()
        exists = any(f.id == memory_id for f in await store.get_facts(qid)) or any(
            s.session_id == memory_id for s in await store.get_sessions(qid)
        )
        if not exists:
            self._registry.pop(memory_id)  # stale entry
            raise ValueError(f"Memory with id {memory_id} not found")
        await store.delete_memory(qid, memory_id)
        self._registry.pop(memory_id)
        self._seen = {h: i for h, i in self._seen.items() if i != memory_id}
        return {"message": "Memory deleted successfully!"}

    async def delete_all(
        self, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None
    ) -> dict:
        """Delete every memory in one scope (mem0 signature: top-level entity ids)."""
        scope = _Scope(user_id=user_id, agent_id=agent_id, run_id=run_id, what="delete_all")
        await (await self._ensure_store()).delete_question(scope.question_id)
        self._scopes.discard(scope.question_id)
        removed = self._registry.drop_scope(scope.question_id)
        self._seen = {h: i for h, i in self._seen.items() if i not in removed}
        return {"message": "Memories deleted successfully!"}

    async def reset(self) -> dict:
        """Reset the whole store: every scope known to the persisted id registry
        (including scopes tracked by other instances/processes), not just this
        instance's — mem0's wipe-everything semantics."""
        store = await self._ensure_store()
        scopes = self._scopes | self._registry.scopes()
        for qid in scopes:
            await store.delete_question(qid)
        self._seen.clear()
        self._registry.clear()
        self._scopes.clear()
        return {"message": "Memory reset successfully!"}

    # ------------------------------------------------------------ chat

    async def chat(self, query: str, *, filters: dict | None = None, **kwargs: Any) -> str:
        """Answer a question from memory (mem0 raises NotImplementedError here).
        Retrieval + reader with the exact abstention string when the retrieved
        evidence cannot answer — no hallucination."""
        scope = _Scope(filters=filters, kwargs=kwargs, what="chat")
        self._scopes.add(scope.question_id)
        llm = self._get_llm()
        store = await self._ensure_store()
        if self._answerer is None:
            self._answerer = Answerer(
                llm, RetrievalPipeline(llm, store, top_k=self._top_k), use_gate=self._use_gate
            )
        result = await self._answerer.answer(scope.question_id, query, date.today().isoformat())
        return result.answer

    # ------------------------------------------------------------ import

    async def import_mem0(
        self,
        items: list[dict],
        *,
        infer: bool = False,
        default_user_id: str = "default",
    ) -> dict:
        """Replay a mem0 `get_all` export (`{"results": [{id, memory, created_at,
        user_id, ...}]}`), preserving `created_at -> recorded_at`. Raw by default
        (exact replay, no LLM); `infer=True` re-extracts facts instead."""
        store = await self._ensure_store()
        stats: dict[str, int] = {"imported": 0, "noop": 0, "skipped": 0}
        for raw in items:
            if not isinstance(raw, dict) or not raw.get("memory"):
                stats["skipped"] += 1
                continue
            scope = _Scope(
                user_id=raw.get("user_id") or default_user_id,
                agent_id=raw.get("agent_id"),
                run_id=raw.get("run_id"),
                what="import",
            )
            created = raw.get("created_at") or raw.get("updated_at")
            recorded = _to_iso(created)
            sdate = _date_part(created) or date.today().isoformat()
            meta = dict(raw.get("metadata") or {})
            meta.setdefault("mem0_id", str(raw.get("id") or ""))
            text = str(raw["memory"])
            h = _content_hash(text)

            if not infer:
                if h in self._seen:
                    stats["noop"] += 1
                    continue
                session = Session(
                    session_id="raw_" + uuid4().hex[:16],
                    date=sdate,
                    turns=[Turn(role="user", content=text)],
                    recorded_at=recorded,
                    metadata={**meta, "role": "user"},
                )
                await store.add_session(scope.question_id, session)
                self._registry.set(session.session_id, scope.question_id, scope.ids)
                self._seen[h] = session.session_id
                self._scopes.add(scope.question_id)
                stats["imported"] += 1
                continue

            extractor = await self._get_extractor()
            session = Session(
                session_id="conv_" + uuid4().hex[:16],
                date=sdate,
                turns=[Turn(role="user", content=text)],
            )
            try:
                facts, _ = await extractor.extract_session(scope.question_id, session)
            except Exception:
                log.exception("import_mem0: extraction failed for %r", raw.get("id"))
                stats["skipped"] += 1
                continue
            if not facts:
                stats["noop"] += 1
                continue
            for f in facts:
                f.recorded_at = recorded  # created_at -> recorded_at
            await extractor.store_facts(scope.question_id, facts, session=None)
            for f in facts:
                self._registry.set(f.id, scope.question_id, scope.ids)
            self._scopes.add(scope.question_id)
            stats["imported"] += len(facts)
        return {"scopes": sorted(self._scopes), **stats}

    async def close(self) -> None:
        """Release the underlying store (sqlite connection / cloud client)."""
        store, self._store = self._store, None
        if store is not None:
            try:
                store.close()
            except Exception:
                log.warning("hydra-mem store close failed", exc_info=True)


# ---------------------------------------------------------------------------
# Memory (sync facade)
# ---------------------------------------------------------------------------


class Memory:
    """mem0-compatible synchronous `Memory`.

    A single daemon event loop thread owns the underlying async store/LLM, so
    per-loop state (sqlite connections, AsyncHydraDB clients) stays on one loop.
    Thread-safe for concurrent callers; use `with Memory(...) as m:` or call
    `m.close()` when done.
    """

    def __init__(self, **kwargs: Any):
        self._async = AsyncMemory(**kwargs)
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="hydra-mem-loop", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()  # propagates exceptions to the caller

    def add(self, messages: Any, **kwargs: Any) -> dict:
        return self._run(self._async.add(messages, **kwargs))

    def search(self, query: str, **kwargs: Any) -> dict:
        return self._run(self._async.search(query, **kwargs))

    def get_all(self, **kwargs: Any) -> dict:
        return self._run(self._async.get_all(**kwargs))

    def get(self, memory_id: str) -> dict | None:
        return self._run(self._async.get(memory_id))

    def history(self, memory_id: str) -> list[dict]:
        return self._run(self._async.history(memory_id))

    def update(self, memory_id: str, **kwargs: Any) -> dict:
        return self._run(self._async.update(memory_id, **kwargs))

    def delete(self, memory_id: str) -> dict:
        return self._run(self._async.delete(memory_id))

    def delete_all(
        self, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None
    ) -> dict:
        return self._run(self._async.delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id))

    def reset(self) -> dict:
        return self._run(self._async.reset())

    def chat(self, query: str, **kwargs: Any) -> str:
        return self._run(self._async.chat(query, **kwargs))

    def import_mem0(self, items: list[dict], **kwargs: Any) -> dict:
        return self._run(self._async.import_mem0(items, **kwargs))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._run(self._async.close())  # release the store on its own loop
        except Exception:
            log.warning("hydra-mem close: async close failed", exc_info=True)
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass
        self._thread.join(timeout=5.0)

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
