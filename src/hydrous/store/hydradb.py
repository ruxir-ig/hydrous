"""HydraDB cloud adapter implementing MemoryStore (optional extra: `hydra`).

Mapping:
- One collection per LongMemEval instance: `qid_{question_id}`.
- Facts ingest with `id` as upsert key; the full Fact is reconstructable from
  `additional_metadata` (subject/predicate/value/fact_date/session_id/valid_from/
  valid_to/superseded_by). Supersession = `update_source_metadata` PATCH
  (valid_to + superseded_by) on the same id — no re-ingest, stable id, and the
  probe verified patched keys sync to the vector store (query-time
  `metadata_filters` on patched keys match).
- Sessions ingest as `user_assistant_pairs` (the server requires text XOR pairs
  and caps `additional_metadata` at 1024 bytes, so full turns cannot live in
  metadata). `get_sessions` reconstructs turns from `context.inspect`, which
  derives the content from pairs ("USER:/ASSISTANT:" lines for a single pair,
  a `{"pairs": [...]}` JSON envelope for multi-pair items). Long sessions are
  split into parts (`{qid}:{sid}`, `{qid}:{sid}#1`, ...) to stay under the
  per-request ingest cost budget; parts are ordered via `additional_metadata.part`.
- Ingest cost limits (probe-verified): ~1000 "memory_tokens" per request and a
  rolling 5000/min budget. `_ingest_items` chunks payloads by estimated cost and
  paces concurrent ingests through a token window; 429s are retried.
- Query: query_by="hybrid", mode="thinking", recency_bias=0.4, graph_context=True.
  Verified metadata_filters grammar: top-level keys match tenant `metadata`;
  `{"additional_metadata": {...}}` matches the additional map; array values are
  set-equality; `operator` is only valid with query_by=text. valid_as_of is
  applied client-side over chunk `additional_metadata` (server-side is possible
  now that the grammar is verified, but interval semantics are date comparisons
  a set-equality filter can't express).
- context.list wire shape differs from the SDK types (items under
  `data.user_memories[]` with `memory_id`; pagination/total at data level), so
  the adapter reads the raw HTTP body there.
- The SDK's MemoryItem.metadata is typed `str`, but the v2 server rejects strings
  for it (400 "cannot unmarshal string into ... metadata of type
  map[string]interface{}") — ingest payloads are built as plain dicts.

Integration tests run only when HYDRA_DB_API_KEY is set. Without the SDK installed,
constructing HydraDBStore raises a clear error.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import time

from ..schema import Evidence, Fact, Session, Turn
from ..text_utils import compare_dates, tokenize
from .base import MemoryStore

_RRF_K = 60
_FACT_LANE_WEIGHT = 0.4


def _content_to_text(text: str) -> str:
    """Normalize inspect content to plain "USER:/ASSISTANT:" text (both formats)."""
    turns = _parse_content_turns(text)
    if turns:
        return "\n".join(f"{'USER' if t.role == 'user' else 'ASSISTANT'}: {t.content}" for t in turns)
    return (text or "").strip()


_SESSION_TEXT_CACHE: dict[tuple[str, str], str] = {}


def _ranks(scores: list[float]) -> list[int]:
    """rank 0 = highest score; ties broken by index (deterministic)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


log = logging.getLogger(__name__)

_SDK_HELP = "HydraDB SDK not installed. Run: uv sync --extra hydra  (or pip install 'hydradb-sdk>=2,<3')"

# HydraDB chunks one memory into `{memory_id}_chunk_{seq}` chunk uuids.
_CHUNK_ID_RE = re.compile(r"^(.*?)_chunk_\d+$")
# context.inspect content is our own "ROLE: content" serialization.
_CONTENT_TURN_RE = re.compile(r"^(USER|ASSISTANT|SYSTEM): (.*)$", re.IGNORECASE)

_MAX_INGEST_BATCH = 100  # sources per ingest / status / delete call (server caps page_size at 100)


def fact_to_memory_item(fact: Fact) -> dict:
    """Fact -> wire MemoryItem dict (metadata is a map; the SDK's str type is stale)."""
    return {
        "id": fact.id,
        "title": f"{fact.subject} {fact.predicate}",
        "text": fact.text,
        "infer": True,
        "metadata": {"question_id": fact.question_id, "kind": "fact"},
        "additional_metadata": {
            "question_id": fact.question_id,
            "kind": "fact",
            "session_id": fact.session_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "value": fact.value,
            "fact_date": fact.fact_date,
            "valid_from": fact.valid_from,
            "valid_to": fact.valid_to,
            "superseded_by": fact.superseded_by,
        },
    }


_MAX_INGEST_BATCH = 100  # sources per ingest / status / delete call (server caps page_size at 100)
# Ingest cost model (empirical): ~5 chars of pair text per cost unit, budget 700
# leaves headroom under the server's 1000 per-request "memory_tokens" limit.
_PAIR_COST = 7  # chars per cost unit (server charges real tokens, ~7 chars/token for English)
_PAIR_MIN_COST = 60
_INGEST_BUDGET = 600  # per-request "memory_tokens" budget (server limit: 1000)
_PER_MIN_BUDGET = 4500  # rolling 60s "memory_tokens" budget (server limit: 5000/min)
_RATE_RETRIES = 6
_MAX_PAIR_CHARS = 4000  # cap a single pair so one part always fits one request


def _pair_cost(pair: dict[str, str]) -> int:
    size = len(str(pair.get("user", ""))) + len(str(pair.get("assistant", "")))
    return max(_PAIR_MIN_COST, (size + _PAIR_COST // 2) // _PAIR_COST)


def _item_cost(item: dict) -> int:
    pairs = item.get("user_assistant_pairs") or []
    if pairs:
        return sum(_pair_cost(p) for p in pairs)
    text = str(item.get("text") or "")
    return max(_PAIR_MIN_COST, (len(text) + _PAIR_COST // 2) // _PAIR_COST)


def session_to_memory_items(question_id: str, session: Session) -> list[dict]:
    """Session -> one or more wire MemoryItem dicts. The server requires text XOR
    user_assistant_pairs, caps additional_metadata at 1024 bytes, and limits each
    ingest request to ~1000 "memory_tokens", so long sessions are split into
    parts: `{qid}:{sid}`, `{qid}:{sid}#1`, ... ordered via additional_metadata.part.
    context.inspect derives the pair text back for get_sessions reconstruction.
    A trailing user turn without an assistant reply is dropped from pairs
    (0.03% of LongMemEval sessions)."""
    pairs: list[dict[str, str]] = []
    pending_user: str | None = None
    for turn in session.turns:
        if turn.role == "user":
            pending_user = turn.content
        elif turn.role == "assistant" and pending_user is not None:
            pairs.append({"user": pending_user, "assistant": turn.content})
            pending_user = None
    if not pairs:
        text = session.text
        if len(text) > _MAX_PAIR_CHARS:
            text = text[:_MAX_PAIR_CHARS]
        return [
            {
                "id": f"{question_id}:{session.session_id}",
                "title": f"session {session.session_id} ({session.date})",
                "text": session.text,
                "infer": True,
                "metadata": {"question_id": question_id, "kind": "session"},
                "additional_metadata": {
                    "question_id": question_id,
                    "kind": "session",
                    "session_id": session.session_id,
                    "date": session.date,
                    "summary": session.summary,
                    "part": 0,
                },
            }
        ]

    parts: list[list[dict[str, str]]] = [[]]
    cost = 0
    for pair in pairs:
        # A single pair must fit in one request: keep the user message whole,
        # truncate the assistant reply (its tail is least critical for retrieval).
        total = len(pair["user"]) + len(pair["assistant"])
        if total > _MAX_PAIR_CHARS:
            budget = _MAX_PAIR_CHARS - len(pair["user"])
            if budget > 0:
                pair["assistant"] = pair["assistant"][:budget]
            else:
                pair["user"] = pair["user"][:_MAX_PAIR_CHARS]
                pair["assistant"] = ""
        c = _pair_cost(pair)
        if parts[-1] and cost + c > _INGEST_BUDGET:
            parts.append([])
            cost = 0
        parts[-1].append(pair)
        cost += c

    items: list[dict] = []
    for i, part_pairs in enumerate(parts):
        sid = session.session_id if i == 0 else f"{session.session_id}#{i}"
        am = {
            "question_id": question_id,
            "kind": "session",
            "session_id": session.session_id,
            "date": session.date,
            "part": i,
            "part_count": len(parts),
        }
        if i == 0:
            am["summary"] = session.summary
        items.append(
            {
                "id": f"{question_id}:{sid}",
                "title": (
                    f"session {session.session_id} ({session.date})"
                    if i == 0
                    else f"session {session.session_id} ({session.date}) part {i + 1}/{len(parts)}"
                ),
                "user_assistant_pairs": part_pairs,
                "infer": True,
                "metadata": {"question_id": question_id, "kind": "session"},
                "additional_metadata": am,
            }
        )
    return items


def session_to_memory_item(question_id: str, session: Session) -> dict:
    """Backward-compatible alias: first item of session_to_memory_items."""
    return session_to_memory_items(question_id, session)[0]


def _fact_valid_as_of(am: dict, as_of_date: str | None) -> bool:
    """Fact.valid_as_of() over chunk additional_metadata (None = currently valid)."""
    valid_from, valid_to = am.get("valid_from"), am.get("valid_to")
    if as_of_date is None:
        return valid_to in (None, "")
    if valid_from is None or compare_dates(valid_from, as_of_date) > 0:
        return False
    return valid_to in (None, "") or compare_dates(as_of_date, valid_to) < 0


def _parse_content_turns(text: str) -> list[Turn]:
    """Parse context.inspect content into turns. Multi-pair memories come back
    as a JSON envelope `{"pairs": [...], "user_name": ...}`; single pairs render
    as "USER:/ASSISTANT:" lines (our canonical serialization)."""
    if not text:
        return []
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            obj = None
        pairs = obj.get("pairs") if isinstance(obj, dict) else None
        if isinstance(pairs, list):
            turns: list[Turn] = []
            for p in pairs:
                if not isinstance(p, dict):
                    continue
                if p.get("user") is not None:
                    turns.append(Turn(role="user", content=str(p["user"])))
                if p.get("assistant") is not None:
                    turns.append(Turn(role="assistant", content=str(p["assistant"])))
            if turns:
                return turns
    turns = []
    role: str | None = None
    buf: list[str] = []
    for line in s.splitlines():
        m = _CONTENT_TURN_RE.match(line)
        if m:
            if role is not None and buf:
                turns.append(Turn(role=role, content="\n".join(buf)))
            role, buf = m.group(1).lower(), [m.group(2)]
        elif role is not None:
            buf.append(line)
    if role is not None and buf:
        turns.append(Turn(role=role, content="\n".join(buf)))
    return turns


def _chunk_memory_id(path: object) -> str:
    """Memory (source) id behind a graph path's chunk ids: `x_chunk_0000` -> `x`."""
    ids = getattr(path, "source_chunk_ids", None) or []
    for cid in ids:
        m = _CHUNK_ID_RE.match(cid or "")
        if m:
            return m.group(1)
    return ids[0] if ids else (getattr(path, "group_id", None) or "")


def _path_date(path: object) -> str:
    """Graph edge date from the first triplet's temporal_details (else '')."""
    for t in getattr(path, "triplets", None) or []:
        rel = getattr(t, "relation", None)
        if isinstance(rel, dict):
            d = rel.get("temporal_details")
            if d:
                return str(d)
    return ""


def _path_text(path: object) -> str:
    """Graph path as text: combined_context, else formatted triplets."""
    combined = getattr(path, "combined_context", None)
    if combined and str(combined).strip():
        return str(combined).strip()
    lines: list[str] = []
    for t in getattr(path, "triplets", None) or []:
        rel = getattr(t, "relation", None) or {}
        src = getattr(t, "source", None) or {}
        tgt = getattr(t, "target", None) or {}
        src_name = src.get("name") if isinstance(src, dict) else getattr(src, "name", "")
        tgt_name = tgt.get("name") if isinstance(tgt, dict) else getattr(tgt, "name", "")
        pred = (
            rel.get("canonical_predicate")
            if isinstance(rel, dict)
            else getattr(rel, "canonical_predicate", "")
        )
        ctx = rel.get("context") if isinstance(rel, dict) else getattr(rel, "context", "")
        line = f"{src_name} {pred} {tgt_name}"
        if ctx:
            line += f" ({ctx})"
        lines.append(line)
    return " | ".join(lines)


def _norm_key(s: str) -> str:
    return " ".join(tokenize(s))


class HydraDBStore(MemoryStore):
    def __init__(
        self,
        api_key: str,
        *,
        database: str = "hydrous",
        base_url: str | None = None,
        poll_interval: float = 2.0,
        poll_timeout: float = 600.0,
    ):
        try:
            from hydra_db import AsyncHydraDB
        except ImportError as e:
            raise RuntimeError(_SDK_HELP) from e
        self._client = (
            AsyncHydraDB(token=api_key, timeout=300.0)
            if base_url is None
            else AsyncHydraDB(token=api_key, base_url=base_url, timeout=300.0)
        )
        self._database = database
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._ready = False
        self._ingest_window: list[tuple[float, int]] = []
        self._ingest_window_lock = asyncio.Lock()

    @staticmethod
    def _collection(question_id: str) -> str:
        return f"qid_{question_id}"

    # ------------------------------------------------------------------ infra

    async def _ensure_database_ready(self) -> None:
        if self._ready:
            return
        try:
            from hydra_db.errors import ConflictError

            await self._client.databases.create(database=self._database)
        except ConflictError:
            pass  # already exists
        deadline = time.monotonic() + self._poll_timeout
        while time.monotonic() < deadline:
            status = await self._client.databases.status(database=self._database)
            if status.data and status.data.infra and status.data.infra.ready_for_ingestion:
                self._ready = True
                return
            await asyncio.sleep(self._poll_interval)
        raise TimeoutError(f"HydraDB database {self._database!r} not ready within {self._poll_timeout}s")

    async def _wait_indexed(self, collection: str, ids: list[str]) -> None:
        deadline = time.monotonic() + self._poll_timeout
        while time.monotonic() < deadline:
            st = await self._client.context.status(
                database=self._database, collection=collection, ids=ids
            )
            statuses = st.data.statuses if st and st.data else []
            if not statuses:
                await asyncio.sleep(self._poll_interval)
                continue
            done = {s.id for s in statuses if s.indexing_status in ("completed", "failed")}
            if done and len(done) >= len(set(ids)):
                failed = [s for s in statuses if s.indexing_status == "failed"]
                if failed:
                    raise RuntimeError(
                        "HydraDB indexing failed: "
                        + json.dumps([(s.id, s.error_message) for s in failed])
                    )
                return
            await asyncio.sleep(self._poll_interval)
        raise TimeoutError(f"HydraDB collection {collection!r} not indexed within {self._poll_timeout}s")

    async def _acquire_ingest_budget(self, cost: int) -> None:
        """Token-window pacing: cap concurrent ingests at ~_PER_MIN_BUDGET
        "memory_tokens" per rolling 60s (server 429s past 5000/min)."""
        async with self._ingest_window_lock:
            while True:
                now = time.monotonic()
                self._ingest_window = [(t, c) for t, c in self._ingest_window if now - t < 60.0]
                if cost >= _PER_MIN_BUDGET:
                    # single oversized payload: no window slot will fit it; take it
                    # anyway (the 429 retry loop in _ingest_items is the backstop).
                    self._ingest_window.append((now, cost))
                    return
                if sum(c for _, c in self._ingest_window) + cost <= _PER_MIN_BUDGET:
                    self._ingest_window.append((now, cost))
                    return
                oldest = self._ingest_window[0]
                await asyncio.sleep(oldest[0] + 60.0 - now + 0.05)

    async def _ingest_items(self, question_id: str, items: list[dict]) -> None:
        if not items:
            return
        from hydra_db.core.api_error import ApiError
        from hydra_db.errors.content_too_large_error import ContentTooLargeError

        await self._ensure_database_ready()
        collection = self._collection(question_id)
        # The server enforces a ~1000 "memory_tokens" budget per ingest request,
        # so split the payload into cost-bounded chunks (one request each).
        chunks: list[list[dict]] = [[]]
        cost = 0
        for item in items:
            c = _item_cost(item)
            if chunks[-1] and cost + c > _INGEST_BUDGET:
                chunks.append([])
                cost = 0
            chunks[-1].append(item)
            cost += c
        ids: list[str] = []
        for batch in chunks:
            batch_cost = sum(_item_cost(i) for i in batch)
            await self._acquire_ingest_budget(batch_cost)
            resp = None
            for attempt in range(_RATE_RETRIES):
                try:
                    resp = await self._client.context.ingest(
                        type="memory",
                        database=self._database,
                        collection=collection,
                        memories=json.dumps(batch),
                        upsert="true",
                    )
                    break
                except ApiError as e:
                    if e.status_code == 429 and attempt < _RATE_RETRIES - 1:
                        log.warning("ingest 429 (rate limit) — retrying in %ss", 5 * (attempt + 1))
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    raise
                except ContentTooLargeError:
                    if attempt == _RATE_RETRIES - 1 or not batch:
                        raise
                    # server cost is tokenizer-driven (CJK/emoji inflate it), so a
                    # chunk can still exceed the per-request budget: shrink pairs
                    # and retry rather than failing the whole session.
                    log.warning("ingest payload too large — shrinking batch")
                    for it in batch:
                        for p in it.get("user_assistant_pairs") or []:
                            p["assistant"] = (p.get("assistant") or "")[:_MAX_PAIR_CHARS // 2]
                            p["user"] = (p.get("user") or "")[:_MAX_PAIR_CHARS // 2]
                        if "text" in it:
                            it["text"] = (it["text"] or "")[:_MAX_PAIR_CHARS // 2]
            got = [
                r.id for r in (resp.data.results or []) if r.id
            ] if resp and resp.data else []
            ids.extend(got or [i["id"] for i in batch])
        await self._wait_indexed(collection, ids)

    async def _list_memories(
        self, question_id: str, *, kind: str | None = None, additional: dict | None = None
    ) -> list[dict]:
        """context.list (raw body: SDK types are stale) with additional_metadata
        filters + pagination. Missing database/collection -> []."""
        from hydra_db.errors import NotFoundError

        await self._ensure_database_ready()
        filters: dict | None = None
        am = dict(additional or {})
        if kind:
            am["kind"] = kind
        if am:
            filters = {"additional_metadata": am}
        collection = self._collection(question_id)
        items: list[dict] = []
        page = 1
        while True:
            try:
                raw = await self._client.context.with_raw_response.list(
                    database=self._database, collection=collection, type="memory",
                    filters=filters, page=page, page_size=_MAX_INGEST_BATCH,
                )
            except NotFoundError:
                # unknown collection on a fresh question id -> empty
                log.debug("context.list: collection %s not found", collection)
                return []
            body = raw._response.json() if hasattr(raw, "_response") else {}
            data = body.get("data") or {}
            items.extend(data.get("user_memories") or [])
            pagination = data.get("pagination") or {}
            if not pagination.get("has_next"):
                break
            page += 1
        return items

    # ------------------------------------------------------------------ interface

    async def add_session(
        self, question_id: str, session: Session, facts: list[Fact] | None = None
    ) -> None:
        items = session_to_memory_items(question_id, session)
        if facts:  # fact-augmented indexing: ingest facts alongside their session
            items.extend(fact_to_memory_item(f) for f in facts)
        await self._ingest_items(question_id, items)

    async def add_facts(self, question_id: str, facts: list[Fact]) -> None:
        if facts and await self.has_question(question_id):
            # facts were already ingested with their session (fact-augmented); skip re-ingest
            return
        await self._ingest_items(question_id, [fact_to_memory_item(f) for f in facts])

    async def update_fact(self, fact: Fact) -> None:
        # metadata PATCH on the stable id — no re-ingest; verified to sync to the
        # vector store (query-time metadata_filters on patched keys match).
        await self._ensure_database_ready()
        await self._client.context.update_source_metadata(
            fact.id,
            database=self._database,
            collection=self._collection(fact.question_id),
            additional_metadata={"valid_to": fact.valid_to, "superseded_by": fact.superseded_by},
        )

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
        await self._ensure_database_ready()
        over_fetch = min(max(top_k * 8, 120), 200)
        kwargs: dict = dict(
            database=self._database,
            collection=self._collection(question_id),
            type="memory",
            query=query,
            query_by="hybrid",
            mode="thinking",
            max_results=over_fetch,
            recency_bias=0.4,
            graph_context=True,
        )
        if include_facts and not include_sessions:
            kwargs["metadata_filters"] = {"kind": "fact"}
        elif include_sessions and not include_facts:
            kwargs["metadata_filters"] = {"kind": "session"}
        result = await self._client.query(**kwargs)
        qdata = result.data if result else None
        if qdata is None:
            return []

        chunks = qdata.chunks or []
        max_chunk_score = max((c.relevancy_score or 0.0) for c in chunks) if chunks else 0.0

        out: list[Evidence] = []
        for c in chunks:
            am = c.additional_metadata or {}
            kind = am.get("kind") or (c.metadata or {}).get("kind")
            score = c.relevancy_score or 0.0
            if kind == "fact":
                if not include_superseded and not _fact_valid_as_of(am, as_of_date):
                    continue
                out.append(
                    Evidence(
                        source="fact",
                        text=(c.chunk_content or "").strip(),
                        date=str(am.get("fact_date") or ""),
                        session_id=str(am.get("session_id") or ""),
                        score=score,
                        fact_id=c.id or "",
                    )
                )
            else:  # sessions have no validity interval
                out.append(
                    Evidence(
                        source="session",
                        text=(c.chunk_content or "").strip(),
                        date=str(am.get("date") or ""),
                        session_id=str(am.get("session_id") or c.id or ""),
                        score=score,
                    )
                )

        # The cloud chunks long sessions into many fragments: one chunk per
        # part (a 16KB session yields 5-6). Dedupe by session, keeping the best
        # fragment's score, so a gold session is ranked as a unit.
        best: dict[tuple[str, str], Evidence] = {}
        for ev in out:
            key = (ev.source, ev.session_id)
            if key not in best or ev.score > best[key].score:
                best[key] = ev
        out = list(best.values())

        # Multi-hop graph lanes (mem0 has nothing equivalent): query paths and
        # chunk relations are query-relevant by construction, so they score
        # relative to the best chunk (else their raw ~0.07 scores get trimmed
        # away before the reader ever sees them).
        gc = qdata.graph_context
        if gc:
            paths = [p for p in (gc.query_paths or []) if _path_text(p)]
            for path in paths[:3]:
                out.append(
                    Evidence(
                        source="graph",
                        text=_path_text(path),
                        date=_path_date(path),
                        session_id=_chunk_memory_id(path),
                        score=max_chunk_score * 0.7 + (path.relevancy_score or 0.0) * 1e-3,
                    )
                )

        # Lane-separated RRF over the cloud scores: facts compete against
        # facts, sessions/graph paths against their own kind, so short fact
        # snippets cannot drown out long gold sessions.
        lanes: dict[str, list[int]] = {}
        for i, ev in enumerate(out):
            lanes.setdefault(ev.source, []).append(i)
        for source, idxs in lanes.items():
            weight = _FACT_LANE_WEIGHT if source == "fact" else 1.0
            ranks = _ranks([out[i].score for i in idxs])
            for pos, i in enumerate(idxs):
                out[i].score = weight * (1.0 / (_RRF_K + ranks[pos]))

        out.sort(key=lambda e: e.score, reverse=True)
        top = out[:top_k]

        # Fragments only cover ~2KB slices of a session; the reader needs the
        # whole session (parts joined) to find statements anywhere in it.
        for ev in top:
            if ev.source == "session" and ev.session_id:
                full = await self._fetch_session_text(question_id, ev.session_id)
                if full:
                    ev.text = full
        return top

    async def _fetch_session_text(self, question_id: str, session_id: str) -> str:
        """Full session text from its cloud parts (memory items), cached."""
        cache_key = (self._collection(question_id), session_id)
        if cache_key in _SESSION_TEXT_CACHE:
            return _SESSION_TEXT_CACHE[cache_key]
        try:
            parts = await self._list_memories(
                question_id, kind="session", additional={"session_id": session_id}
            )
            parts.sort(key=lambda m: int((m.get("additional_metadata") or {}).get("part") or 0))
            chunks: list[str] = []
            for m in parts:
                insp = await self._client.context.inspect(
                    id=m.get("memory_id") or session_id,
                    database=self._database,
                    collection=self._collection(question_id),
                )
                content = insp.data.content if insp and insp.data else ""
                chunks.append(_content_to_text(content))
            text = "\n\n".join(c for c in chunks if c).strip()
        except Exception:
            log.warning("session text hydration failed for %s", session_id, exc_info=True)
            text = ""
        if text:
            if len(_SESSION_TEXT_CACHE) >= 64:
                _SESSION_TEXT_CACHE.pop(next(iter(_SESSION_TEXT_CACHE)))
            _SESSION_TEXT_CACHE[cache_key] = text
        return text

    async def get_facts(self, question_id: str) -> list[Fact]:
        facts: list[Fact] = []
        for m in await self._list_memories(question_id, kind="fact"):
            am = m.get("additional_metadata") or {}
            required = ("subject", "predicate", "value", "fact_date", "session_id", "valid_from")
            if not all(k in am for k in required):
                log.warning("skipping malformed fact memory %s", m.get("memory_id"))
                continue
            facts.append(
                Fact(
                    id=m.get("memory_id") or "",
                    question_id=am.get("question_id") or question_id,
                    subject=am["subject"],
                    predicate=am["predicate"],
                    value=am["value"],
                    fact_date=am["fact_date"],
                    session_id=am["session_id"],
                    valid_from=am["valid_from"],
                    valid_to=am.get("valid_to"),
                    recorded_at=m.get("timestamp") or "",
                    superseded_by=am.get("superseded_by"),
                )
            )
        return facts

    async def get_sessions(self, question_id: str) -> list[Session]:
        collection = self._collection(question_id)
        by_sid: dict[str, list[dict]] = {}
        for m in await self._list_memories(question_id, kind="session"):
            am = m.get("additional_metadata") or {}
            sid = am.get("session_id") or ""
            if not sid:
                continue
            by_sid.setdefault(sid, []).append(m)
        sessions: list[Session] = []
        for sid, parts in by_sid.items():
            parts.sort(key=lambda m: int((m.get("additional_metadata") or {}).get("part") or 0))
            turns: list[Turn] = []
            for m in parts:
                try:
                    insp = await self._client.context.inspect(
                        id=m.get("memory_id") or sid, database=self._database, collection=collection
                    )
                    turns.extend(_parse_content_turns(insp.data.content if insp and insp.data else ""))
                except Exception:
                    log.warning(
                        "context.inspect failed for session part %s", m.get("memory_id"), exc_info=True
                    )
            am = (parts[0].get("additional_metadata") or {}) if parts else {}
            sessions.append(
                Session(
                    session_id=sid,
                    date=str(am.get("date") or ""),
                    summary=am.get("summary"),
                    turns=turns,
                )
            )
        return sorted(sessions, key=functools.cmp_to_key(lambda a, b: compare_dates(a.date, b.date)))

    async def get_timeline(self, question_id: str, subject: str, predicate: str) -> list[Fact]:
        sk, pk = _norm_key(subject), _norm_key(predicate)
        facts = [
            f for f in await self.get_facts(question_id)
            if _norm_key(f.subject) == sk and _norm_key(f.predicate) == pk
        ]
        return sorted(facts, key=functools.cmp_to_key(lambda a, b: compare_dates(a.valid_from, b.valid_from)))

    async def find_candidates(
        self, question_id: str, subject: str, predicate: str, text: str, threshold: float = 0.75
    ) -> list[Fact]:
        """Currently-valid facts that might conflict: exact (subject, predicate)
        matches from context.list + a semantic lane over the vector store (same
        subject, chunk relevancy floor — cloud scores are not cosine similarities,
        so the threshold maps to min(threshold, 0.5))."""
        await self._ensure_database_ready()
        valid = [f for f in await self.get_facts(question_id) if f.valid_to is None]
        sk, pk = _norm_key(subject), _norm_key(predicate)
        exact = [f for f in valid if _norm_key(f.subject) == sk and _norm_key(f.predicate) == pk]
        exact_ids = {f.id for f in exact}

        similar: list[Fact] = []
        by_id = {f.id: f for f in valid}
        try:
            result = await self._client.query(
                database=self._database,
                collection=self._collection(question_id),
                type="memory",
                query=text,
                query_by="hybrid",
                mode="thinking",
                max_results=40,
                recency_bias=0.0,
                graph_context=False,
                metadata_filters={"additional_metadata": {"kind": "fact", "subject": subject}},
            )
            floor = min(threshold, 0.5)
            for c in (result.data.chunks or []) if result and result.data else []:
                cand = by_id.get(c.id)
                if cand is None or cand.id in exact_ids:
                    continue
                if (c.relevancy_score or 0.0) >= floor:
                    similar.append(cand)
        except Exception:
            log.warning("HydraDBStore.find_candidates semantic lane failed", exc_info=True)
        return exact + similar

    async def has_question(self, question_id: str) -> bool:
        from hydra_db.errors import NotFoundError

        await self._ensure_database_ready()
        try:
            raw = await self._client.context.with_raw_response.list(
                database=self._database,
                collection=self._collection(question_id),
                type="memory",
                page=1,
                page_size=1,
            )
            body = raw._response.json() if hasattr(raw, "_response") else {}
            return ((body.get("data") or {}).get("total") or 0) > 0
        except NotFoundError:
            return False

    async def delete_question(self, question_id: str) -> None:
        items = await self._list_memories(question_id)
        ids = [m["memory_id"] for m in items if m.get("memory_id")]
        for start in range(0, len(ids), _MAX_INGEST_BATCH):
            await self._client.context.delete(
                database=self._database,
                collection=self._collection(question_id),
                ids=ids[start : start + _MAX_INGEST_BATCH],
                type="memory",
            )

    def close(self) -> None:
        pass
