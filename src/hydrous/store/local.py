"""Local MemoryStore: SQLite + dense embeddings + BM25 hybrid with temporal filtering.

Per-question corpora are small (~40 sessions, a few hundred facts), so search loads the
question's rows, computes cosine similarity in numpy, and builds BM25 on the fly.
Dense and BM25 rankings are merged with reciprocal rank fusion (RRF, k=60).
"""

from __future__ import annotations

import functools
import json
import sqlite3
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ..embeddings import Embedder
from ..schema import Evidence, Fact, Session, Turn
from ..text_utils import compare_dates, tokenize
from .base import MemoryStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    question_id TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    date        TEXT NOT NULL,
    summary     TEXT,
    text        TEXT NOT NULL,
    recorded_at TEXT,
    metadata    TEXT,
    embedding   BLOB,
    PRIMARY KEY (question_id, session_id)
);
CREATE TABLE IF NOT EXISTS turns (
    question_id TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    PRIMARY KEY (question_id, session_id, idx)
);
CREATE TABLE IF NOT EXISTS facts (
    id            TEXT PRIMARY KEY,
    question_id   TEXT NOT NULL,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    value         TEXT NOT NULL,
    fact_date     TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    valid_from    TEXT NOT NULL,
    valid_to      TEXT,
    recorded_at   TEXT NOT NULL,
    superseded_by TEXT,
    quality_score REAL,
    embedding     BLOB
);
CREATE INDEX IF NOT EXISTS idx_facts_qid ON facts(question_id);
CREATE INDEX IF NOT EXISTS idx_facts_sp  ON facts(question_id, subject, predicate);
"""

# old sqlite files predate these columns (CREATE TABLE IF NOT EXISTS won't add them)
_MIGRATIONS = (
    "ALTER TABLE sessions ADD COLUMN recorded_at TEXT",
    "ALTER TABLE sessions ADD COLUMN metadata TEXT",
    "ALTER TABLE facts ADD COLUMN quality_score REAL",
)

_RRF_K = 60
_FACT_LANE_WEIGHT = 0.4


def _norm_key(s: str) -> str:
    return " ".join(tokenize(s))


def _to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class LocalMemoryStore(MemoryStore):
    def __init__(self, path: str | Path, embedder: Embedder):
        self._path = Path(path)
        if str(self._path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._embedder = embedder
        self._conn.executescript(_SCHEMA)
        for stmt in _MIGRATIONS:  # tolerate already-migrated files ("duplicate column")
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    # ------------------------------------------------------------------ write

    async def add_session(
        self, question_id: str, session: Session, facts: list[Fact] | None = None
    ) -> None:
        # Fact-augmented indexing: embed session text together with summary + fact texts.
        parts = [session.text]
        if session.summary:
            parts.append(session.summary)
        if facts:
            parts.extend(f.text for f in facts)
        emb = self._embedder.embed(["\n".join(parts)])[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions"
            " (question_id, session_id, date, summary, text, recorded_at, metadata, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                question_id, session.session_id, session.date, session.summary, session.text,
                session.recorded_at, json.dumps(session.metadata, ensure_ascii=False), _to_blob(emb),
            ),
        )
        self._conn.execute(
            "DELETE FROM turns WHERE question_id = ? AND session_id = ?", (question_id, session.session_id)
        )
        self._conn.executemany(
            "INSERT INTO turns (question_id, session_id, idx, role, content) VALUES (?, ?, ?, ?, ?)",
            [(question_id, session.session_id, i, t.role, t.content) for i, t in enumerate(session.turns)],
        )
        self._conn.commit()

    async def add_facts(self, question_id: str, facts: list[Fact]) -> None:
        if not facts:
            return
        embs = self._embedder.embed([f.text for f in facts])
        self._conn.executemany(
            "INSERT OR REPLACE INTO facts"
            " (id, question_id, subject, predicate, value, fact_date, session_id,"
            "  valid_from, valid_to, recorded_at, superseded_by, quality_score, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    f.id, question_id, f.subject, f.predicate, f.value, f.fact_date, f.session_id,
                    f.valid_from, f.valid_to, f.recorded_at, f.superseded_by, f.quality_score,
                    _to_blob(embs[i]),
                )
                for i, f in enumerate(facts)
            ],
        )
        self._conn.commit()

    async def update_fact(self, fact: Fact) -> None:
        self._conn.execute(
            "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?",
            (fact.valid_to, fact.superseded_by, fact.id),
        )
        self._conn.commit()

    async def delete_memory(self, question_id: str, memory_id: str) -> None:
        """Delete one memory: a fact by id, else a session by session_id. No-op if unknown."""
        cur = self._conn.execute(
            "DELETE FROM facts WHERE question_id = ? AND id = ?", (question_id, memory_id)
        )
        if cur.rowcount:
            self._conn.commit()
            return
        cur = self._conn.execute(
            "DELETE FROM sessions WHERE question_id = ? AND session_id = ?", (question_id, memory_id)
        )
        if cur.rowcount:
            self._conn.execute(
                "DELETE FROM turns WHERE question_id = ? AND session_id = ?", (question_id, memory_id)
            )
            self._conn.commit()

    # ------------------------------------------------------------------ read

    def _load_facts(self, question_id: str) -> list[tuple[Fact, np.ndarray]]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE question_id = ?", (question_id,)
        ).fetchall()
        out = []
        for r in rows:
            fact = Fact(
                id=r["id"], question_id=r["question_id"], subject=r["subject"],
                predicate=r["predicate"], value=r["value"], fact_date=r["fact_date"],
                session_id=r["session_id"], valid_from=r["valid_from"], valid_to=r["valid_to"],
                recorded_at=r["recorded_at"], superseded_by=r["superseded_by"],
                quality_score=r["quality_score"],
            )
            out.append((fact, _from_blob(r["embedding"])))
        return out

    def _load_sessions(
        self, question_id: str
    ) -> list[tuple[str, str, str, str | None, str, dict, np.ndarray]]:
        rows = self._conn.execute(
            "SELECT session_id, date, text, summary, recorded_at, metadata, embedding"
            " FROM sessions WHERE question_id = ?",
            (question_id,),
        ).fetchall()
        return [
            (
                r["session_id"], r["date"], r["text"], r["summary"],
                r["recorded_at"] or r["date"],  # legacy rows: fall back to session date
                json.loads(r["metadata"]) if r["metadata"] else {},
                _from_blob(r["embedding"]),
            )
            for r in rows
        ]

    async def get_facts(self, question_id: str) -> list[Fact]:
        return [f for f, _ in self._load_facts(question_id)]

    async def get_sessions(self, question_id: str) -> list[Session]:
        sessions = []
        for sid, date, _text, summary, recorded_at, meta, _emb in self._load_sessions(question_id):
            rows = self._conn.execute(
                "SELECT role, content FROM turns WHERE question_id = ? AND session_id = ? ORDER BY idx",
                (question_id, sid),
            ).fetchall()
            sessions.append(
                Session(
                    session_id=sid, date=date, summary=summary,
                    recorded_at=recorded_at, metadata=meta,
                    turns=[Turn(role=r["role"], content=r["content"]) for r in rows],
                )
            )
        return sorted(sessions, key=functools.cmp_to_key(lambda a, b: compare_dates(a.date, b.date)))

    async def get_timeline(self, question_id: str, subject: str, predicate: str) -> list[Fact]:
        sk, pk = _norm_key(subject), _norm_key(predicate)
        facts = [
            f for f, _ in self._load_facts(question_id)
            if _norm_key(f.subject) == sk and _norm_key(f.predicate) == pk
        ]
        return sorted(facts, key=functools.cmp_to_key(lambda a, b: compare_dates(a.valid_from, b.valid_from)))

    async def find_candidates(
        self, question_id: str, subject: str, predicate: str, text: str, threshold: float = 0.75
    ) -> list[Fact]:
        facts = [(f, emb) for f, emb in self._load_facts(question_id) if f.valid_to is None]
        if not facts:
            return []
        sk, pk = _norm_key(subject), _norm_key(predicate)
        exact = [f for f, _ in facts if _norm_key(f.subject) == sk and _norm_key(f.predicate) == pk]
        exact_ids = {f.id for f in exact}
        q = self._embedder.embed([text])[0]
        similar = [
            f for (f, emb), sim in zip(facts, [float(emb @ q) for _, emb in facts], strict=True)
            if f.id not in exact_ids and _norm_key(f.subject) == sk and sim >= threshold
        ]
        return exact + similar

    async def has_question(self, question_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE question_id = ? LIMIT 1", (question_id,)
        ).fetchone()
        return row is not None

    async def delete_question(self, question_id: str) -> None:
        for table in ("sessions", "turns", "facts"):
            self._conn.execute(f"DELETE FROM {table} WHERE question_id = ?", (question_id,))
        self._conn.commit()

    # ------------------------------------------------------------------ search

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
        candidates: list[Evidence] = []
        embs: list[np.ndarray] = []
        lanes: list[list[int]] = [[], []]
        if include_facts:
            for fact, emb in self._load_facts(question_id):
                if not include_superseded and not fact.valid_as_of(as_of_date):
                    continue
                lanes[0].append(len(candidates))
                candidates.append(
                    Evidence(
                        source="fact", text=fact.text, date=fact.fact_date,
                        session_id=fact.session_id, fact_id=fact.id,
                    )
                )
                embs.append(emb)
        if include_sessions:
            for sid, date, text, _summary, _recorded_at, _meta, emb in self._load_sessions(question_id):
                lanes[1].append(len(candidates))
                candidates.append(Evidence(source="session", text=text, date=date, session_id=sid))
                embs.append(emb)
        if not candidates:
            return []

        q = self._embedder.embed([query])[0]
        dense_scores = np.stack(embs) @ q

        corpus = [tokenize(ev.text) for ev in candidates]
        query_tokens = tokenize(query)
        if query_tokens and any(corpus):
            bm25_scores = BM25Okapi(corpus).get_scores(query_tokens)
        else:
            bm25_scores = np.zeros(len(candidates))

        # Lane-separated RRF: facts compete against facts, sessions against
        # sessions, so hundreds of short fact snippets cannot drown out long
        # gold sessions (which lose token-overlap counts despite matching).
        for idxs, weight in ((lanes[0], _FACT_LANE_WEIGHT), (lanes[1], 1.0)):
            if not idxs:
                continue
            dense_ranks = _ranks(dense_scores[idxs])
            bm25_ranks = _ranks(np.asarray(bm25_scores[idxs], dtype=np.float64))
            for pos, i in enumerate(idxs):
                candidates[i].score = weight * (
                    1.0 / (_RRF_K + dense_ranks[pos]) + 1.0 / (_RRF_K + bm25_ranks[pos])
                )

        candidates.sort(key=lambda e: e.score, reverse=True)
        return candidates[:top_k]

    def close(self) -> None:
        self._conn.close()


def _ranks(scores: np.ndarray) -> np.ndarray:
    """rank 0 = highest score; ties broken by index (deterministic)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = np.empty(len(scores), dtype=np.int64)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks
