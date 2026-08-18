"""Phase 3 live cloud re-probe (small, quota-light).

Verifies the wire shapes Phase 2 added but never ran against the real API:
  1. fact additional_metadata carries quality_score (and round-trips via get_facts)
  2. session additional_metadata carries recorded_at + meta (round-trips via get_sessions)
  3. delete_memory via context.delete (a source id, not a chunk id)
  4. search returns graph info for graph_context=True (the multi-hop demo path)

Usage:  HYDRA_DB_API_KEY=... .venv/bin/python scripts/probe_phase3.py
Creates database 'hydrous_probe3', deletes the probe collection at the end.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hydrous.schema import Fact, Session, Turn  # noqa: E402
from hydrous.store.hydradb import HydraDBStore  # noqa: E402

DB = os.environ.get("HYDRA_DB_PROBE3_DB", "hydrous_probe3")
QID = "qprobe3"
KEY = os.environ.get("HYDRA_DB_API_KEY") or os.environ.get("HYDRA_DB_PROBE_KEY")
if not KEY:
    raise SystemExit("HYDRA_DB_API_KEY is required")
URL = os.environ.get("HYDRA_DB_BASE_URL")


def fact(fid, subject, predicate, value, fact_date, quality=None) -> Fact:
    return Fact(
        id=fid, question_id=QID, subject=subject, predicate=predicate, value=value,
        fact_date=fact_date, session_id=fid, valid_from=fact_date, quality_score=quality,
    )


async def main() -> None:
    store = HydraDBStore(KEY, database=DB, base_url=URL)
    await store.delete_question(QID)  # clean slate (ignores missing collection)

    scored = fact("pf_scored", "user", "favorite_color", "blue", "2023-05-01", quality=0.87)
    unscored = fact("pf_unscored", "user", "lives_in", "Berlin", "2023-06-01")
    session = Session(
        session_id="pf_session1",
        date="2023-06-10",
        turns=[Turn(role="user", content="I am planning a trip to Japan in April.")],
        recorded_at="2025-01-15T10:30:00+00:00",
        metadata={"category": "plans"},
    )
    await store.add_facts(QID, [scored, unscored])
    await store.add_session(QID, session)

    # 1. quality_score round-trip
    facts = {f.id: f for f in await store.get_facts(QID)}
    assert facts["pf_scored"].quality_score == 0.87, facts["pf_scored"].quality_score
    assert facts["pf_unscored"].quality_score is None
    print("OK  quality_score round-trips (0.87 / None)")

    # 2. recorded_at + meta round-trip
    sessions = {s.session_id: s for s in await store.get_sessions(QID)}
    s = sessions["pf_session1"]
    assert s.recorded_at.startswith("2025-01-15T10:30:00"), s.recorded_at
    assert s.metadata == {"category": "plans"}, s.metadata
    assert s.turns and "Japan" in s.turns[0].content
    print("OK  session recorded_at + meta round-trip")

    # 3. search with graph_context (the multi-hop demo path) — raw body check
    result = await store.search(QID, "favorite color", top_k=5)
    texts = {e.text for e in result}
    assert any("blue" in t for t in texts), texts
    raw = await store._client.query(
        database=DB, collection=store._collection(QID), type="memory",
        query="favorite color", query_by="hybrid", mode="thinking",
        max_results=5, graph_context=True,
    )
    qdata = raw.data if raw else None
    gc = qdata.graph_context if qdata else None
    print("OK  hybrid search hits; graph_context object present:", gc is not None)
    paths = (gc.query_paths or []) if gc else []
    print(f"    graph query_paths returned: {len(paths)}")

    # 4. delete_memory via context.delete (source id)
    await store.delete_memory(QID, "pf_scored")
    remaining = {f.id for f in await store.get_facts(QID)}
    assert "pf_scored" not in remaining, remaining
    print("OK  delete_memory removed source id pf_scored")

    await store.delete_question(QID)  # leave no trace (quota)
    print(f"done — probe collection cleaned, database {DB!r} left in place")


asyncio.run(main())
