"""Live verification of the rewritten HydraDBStore against the real API.

Covers the whole MemoryStore surface the eval path touches:
add_session(+facts) -> has_question -> get_facts/get_sessions -> search (with
as-of-date, superseded, fact/session-only variants) -> find_candidates ->
update_fact (PATCH) -> get_timeline -> search again -> delete_question.

Usage:  HYDRA_DB_API_KEY=... .venv/bin/python scripts/verify_hydradb_store.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hydrous.schema import Fact, Session, Turn  # noqa: E402
from hydrous.store.hydradb import HydraDBStore  # noqa: E402

QID = "qverify1"
KEY = os.environ.get("HYDRA_DB_API_KEY") or os.environ.get("HYDRA_DB_PROBE_KEY")
BASE = os.environ.get("HYDRA_DB_BASE_URL") or os.environ.get("HYDRA_DB_VERIFY_URL")


def fact(qid, sid, subject, predicate, value, fact_date) -> Fact:
    return Fact(
        id=f"{qid}_{sid}_{subject}_{predicate}".replace(" ", "_").lower()[:40],
        question_id=qid, subject=subject, predicate=predicate, value=value,
        fact_date=fact_date, session_id=sid, valid_from=fact_date,
    )


async def main() -> None:
    store = HydraDBStore(KEY, database="hydrous_verify", base_url=BASE)
    await store.delete_question(QID)  # clean slate (ignores missing collection)

    s1 = Session(
        session_id="s1", date="2023/05/01 (Mon) 10:00",
        turns=[
            Turn(role="user", content="hi, where do I live?"),
            Turn(role="assistant", content="You live in Berlin."),
        ],
        summary="user asked where they live; answered Berlin.",
    )
    f1 = fact(QID, "s1", "user", "lives_in", "Berlin", "2023-05-01")

    s2 = Session(
        session_id="s2", date="2023/07/01 (Sat) 09:30",
        turns=[
            Turn(role="user", content="I moved. Now I live in Munich."),
            Turn(role="assistant", content="Got it — Munich now."),
        ],
        summary="user moved to Munich.",
    )
    f2 = fact(QID, "s2", "user", "lives_in", "Munich", "2023-07-01")

    print("== add_session (facts ingested alongside) ==")
    await store.add_session(QID, s1, facts=[f1])
    await store.add_session(QID, s2, facts=[f2])
    await store.add_facts(QID, [f1, f2])  # must be a no-op (already ingested)

    print("has_question:", await store.has_question(QID))

    print("\n== get_facts ==")
    facts = await store.get_facts(QID)
    for f in sorted(facts, key=lambda x: x.fact_date):
        print(f"  {f.id} | {f.subject} {f.predicate}: {f.value} | valid {f.valid_from}->{f.valid_to} | rec {f.recorded_at[:10]}")

    print("\n== get_sessions ==")
    for s in await store.get_sessions(QID):
        print(f"  {s.session_id} | {s.date} | {len(s.turns)} turns | summary={s.summary!r}")
        for t in s.turns:
            print(f"    {t.role}: {t.content[:60]!r}")

    print("\n== search 'where does the user live' (as_of 2023-06-01) ==")
    evs = await store.search(QID, "where does the user live", as_of_date="2023-06-01", top_k=5)
    for e in evs:
        print(f"  [{e.source}] score={e.score:.3f} date={e.date} id={e.fact_id or e.session_id}")
        print(f"      {e.text[:90]!r}")

    print("\n== search 'where does the user live' (as_of None = current only) ==")
    for e in await store.search(QID, "where does the user live", top_k=5):
        print(f"  [{e.source}] score={e.score:.3f} date={e.date} id={e.fact_id or e.session_id}")

    print("\n== search 'did we ever talk about Berlin' (include_superseded) ==")
    for e in await store.search(QID, "did we ever talk about Berlin", include_superseded=True, top_k=5):
        print(f"  [{e.source}] score={e.score:.3f} date={e.date} id={e.fact_id or e.session_id}")

    print("\n== search sessions-only ==")
    for e in await store.search(QID, "where does the user live", include_facts=False, top_k=5):
        print(f"  [{e.source}] score={e.score:.3f} date={e.date} id={e.session_id}")

    print("\n== find_candidates(user, lives_in) ==")
    for c in await store.find_candidates(QID, "user", "lives_in", "user lives_in: Berlin"):
        print(f"  {c.id} | {c.value} | valid_to={c.valid_to}")

    print("\n== update_fact: supersede f1 ==")
    f1.valid_to = "2023-07-01"
    f1.superseded_by = f2.id
    await store.update_fact(f1)
    got = {f.id: f for f in await store.get_facts(QID)}
    print(f"  f1 after: valid_to={got[f1.id].valid_to} superseded_by={got[f1.id].superseded_by}")

    print("\n== search again (f1 must be excluded as of 2023-06-01 AND current) ==")
    for e in await store.search(QID, "where does the user live", as_of_date="2023-06-01", top_k=5):
        print(f"  [{e.source}] score={e.score:.3f} date={e.date} id={e.fact_id or e.session_id}")

    print("\n== get_timeline(user, lives_in) — both versions, oldest first ==")
    for f in await store.get_timeline(QID, "user", "lives_in"):
        print(f"  {f.id} | {f.value} | valid {f.valid_from}->{f.valid_to}")

    print("\n== delete_question ==")
    await store.delete_question(QID)
    print("  has_question after delete:", await store.has_question(QID))
    print("  get_facts after delete:", await store.get_facts(QID))
    print("  get_sessions after delete:", await store.get_sessions(QID))
    print("\n== ALL CHECKS DONE ==")


if __name__ == "__main__":
    asyncio.run(main())
