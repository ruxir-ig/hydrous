"""10-minute live probe against a HydraDB gateway.

Pins down the two things SDK types don't reveal:
  1. the `metadata_filters` / `filters` grammar (flat vs nested, scalars vs arrays, operator)
  2. what `context.list` returns for memory content (metadata only? content? inferred?)

Flow: ensure database -> ingest (2 facts + 1 session) -> list -> query (no filter,
then filter experiments) -> metadata PATCH -> list + query again -> inspect.

Usage:  HYDRA_DB_API_KEY=... HYDRA_DB_BASE_URL=http://127.0.0.1:8317/v1 \
        .venv/bin/python scripts/probe_hydradb.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# NOTE: the SDK's MemoryItem.metadata is typed `str`, but the v2 server wants a map
# (400: cannot unmarshal string into MemoryItem.metadata of type map[string]interface{}).
# We build the wire payload directly, bypassing the stale model.

DB = os.environ.get("HYDRA_DB_PROBE_DB", "hydrous_probe")
COLL = "qprobe1"
BASE_URL = os.environ.get("HYDRA_DB_BASE_URL")  # None -> SDK default (https://api.hydradb.com)
API_KEY = os.environ.get("HYDRA_DB_API_KEY") or os.environ.get("HYDRA_DB_PROBE_KEY")
if not API_KEY:
    raise SystemExit("HYDRA_DB_API_KEY is required")

_items: list[dict] = [
    {
        "id": "fact_alice_berlin",
        "title": "user lives_in Berlin",
        "text": "user lives_in: Berlin",
        "infer": True,
        "metadata": {"question_id": "qprobe1", "kind": "fact"},
        "additional_metadata": {
            "question_id": "qprobe1", "kind": "fact", "session_id": "s1",
            "subject": "user", "predicate": "lives_in", "value": "Berlin",
            "fact_date": "2023-05-01", "valid_from": "2023-05-01",
        },
    },
    {
        "id": "fact_alice_munich",
        "title": "user lives_in Munich",
        "text": "user lives_in: Munich",
        "infer": True,
        "metadata": {"question_id": "qprobe1", "kind": "fact"},
        "additional_metadata": {
            "question_id": "qprobe1", "kind": "fact", "session_id": "s2",
            "subject": "user", "predicate": "lives_in", "value": "Munich",
            "fact_date": "2023-07-01", "valid_from": "2023-07-01",
        },
    },
    {
        "id": "qprobe1:s1",
        "title": "session s1 (2023-05-01)",
        "text": "USER: hi\nASSISTANT: hello",
        "infer": True,
        "metadata": {"question_id": "qprobe1", "kind": "session"},
        "additional_metadata": {
            "question_id": "qprobe1", "kind": "session", "session_id": "s1",
            "date": "2023/05/01 (Mon) 10:00",
        },
    },
]


def _fmt(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


async def main() -> None:
    from hydra_db import AsyncHydraDB

    client_kwargs = {"token": API_KEY}
    if BASE_URL:
        client_kwargs["base_url"] = BASE_URL

    client = AsyncHydraDB(**client_kwargs)
    print(f"== base_url={BASE_URL or "https://api.hydradb.com"} db={DB} collection={COLL}")

    # -- 0. database readiness -------------------------------------------------
    try:
        await client.databases.create(database=DB)
        print("databases.create -> ok (may already exist)")
    except Exception as e:
        print(f"databases.create -> {type(e).__name__}: {e}")
    for _ in range(40):
        st = await client.databases.status(database=DB)
        if st.data and st.data.infra and st.data.infra.ready_for_ingestion:
            print("database ready")
            break
        await asyncio.sleep(1)
    else:
        raise SystemExit("database not ready in 40s")

    # -- 1. ingest --------------------------------------------------------------
    memories = json.dumps(_items)
    resp = await client.context.ingest(
        type="memory", database=DB, collection=COLL, memories=memories, upsert="true"
    )
    print("\n-- ingest response --")
    print((resp.data or resp).model_dump(exclude_none=True) if resp else resp)

    ids = [i["id"] for i in _items]
    if resp and resp.data and resp.data.results:
        ids = [r.id or i["id"] for r, i in zip(resp.data.results, _items, strict=False)]
    for _ in range(60):
        st = await client.context.status(database=DB, collection=COLL, ids=ids)
        statuses = (st.data.statuses or []) if st and st.data else []
        if statuses and all(s.indexing_status == "completed" for s in statuses):
            print(f"indexed: {[(s.id, s.indexing_status) for s in statuses]}")
            break
        await asyncio.sleep(2)
    else:
        print(f"NOT indexed after 120s: {statuses}")
    print(json.dumps([s.model_dump() for s in statuses], indent=2))

    # -- 2. context.list (what does it return for content?) ----------------------
    # NOTE: SDK ListV2SourceListResponse is stale vs the wire: items live under
    # data.user_memories[] with `memory_id` (not `sources`/`id`), pagination+total at
    # data level. Parse the raw body.
    print("\n-- context.list (no filters) --")
    raw = await client.context.with_raw_response.list(database=DB, collection=COLL, type="memory", page=1, page_size=50)
    body = raw._response.json()
    d = body.get("data", {})
    print("data keys:", sorted(d.keys()))
    print(f"total={d.get('total')} success={d.get('success')}")
    print("pagination:", _fmt(d.get("pagination")))
    for src in d.get("user_memories", []):
        print("--- memory keys:", sorted(src.keys()))
        print(_fmt(src))

    # -- 3. query: no filter -----------------------------------------------------
    print("\n-- query 'where does the user live' (no filters, graph_context=True) --")
    res = await client.query(
        database=DB, collection=COLL, type="memory", query="where does the user live",
        query_by="hybrid", mode="thinking", max_results=10, graph_context=True,
        recency_bias=0.4,
    )
    qdata = res.data if res else None
    for i, c in enumerate(qdata.chunks or []):
        print(f"chunk[{i}] keys={sorted(c.model_dump(exclude_none=True).keys())}")
        print(_fmt(c.model_dump(exclude_none=True)))
    gc = qdata.graph_context if qdata else None
    if gc:
        print("graph_context keys:", sorted(gc.model_dump(exclude_none=True).keys()))
        for p in gc.query_paths or []:
            print("query_path:", p.model_dump(exclude_none=True))
        for p in gc.chunk_relations or []:
            print("chunk_relation:", p.model_dump(exclude_none=True))
    for s in qdata.sources or []:
        print("source:", _fmt(s.model_dump(exclude_none=True)))

    # -- 4. metadata_filters grammar experiments --------------------------------
    async def q(filters, op=None, label=""):
        print(f"\n-- query filters={_fmt(filters)} operator={op} {label} --")
        try:
            r = await client.query(
                database=DB, collection=COLL, type="memory",
                query="where does the user live", query_by="hybrid", mode="thinking",
                max_results=10, metadata_filters=filters, operator=op,
            )
            chunks = (r.data.chunks or []) if r and r.data else []
            for c in chunks:
                print(f"  [{c.id}] score={c.relevancy_score} title={c.source_title!r}")
        except Exception as e:
            print(f"  ERR {type(e).__name__}: {e}")

    await q({"kind": "fact"}, label="flat scalar")
    await q({"kind": "fact"}, "and", label="flat scalar + operator=and")
    await q({"additional_metadata": {"kind": "fact"}}, label="nested additional_metadata")
    await q({"kind": ["fact"]}, "or", label="flat array + operator=or")
    await q({"kind": ["fact", "session"]}, "or", label="flat multi array + operator=or")
    await q({"additional_metadata": {"kind": ["fact"]}}, "or", label="nested array + operator=or")
    await q({"question_id": "qprobe1"}, label="metadata question_id flat")
    await q({"metadata": {"question_id": "qprobe1"}}, label="nested metadata")

    # -- 5. context.list filters -------------------------------------------------
    print("\n-- context.list filters experiments --")
    for filt, label in [
        ({"additional_metadata": {"kind": "fact"}}, "nested additional_metadata"),
        ({"kind": "fact"}, "flat scalar"),
        ({"metadata": {"question_id": "qprobe1"}}, "nested metadata"),
        ({"additional_metadata": {"kind": "fact", "session_id": "s1"}}, "nested multi-key"),
        ({"metadata": {"kind": ["fact", "session"]}}, "nested array"),
    ]:
        try:
            r = await client.context.with_raw_response.list(
                database=DB, collection=COLL, type="memory", page=1, page_size=50, filters=filt
            )
            body = r._response.json()
            d = body.get("data", {}) or {}
            print(f"filters={_fmt(filt)} -> total={d.get('total')} n={len(d.get('user_memories', []))}")
            for src in d.get("user_memories", [])[:10]:
                print("   ", src.get("memory_id"))
        except Exception as e:
            print(f"filters={_fmt(filt)} -> ERR {type(e).__name__}: {str(e)[:200]}")

    # -- 6. metadata PATCH + verify ---------------------------------------------
    print("\n-- update_source_metadata PATCH (fact_alice_berlin -> superseded) --")
    patch = await client.context.update_source_metadata(
        "fact_alice_berlin", database=DB, collection=COLL,
        additional_metadata={"valid_to": "2023-07-01", "superseded_by": "fact_alice_munich"},
    )
    print(patch.model_dump(exclude_none=True) if patch else patch)

    print("\n-- context.list after PATCH (fact_alice_berlin) --")
    r = await client.context.with_raw_response.list(database=DB, collection=COLL, type="memory", page=1, page_size=50)
    d = r._response.json().get("data", {}) or {}
    for src in d.get("user_memories", []):
        if src.get("memory_id") == "fact_alice_berlin":
            print(_fmt(src))

    print("\n-- query again after PATCH --")
    await q({"kind": "fact"}, label="fact filter post-PATCH")

    # -- 7. context.inspect -------------------------------------------------------
    print("\n-- context.inspect(fact_alice_berlin) --")
    insp = await client.context.inspect(id="fact_alice_berlin", database=DB, collection=COLL)
    print(insp.model_dump(exclude_none=True) if insp else insp)

    print("\n== probe done ==")


if __name__ == "__main__":
    asyncio.run(main())
