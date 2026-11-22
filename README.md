# Hydrous

Agent memory that keeps time. Facts are bi-temporal and versioned: each one has a
validity interval, superseded versions stay retrievable, and retrieval can answer
what was true then, not only what is true now.

Hydrous runs on a local SQLite store, a local HydraDB OSS graph node, or an
optional HydraDB cloud adapter. You talk to it from Python, TypeScript, HTTP, or
MCP (Cursor / Claude Desktop).

```python
from hydrous import Memory

m = Memory(store="hydradb-oss")
m.add("I use a Dell XPS 13", user_id="alice", infer=False)
hits = m.search("laptop", filters={"user_id": "alice"})
```

Design details live in [ARCHITECTURE.md](ARCHITECTURE.md).

## What it does


| Capability               | Behavior                                                                                           |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| Bi-temporal facts        | Supersession closes `valid_to` on the old fact (stable id). Old versions stay queryable.           |
| Quality gate             | Extracted facts are scored; junk is rejected before storage; `quality_score` is kept for audit.    |
| Feedback-loop prevention | Content returned by `search()` is tagged; re-`add` of the same text is a `NOOP`.                   |
| As-of retrieval          | `reference_date` filters facts by validity interval (point-in-time search).                        |
| Safe abstention          | `chat()` answers from evidence, or returns the exact abstention string when evidence is missing.   |
| Memory graph             | Local HydraDB OSS stores entities, predicates, sessions, supersession edges, and multi-hop paths.  |
| Memory + knowledge       | Same database holds `type="memory"` and `type="knowledge"` with weighted `collections` search.     |
| Webhook indexing         | `hydrous serve --callback-url ...` registers `indexing.status_changed` (polling remains fallback). |
| Migration import         | `hydrous import-mem0` replays a memory export, mapping `created_at → recorded_at`.                 |




## Quickstart

```bash
uv sync
cp .env.example .env           # LLM API key, or use MockLLM / --mock
./scripts/hydradb_up.sh
uv run hydrous demo --mock --store hydradb-oss
uv run hydrous download-data
uv run hydrous eval --limit 10          # LongMemEval subset
uv run hydrous eval --limit 10 --mock   # offline dry run
uv run pytest
```



### Local HydraDB OSS graph

The `hydradb-oss` backend talks to the AGPL HydraDB graph engine over its local OpenCypher HTTP API. SQLite stays the hybrid text index; entities, facts, sessions, predicate edges, validity intervals, supersession, and multi-hop paths live in HydraDB.

```bash
./scripts/hydradb_up.sh
uv run hydrous demo --mock --store hydradb-oss
uv run hydrous add --store hydradb-oss --raw --user-id alice "I use a Dell XPS 13"
uv run hydrous search --store hydradb-oss --user-id alice laptop
```

`./scripts/hydradb_up.sh` creates `.hydradb/` (gitignored), starts
`ghcr.io/hydra-db/hydradb`, and persists the graph under `.hydradb/store`.
Readiness: `curl -fsS http://127.0.0.1:9090/readyz`. Use `--store hydradb` only
for the optional cloud adapter.

### Interactive playground

SVG graph of seeded facts, as-of slider, canned OpenCypher for the four demos,
one-click reseed. Serve over HTTP (file:// will not hit the API):

```bash
uv run hydrous serve --mock --port 8765
# open http://127.0.0.1:8765/
# with a local HydraDB container: add --store hydradb-oss
```

Source: `playground/index.html`.



### MCP for Cursor and Claude Desktop

`src/hydrous/mcp.py` is a stdio MCP server (JSON-RPC 2.0, Content-Length framing,
no extra packages). Tools: `memory_add`, `memory_search`, `memory_get_timeline`,
`memory_ask`, `memory_handoff`.

```json
{
  "mcpServers": {
    "hydrous": {
      "command": "uv",
      "args": ["run", "hydrous", "mcp", "--user-id", "my-workspace"]
    }
  }
}
```

Use `--store hydradb-oss` only after `./scripts/hydradb_up.sh` has succeeded.

### TypeScript SDK (`typescript/`)

```bash
cd typescript && npm test
cd ..
.venv/bin/hydrous serve --mock --port 8765
```

```ts
import { Memory } from "hydrous-ts";
const m = new Memory({ baseUrl: "http://127.0.0.1:8765" });
await m.add("I love hiking on weekends", { user_id: "alice" });
const hits = await m.search("hiking", { filters: { user_id: "alice" } });
```

Methods: `add` / `search` / `get_all` / `get` / `history` / `update` / `delete` /
`delete_all` / `reset`, plus `chat`, `add_knowledge`, `reference_date`, and
weighted `collections`. Dependency-free (global `fetch`).

### Python API

```python
from hydrous import Memory

with Memory(store="hydradb-oss") as m:
    m.add("I am planning to go to the market tomorrow", user_id="alice")
    hits = m.search(
        "what is alice planning to do tomorrow?",
        filters={"user_id": "alice"},
        top_k=3,
    )
    m.get_all(filters={"user_id": "alice"})
    m.get(memory_id)
    m.history(memory_id)          # full supersession chain
    m.update(memory_id, text="...")
    m.delete(memory_id)
```

Ids resolve from a JSON registry next to the store, so `get` / `history` /
`update` / `delete` survive process restarts. `history()` walks the supersession
chain in both directions.

`add(..., infer=False)` stores raw messages with no LLM call.
`infer=True` runs quality-gated extraction and stores only the extracted facts.

Explicit `delete()` / `delete_all()` / `reset()` are hard deletes (privacy).
"Append-only" means supersession never silently deletes a contradicted fact; it
closes the old validity interval instead.

## Four demos

`uv run hydrous demo` runs all four against HydraDB OSS (MockLLM by default).
`--real` uses your configured LLM for chat answers.

**1. Previous state from superseded facts**

```python
m.add("I use a Dell XPS 13", user_id="alice")
m.add("I switched to a MacBook Pro in March 2024", user_id="alice")
m.chat("what laptop did I use before the MacBook?", filters={"user_id": "alice"})
# -> "Dell XPS 13"
```

**2. As-of time travel**

```python
m.search("laptop", filters={"user_id": "alice"}, reference_date="2024-01-01")
# -> Dell (valid then); current search -> MacBook only
```

**3. Exact abstention when evidence is missing**

```python
m.chat("what did I eat for breakfast on 3 March 1999?", filters={"user_id": "alice"})
# -> "I don't know. The chat history does not contain this information."
```

**4. Multi-hop graph question**

```python
m.search("what city does my sister's partner work in?", filters={"user_id": "alice"})
# path evidence: user -> Anna -> London
```



## Quality gate

`EXTRACTION_SYSTEM` scores every candidate fact `0..1` and lists REJECT categories
(small talk, one-off questions, assistant boilerplate, transient state, session
logistics). Facts below `QUALITY_MIN_SCORE` (default 0.4) or flagged `"reject": true`
are dropped before storage. The score is kept on the fact. Extraction cache keys
include a prompt version so prompt edits invalidate stale results.

## Feedback-loop prevention

```python
hit = m.search("favorite movie", filters={"user_id": "u"})["results"][0]
m.add(hit["memory"], user_id="u")   # -> {"event": "NOOP", "id": <same id>}
```



## Importing an existing memory export

```bash
uv run hydrous import-mem0 export.json --user-id alice
uv run hydrous search "hiking" --user-id alice
```

Expects a `get_all`-shaped JSON export (`{"results": [{id, memory, metadata, created_at, user_id, ...}]}`). Default is raw replay (`created_at → recorded_at`);
`--infer` re-extracts. Exact re-import is idempotent via the feedback-loop hash.

## Memory and knowledge in one database

```python
m.add_knowledge("The hydrous engine stores bi-temporal facts", user_id="alice")
m.search(
    "bi-temporal facts",
    filters={"user_id": "alice"},
    collections=[
        {"name": "knowledge", "weight": 0.5},
        {"name": "memory", "weight": 1.0},
    ],
)
```

No `collections` argument means memory only.

## Webhook-driven indexing

```bash
hydrous serve --store hydradb \
    --callback-url https://your-tunnel.example --port 8765
```

Registers HydraDB's `indexing.status_changed` webhook. Ingest waits on events
(~0.5s wake) instead of busy-polling. Polling remains the timeout fallback.
`hydrous webhook-register URL` registers without serving.

## License

**AGPL-3.0** - see [LICENSE](LICENSE).

### Growtx Off Campus code
G5PBRH
