# Hydrous

**Agent memory layer for cross-session continuity — with a mem0-compatible API
that is contract-tested as a drop-in, plus the temporal machinery mem0's OSS
structurally cannot do.**

Hydrous stores *bi-temporal, versioned facts* (not flat strings): every memory
has a validity interval, superseded versions stay retrievable, and retrieval
can answer "what was true back then" instead of only "what is true now". It runs
on a local SQLite store or the HydraDB cloud adapter, and exposes itself in two
ways:

```python
from hydra_mem import Memory          # mem0 drop-in (see below)

m = Memory()                          # or Memory(store="hydradb", ...)
m.add("I use a Dell XPS 13", user_id="alice", infer=False)
```

**Read [ARCHITECTURE.md](ARCHITECTURE.md) — it is the single living doc**: problem,
design, algorithms, eval protocol, current status, and handoff notes.

---

## The pitch: where mem0 breaks, what Hydrous does instead

| mem0 OSS | Hydrous |
| --- | --- |
| Flat strings; a newer fact is a *new* row, the old one is still searched as current | Bi-temporal facts; supersession closes `valid_to` (PATCH, stable id); old versions stay retrievable on demand |
| Known junk-extraction problem ("98% junk", mem0 issue #4573) | Extraction **quality gate**: every fact is scored (`confidence`), junk is REJECTed before storage, the score is kept in metadata |
| Re-adding search results duplicates memory | **Feedback-loop prevention**: anything `search()` returns is tagged, re-`add` is a no-op (`event: "NOOP"`); deterministic sha1 fact ids dedupe exact re-extraction |
| `search(reference_date=...)` raises "platform-only" | `reference_date` **works**: true as-of time-travel over validated intervals |
| No answering at all (`chat()` raises `NotImplementedError`) | `chat()` retrieves + reads, and **abstains with the exact string** when evidence is missing — no hallucination |
| No multi-hop | HydraDB `graph_context` lanes merged into evidence (scored path contexts); nothing equivalent in mem0 |
| Memory only | **RAG + memory in one database**: multi-corpus `type="memory"\|"knowledge"` with per-scope `collections` weighting |
| Polls for indexing status | **Webhook-driven indexing**: `hydra-mem serve --callback-url ...` registers `indexing.status_changed` and wakes on events (polling kept as fallback) |
| No migration path | `hydra-mem import-mem0` replays a `get_all` export preserving `created_at → recorded_at` |

---

## Quickstart

```bash
uv sync
cp .env.example .env           # add your LLM API key (or use MockLLM / --mock)
uv run hydrous download-data
uv run hydrous eval --limit 10          # real LongMemEval run
uv run hydrous eval --limit 10 --mock   # deterministic dry run, no key needed
uv run pytest                            # all tests pass with no API key
```

### TypeScript SDK (`typescript/`)

```bash
cd typescript && npm test        # builds + E2E suite against a real server (MockLLM)
cd ..
.venv/bin/hydra-mem serve --mock --port 8765          # HTTP bridge for the SDK
```

```ts
import { Memory } from "hydrous-ts";
const m = new Memory({ baseUrl: "http://127.0.0.1:8765" });
await m.add("I love hiking on weekends", { user_id: "alice" });
const hits = await m.search("hiking", { filters: { user_id: "alice" } });
```

The TS SDK mirrors the mem0 surface (`add` / `search` / `get_all` / `get` /
`history` / `update` / `delete` / `delete_all` / `reset`) plus the hydrous
extensions (`chat`, `add_knowledge`, `reference_date`, weighted `collections`)
over the `hydra-mem serve` HTTP bridge. Dependency-free (global `fetch`).

### mem0 drop-in usage

Phase 2 ships `src/hydra_mem/api.py` — `Memory` / `AsyncMemory` with mem0's exact
surface (`add` / `search` / `get_all` / `update` / `delete` / `delete_all` /
`reset`), the same result shapes (`{"results": [...]}`, `{"message": ...}`), and
the same validation errors. The contract tests in `tests/test_mem0_compat.py`
are lifted from mem0's own README examples and v1.1+ docstrings — "drop-in" is a
tested claim:

```python
from hydra_mem import Memory

with Memory() as m:                     # local store; Memory(store="hydradb", ...) for cloud
    m.add("I am planning to go to the market tomorrow", user_id="alice")
    hits = m.search("what is alice planning to do tomorrow?",
                    filters={"user_id": "alice"}, top_k=3)
    # hits == {"results": [{"id": ..., "memory": ..., "score": ...}]}
    m.get_all(filters={"user_id": "alice"})      # {"results": [ ... ]}
    m.get(memory_id)                             # {"id", "memory", "hash", ...} or None
    m.history(memory_id)                         # full supersession chain (mem0 field names)
    m.update(memory_id, text="...")              # {"message": "Memory updated successfully!"}
    m.delete(memory_id)                          # {"message": "Memory deleted successfully!"}
```

Ids resolve from a **persisted registry** (a JSON sidecar next to the store), so
`get` / `history` / `update` / `delete` keep working after a process restart —
exactly like mem0 resolving ids from storage. `history()` is the versioned
showcase: for a fact it walks its slot's supersession chain in both directions,
returning mem0-shaped `{memory_id, old_memory, new_memory, event, ...}` records
(ADD per version, UPDATE pointing at the successor) — strictly richer than
mem0's flat `previous_value`/`new_value` log.

`add(..., infer=False)` stores raw messages with **no LLM call** — the cheap/fast
path, exactly mem0's semantics (one memory per non-system message). `infer=True`
runs quality-gated extraction and stores only the extracted facts (mem0 parity).

**Params we accept but don't implement yet** (accepted-and-logged, never crash):
`rerank`, `categories`, non-entity metadata filters, `memory_type`, `prompt`,
`show_expired`; `expiration_date` is stored in metadata but not yet enforced.
`timestamp` raises, exactly like mem0 OSS. `search()`/`get_all()` also accept the
legacy top-level `user_id=` form (mem0 1.1+ rejects it; accepted here with a log
note for old code). Scores are rank-mapped to [0.1, 1.0] — best hit 1.0, lowest
returned hit 0.1 — so the default `threshold=0.1` keeps every hit; the threshold
is a relative tail-trimmer, not an absolute similarity floor (the engine uses
RRF, not cosine). `explain=True` exposes the raw RRF via `score_details`.

**Deletion semantics:** explicit `delete()` / `delete_all()` / `reset()` are
honored as real hard deletes (user-initiated deletion is a privacy expectation).
"Append-only" means supersession never *silently* deletes a fact when a newer one
contradicts it — it closes the old fact's validity interval instead. You keep
full control; the system never deletes behind your back.

## Four demos mem0 structurally cannot do

`uv run hydra-mem demo` runs all four offline (MockLLM + SQLite); `--real` uses
your configured LLM for the chat answers.

**1. "What laptop did I use *before* the MacBook?" — answered from superseded facts.**
A fact is never deleted: supersession closes its validity interval, and
"previously / before / used to" questions retrieve the old version.

```python
m.add("I use a Dell XPS 13", user_id="alice")
m.add("I switched to a MacBook Pro in March 2024", user_id="alice")
m.chat("what laptop did I use before the MacBook?", filters={"user_id": "alice"})
# real LLM -> "Dell XPS 13"  (the superseded fact is still retrievable)
```

**2. `as_of="2024-01-01"` time-travel.** mem0 OSS raises `ValueError` on
`reference_date`; here it is real as-of filtering over validated intervals:

```python
m.search("laptop", filters={"user_id": "alice"}, reference_date="2024-01-01")
# -> the Dell fact (valid then); a current search -> only the MacBook fact
```

**3. No evidence → exact abstention, never a hallucination.**

```python
m.chat("what did I eat for breakfast on 3 March 1999?", filters={"user_id": "alice"})
# -> "I don't know. The chat history does not contain this information."  (verbatim)
```

**4. Multi-hop graph question.** On the HydraDB backend, `search` passes
`graph_context=True` and merges scored path contexts as `graph` evidence —
retrieval across a fact chain, which mem0 has nothing equivalent to:

```python
m.search("what city does my sister's partner work in?", filters={"user_id": "alice"})
```

## Quality gate (the junk-extraction fix)

`EXTRACTION_SYSTEM` (v2, `src/hydrous/prompts.py`) instructs the extractor to
score every candidate fact `0..1` and lists explicit REJECT categories with
negative few-shots (small talk, one-off questions, assistant boilerplate,
transient state, session logistics). `FactExtractor` drops anything below
`quality_min_score` (env `QUALITY_MIN_SCORE`, default 0.4) or explicitly flagged
(`"reject": true`), before storage. The score survives on the stored fact
(`quality_score` in the local sqlite column and in HydraDB
`additional_metadata`) for audit and later tuning. Extraction caching is keyed
on a prompt version, so prompt edits invalidate old extractions automatically.

## Feedback-loop prevention

Anything `search()` returns is tagged (content-hash → memory id). Calling
`add()` again with that exact content is a `NOOP` — no duplicate vectors, no
poisoned memory from an agent re-saving what it just read:

```python
hit = m.search("favorite movie", filters={"user_id": "u"})["results"][0]
m.add(hit["memory"], user_id="u")   # -> {"event": "NOOP", "id": <same id>}
```

## Migrating from mem0: `hydra-mem import-mem0`

```bash
uv run hydra-mem import-mem0 mem0_export.json --user-id alice
# raw replay by default: memory text is stored as-is, created_at -> recorded_at,
# user_id/agent_id scopes preserved; use --infer to re-extract facts instead.
uv run hydra-mem search "hiking" --user-id alice
```

The export format is mem0's `get_all` JSON (`{"results": [{id, memory, metadata,
created_at, user_id, ...}]}`). Exact re-import is idempotent (feedback-loop
no-op). Together with the compat API, migration is zero-downside: keep your
mem0 call sites, swap the import, replay your data.

## RAG + memory in one database

The unified multi-corpus story: the same database holds the scope's **memory**
bucket (`type="memory"`) and a **knowledge** corpus (`type="knowledge"`),
searched together with per-scope `collections` weights:

```python
m.add_knowledge("The hydrous engine stores bi-temporal facts", user_id="alice")
m.search("bi-temporal facts", filters={"user_id": "alice"},
         collections=[{"name": "knowledge", "weight": 0.5},
                      {"name": "memory", "weight": 1.0}])
```

No `collections` = memory bucket only (mem0 behavior). On HydraDB the weighted
shape maps to the query API's `collections` dict; on the local store the buckets
are isolated `question_id` rows with weight-scaled lane ranks merged by RRF.

## Webhook-driven indexing (replaces polling)

When you have a public callback URL, `hydra-mem serve` registers HydraDB's
`indexing.status_changed` webhook and ingests stop polling — the server's
`/webhooks/hydradb/indexing` endpoint marks the shared tracker and waits wake
within ~0.5s:

```bash
hydra-mem serve --store hydradb \
    --callback-url https://your-tunnel.example --port 8765
# (tunnel the port to a public https URL, e.g. a reverse proxy or ssh -R)
```

Plain polling remains the fallback when no tracker is attached or on timeout.
`hydra-mem webhook-register URL` registers the webhook without serving.

## LongMemEval comparison

See [docs/comparison.md](docs/comparison.md) for the table and full methodology:
GPT-4o baseline 60.6%, oracle 87.0%, mem0's published (platform) 94.4 with their
protocol caveats; Hydrous's full 500-instance run is in progress (pilot subsets
and this repo's status are documented there).

## License

**AGPL-3.0** (see [LICENSE](LICENSE)). mem0's OSS is **Apache-2.0** — note the
difference before vendoring Hydrous as a drop-in replacement in commercial /
closed-source deployments: AGPL's copyleft obligations apply to the combined
work. (Everyone is free to use it for personal, research, and internal tooling.)
