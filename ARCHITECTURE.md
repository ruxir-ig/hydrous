# Hydrous — Architecture (THE single living doc)

Agent memory layer for cross-session continuity.
This is the ONLY design doc. Keep it updated as the code changes. No other .md files except README.md (a pointer here).

## Problem statement

Answer questions over chat histories spanning 30–40 sessions / ~115K tokens per question
(LongMemEval-S benchmark). Requirements:

- Synthesize facts across sessions; maintain chronological reasoning.
- Track information that was later overwritten (**knowledge updates**) — bi-temporal versioning.
- **Abstain** when the answer is not in the history (30 `_abs` items in the benchmark).
- Baseline to beat: GPT-4o full-history = 0.606 accuracy on LongMemEval-S; oracle evidence = 0.870.

## Design goals

1. **Working end-to-end pipeline first** (ingest → retrieve → answer → eval JSONL), extras later.
2. **Provider-agnostic LLM client** — OpenAI / OpenRouter / custom endpoint via env config; every
   LLM call goes through one client. `MockLLM` makes ALL tests deterministic with no API key.
3. **Pluggable storage** — `MemoryStore` ABC; default local store (SQLite + dense embeddings +
   BM25); HydraDB cloud adapter behind the same interface.
4. **Bi-temporal fact versioning** — never delete; close validity intervals on supersession.
5. **Safe abstention** — strict evidence-sufficiency gate before answering.
6. **Cheap iteration** — extraction cached to disk; eval runner resumable; subset runs (`--limit`,
   `--type`, `--ids`).

## System diagram

```
            LongMemEval JSON (data/)
                    │
            ingest/longmemeval.py            (streams JSON array, strips has_answer)
                    │
                    ▼
            ingest/extractor.py              per session, chronological order:
                    │                        (a) store raw turns  (b) LLM-extract atomic facts
                    │                        (c) supersession check vs existing facts
                    │                        (d) session summary   → cached to disk per (qid, sid)
                    ▼
        ┌───────────────────────┐
        │      MemoryStore      │  base.py ABC
        │  local.py | hydradb.py│
        │ SQLite+emb+BM25 | cloud│
        └───────────────────────┘
                    │
                    ▼
            retrieve/pipeline.py   query analysis (time-aware rewrite) → hybrid search
                    │              (dense + BM25 over facts AND sessions) → as-of-date
                    │              validity filter → dedupe/rerank → chronological EvidencePack
                    ▼
            answer/reader.py       sufficiency gate (CRAG-style) ──insufficient──► ABSTAIN
                    │ sufficient
                    ▼
            Chain-of-Note reader (question_date always passed) → final answer
                    │
                    ▼
            eval/runner.py → hypotheses JSONL → eval/judge.py (official prompts, GPT-4o judge)
                    │
                    ▼
            eval/metrics.py → overall + per-question_type + abstention-subset accuracy
```

## Module map (`src/hydrous/`)

| Module | Responsibility |
|---|---|
| `config.py` | pydantic-settings, env-driven (`Settings`), provider presets, `build_llm`, `build_embedder` helpers |
| `schema.py` | `Turn`, `Session`, `Fact` (bi-temporal), `Evidence`, `EvidencePack`, `MemoryItem`, `EvalInstance`, `ABSTAIN_STRING` |
| `text_utils.py` | tokenization, tolerant date parsing/normalization, word-overlap helpers |
| `prompts.py` | ALL prompt templates (extraction, summary, query analysis, conflict, sufficiency, reader). Each starts with a stable `TASK:` marker line that `MockLLM` dispatches on. |
| `llm.py` | `LLMClient` protocol, `OpenAICompatibleLLM` (async, retry/backoff), `MockLLM` (scripted FIFO + deterministic heuristic handler) |
| `embeddings.py` | `Embedder` protocol, `SentenceTransformerEmbedder` (lazy), `TfidfEmbedder` (pure-numpy hashing, stateless, default fallback) |
| `ingest/longmemeval.py` | streaming JSON-array loader (`iter_instances`), `sessions_from_instance` (strips `has_answer`, sorts by date) |
| `ingest/extractor.py` | `FactExtractor`: LLM fact extraction + summary per session, disk cache, supersession resolution, **quality gate** (scored facts, REJECT before storage), prompt-versioned cache |
| `store/base.py` | `MemoryStore` ABC (incl. `delete_memory` for the compat layer) |
| `store/local.py` | `LocalMemoryStore`: SQLite + dense embeddings + BM25 hybrid (RRF), temporal filtering; `recorded_at`/`metadata` on sessions, `quality_score` on facts |
| `store/hydradb.py` | `HydraDBStore`: cloud adapter, same interface (import-guarded) |
| `retrieve/pipeline.py` | `RetrievalPipeline`: query analysis → hybrid search → as-of filter → chronological `EvidencePack` |
| `answer/reader.py` | `Answerer`: sufficiency gate → CoN reader → answer or exact abstention string |
| `eval/runner.py` | `EvalRunner`: iterate instances → ingest (cached) → answer → hypotheses JSONL; resumable |
| `eval/judge.py` | official judge prompts verbatim, configurable judge model, `.log` output |
| `eval/metrics.py` | overall + per-`question_type` + abstention-subset accuracy table |
| `cli.py` | typer CLI: `download-data` / `ingest` / `answer` / `eval` / `judge` / `report` |

## Phase 2 surface (`src/hydra_mem/`)

| Module | Responsibility |
|---|---|
| `api.py` | mem0-compatible `Memory` / `AsyncMemory` (mem0 v1.1+ contract): `add` (raw `infer=False` path / quality-gated facts), `search` (hybrid + `reference_date` as-of), `get_all`, `update`, `delete`, `delete_all`, `reset`, `chat` (reader + exact abstention), `import_mem0`. Accept-and-log: `rerank`, `categories`, metadata filters, `memory_type`, `prompt`, `expiration_date` (stored, not enforced), `show_expired`. Feedback-loop: search results tagged by content hash → re-`add` is a `NOOP`. Entity scopes map to `mem0_u_<id>` store buckets; a background event-loop thread owns the store for the sync facade. |
| `importer.py` | `load_mem0_export`: validates the `get_all` export file shape |
| `cli.py` | `hydra-mem` typer CLI: `add` / `search` / `get-all` / `import-mem0` / `demo` (the four positioning demos) |

## Data schemas

```python
Turn(role: str, content: str)                      # has_answer NEVER stored
Session(session_id, date, turns, summary=None,
        recorded_at=None,   # transaction time (import-mem0: mem0 created_at)
        metadata={})        # free-form user metadata (mem0 add(metadata=...))

Fact(                                               # bi-temporal, append-only
    id, question_id, subject, predicate, value,
    fact_date,            # when the fact became true (LLM-extracted, else session date)
    session_id,           # provenance
    valid_from,           # start of validity interval (= fact_date)
    valid_to: None,       # None = currently valid; set when superseded
    recorded_at,          # wall-clock ingestion time (system time)
    superseded_by: None,  # id of the fact that replaced this one
    quality_score: None,  # extraction confidence 0..1 from the quality gate
)
Fact.text == f"{subject} {predicate}: {value}"      # what gets embedded/indexed
Fact.valid_as_of(date) == valid_from <= date < valid_to (or valid_to is None)

Evidence(source: "fact"|"session", text, date, session_id, score, fact_id=None)
EvidencePack(question, question_date, items)        # .chronological(), .to_prompt()

MemoryItem(id, title, text|user_assistant_pairs, infer, metadata: JSON-str, additional_metadata)
                                                    # HydraDB ingest item shape
EvalInstance                                        # pydantic model of one LongMemEval row
```

## Algorithms

### 1. Ingestion (`FactExtractor.ingest_question`)

1. Sessions sorted chronologically (dataset S is already sorted; we sort defensively).
2. Extraction is **order-independent**, so all sessions are extracted concurrently
   (semaphore = `MAX_CONCURRENCY`, default 8): per session, LLM → JSON array of
   `{subject, predicate, value, fact_date, confidence}` + a 2–3 sentence summary. The
   **quality gate** (Phase 2) drops facts below `QUALITY_MIN_SCORE` (default 0.4) or
   explicitly flagged `reject`; the score is stored as `Fact.quality_score`. Results
   cached at `cache/extractions/{qid}__{sid}.json` (invalidated on model OR
   extraction-prompt-version change) — reruns are free.
3. **Versioning is order-dependent**, so after extraction, sessions are applied sequentially in
   date order: `add_session` (raw turns + summary) → for each new fact, find candidate conflicts
   among *currently valid* facts: same normalized `(subject, predicate)`, or same subject with
   embedding cosine ≥ 0.75 → LLM conflict check → if "yes": `old.valid_to = new.fact_date`,
   `old.superseded_by = new.id` (NEVER delete) → `add_facts`.

### 2. Retrieval (`RetrievalPipeline.retrieve`)

1. **Query analysis** (LLM): rewrite question for retrieval, resolve relative time expressions
   against `question_date`, emit subqueries + optional time range. Falls back to identity on
   parse failure.
2. **Hybrid search** per query: dense cosine (L2-normalized embeddings) + BM25, merged with
   reciprocal rank fusion (k=60), over BOTH facts and session chunks (fact-augmented indexing:
   session text is indexed together with its extracted facts' text).
3. **As-of-date filter**: facts must satisfy `valid_as_of(question_date)` — superseded versions
   are excluded, so knowledge-update answers use the latest value. Exception: if the question
   asks about a *previous* state ("did I previously...", "used to..."), superseded facts are
   re-included; raw sessions are always searchable and retain old statements verbatim.
4. Dedupe by id (max score), top-k (`TOP_K`, default 12), then **chronological ordering** for
   the reader prompt.

### 3. Answering (`Answerer.answer`)

1. Empty evidence pack → abstain immediately (no LLM call).
2. **Sufficiency gate** (CRAG-style, strict, `USE_GATE=1`): LLM judges whether the evidence
   explicitly contains the asked information. "no" → return the exact abstention string.
   **FINDING (2026-08-15, final): the LLM verdict gate is unreliable with deepseek-v4-flash
   and is a net negative** — it says "no" even when gold-only evidence is in the prompt, and it
   abstains on exactly the questions the reader also fails (gpt4, 6d550036). Gate-ON and
   gate-OFF both land at 60% locally, so the gate adds no accuracy while risking false abstentions.
   Recommended disposition: keep the empty-pack heuristic, drop the LLM verdict, and rely on the
   reader's own abstention marker (step 3). The `gate_prompt` compact projection (12 × 8000, truncate
   raw before collapse) is retained for reference but not recommended for the current model.
3. **CoN reader**: official-style prompt — extract relevant info step by step, then reason;
   `question_date` always passed; explicit instruction to answer with the abstention string if
   evidence is insufficient. Reader output containing "don't know"/"does not contain" is
   normalized to the exact abstention string.

Abstention string (exact): `I don't know. The chat history does not contain this information.`

### 4. Eval

- `eval/runner.py`: streams instances, filters (`--limit`, `--type`, `--ids`, `--exclude-abs`),
  ingests (skip if already ingested), answers, appends `{"question_id", "hypothesis"}` JSONL.
  Resumable: question_ids already in the output file are skipped.
- `eval/judge.py`: official LongMemEval judge prompts **verbatim** (default / temporal-reasoning
  / knowledge-update / single-session-preference / abstention for `_abs` ids), judge model
  configurable (`JUDGE_MODEL`, default `gpt-4o-2024-08-06`, temp 0). Label = `'yes' in
  response.lower()`. Writes `<hypotheses>.log` JSONL.
- `eval/metrics.py`: overall accuracy, per-`question_type`, abstention subset (`_abs` ids).

## LLM configuration (provider-agnostic)

Env-driven (`.env`, see `.env.example`):

- `LLM_PROVIDER=openai|openrouter|custom|mock`
- openai: `OPENAI_API_KEY` (+ optional `LLM_BASE_URL`)
- openrouter: `OPENROUTER_API_KEY`, base `https://openrouter.ai/api/v1`
- custom: `LLM_BASE_URL` + `LLM_API_KEY` (or `OPENAI_API_KEY`)
- `LLM_MODEL` (worker), `JUDGE_MODEL` (judge)
- `mock`: `MockLLM`, no key, deterministic — used by all tests and `--mock` CLI runs.

`MockLLM` modes: (a) scripted FIFO responses for unit tests; (b) default deterministic heuristic
handler dispatched on the `TASK:` marker of each prompt (extraction → one fact per user turn;
conflict → "yes" iff values differ; sufficiency → content-word overlap; reader → best-overlap
evidence echo; judge → overlap / IDK detection). Every LLM call is recorded in `.calls`.

## Storage

### Local (default)

SQLite at `STORE_PATH` (default `data/hydrous.sqlite3`). Tables: `sessions`, `turns`,
`facts`, `meta`. Embeddings stored as float32 blobs; per-question corpus is small (~40 sessions,
≤ a few hundred facts) so search loads that question's rows, computes cosine in numpy, and
builds BM25 on the fly. Hybrid merge = lane-separated RRF (facts vs sessions compete within
their own lane; facts weight 0.4) — prevents short fact snippets from drowning gold sessions.
Per-item evidence window: 8000 chars (`EvidencePack.to_prompt`).

Embedding backends (`EMBEDDING_BACKEND`):
- `sentence-transformers` (default when importable): `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`.
- `tfidf`: pure-numpy stateless hashing TF-IDF (4096-d, sublinear tf, L2-normalized). Used by
  tests (no network/model download) and as automatic fallback.

### HydraDB (optional extra `[hydra]`)

`HydraDBStore` implements the same `MemoryStore` interface. One collection per LongMemEval
instance (`qid_{question_id}`).

- Facts ingest with `id` upsert key; the full Fact is reconstructable from `additional_metadata`
  (`subject`/`predicate`/`value`/`fact_date`/`session_id`/`valid_from`/`valid_to`/`superseded_by`).
  Supersession = `context.update_source_metadata` PATCH on the stable id (probe-verified to sync
  to the vector store — query-time `metadata_filters` on patched keys match; no re-ingest).
- Sessions ingest as `user_assistant_pairs` (server requires text XOR pairs and caps
  `additional_metadata` at 1024 bytes). Long sessions split into parts (`{qid}:{sid}`,
  `{qid}:{sid}#1`, ...) to stay under the ~1000 "memory_tokens" per-request budget; `get_sessions`
  reconstructs turns from `context.inspect` (single pair → "USER:/ASSISTANT:" lines, multi-pair →
  `{"pairs": [...]}` JSON envelope). `_ingest_items` chunks payloads by estimated cost and paces
  concurrent ingests through a rolling 5000/min token window; 429s are retried and oversized
  payloads are shrunk + retried.
- Query: `query_by="hybrid"`, `mode="thinking"`, `recency_bias=0.4`, `graph_context=True`
  (multi-hop paths merged into the evidence list, scored relative to the best chunk). `valid_as_of`
  filtering is applied client-side over chunk `additional_metadata` (server `metadata_filters` is
  set-equality, which can't express date intervals).
- SDK types are stale in places (`context.list` wire shape: `data.user_memories[]` with
  `memory_id`; ingest `metadata` must be a map, not the SDK's `str`) — the adapter builds payloads
  as plain dicts and reads raw HTTP bodies for list. Integration tests run only when
  `HYDRA_DB_API_KEY` is set.

## How to run

```bash
uv sync                              # create venv + install deps
cp .env.example .env                 # set LLM_PROVIDER + key + model
uv run hydrous download-data       # fetch LongMemEval oracle + s_cleaned into data/

# deterministic smoke run, NO api key needed:
uv run hydrous eval --dataset data/longmemeval_oracle.json --limit 5 --mock --out hyp_mock.jsonl
uv run hydrous judge --hypotheses hyp_mock.jsonl --dataset data/longmemeval_oracle.json --mock
uv run hydrous report --log hyp_mock.jsonl.log

# real run (subset first!):
uv run hydrous eval --limit 10 --type knowledge-update --out hyp.jsonl
uv run hydrous judge --hypotheses hyp.jsonl --dataset data/longmemeval_s_cleaned.json
uv run hydrous report --log hyp.jsonl.log

# same subset against the HydraDB cloud store (HYDRA_DB_API_KEY in .env):
uv run hydrous eval --id 6a1eabeb --id 6d550036 --id gpt4_59149c77 --store hydradb --reingest --out hyp_hydradb.jsonl

uv run pytest                        # all green with MockLLM, no network, no key
uv run ruff check
```

## Current status / TODO

- [x] Scaffold, env config, packaging (uv, Python 3.14.6 venv via `.python-version`)
- [x] ARCHITECTURE.md first draft
- [x] Core: schema / config / llm / MockLLM + tests
- [x] Dataset download + streaming loader (oracle 15MB + s_cleaned 277MB in `data/`)
- [x] Local store + extraction + versioning + tests
- [x] Retrieval + reader + abstention gate + tests
- [x] Eval runner + judge + metrics + CLI
- [x] HydraDB adapter (interface-complete; integration test skipped without HYDRA_DB_API_KEY)
- [x] E2E smoke on oracle with MockLLM (CLI-verified: eval → judge → report, resume works)
- [x] `ruff check` clean; `pytest` green
- [x] HydraDB live validation on the real cloud API (probe + full-interface verify +
      mock LongMemEval subset: 3 ids, all answered; parts ingest, PATCH supersession,
      graph-context search, rate-limit pacing)
- [ ] Real-API judged subset (LLM auth + free-tier quota permitting; ≈5-6 ids fit
      the 3 free accounts' ~300k token budget)
- [x] Phase 2 — mem0-compat `Memory`/`AsyncMemory` (`src/hydra_mem/api.py`) with contract
      tests lifted from mem0's own README examples + v1.1+ signatures; accept-and-log
      (`rerank`, `categories`, metadata filters, `memory_type`, `prompt`, `expiration_date`)
- [x] Phase 2 — extraction quality gate (scored facts via negative few-shots in
      `EXTRACTION_SYSTEM` v2, REJECT path, `quality_score` kept in metadata, prompt-versioned cache)
- [x] Phase 2 — feedback-loop prevention (search results tagged by content hash; re-`add` = NOOP;
      sha1 fact ids dedupe exact re-extraction)
- [x] Phase 2 — `hydra-mem import-mem0` (mem0 `get_all` export replay, `created_at → recorded_at`)
- [x] Phase 2 — README positioning: four "mem0 cannot do" demos (`hydra-mem demo`), AGPL-vs-Apache
      license flag; LICENSE (AGPL-3.0) added
- [x] Phase 3 — live cloud re-probe (quality_score / meta+recorded_at round-trips,
      `delete_memory`, graph_context) — all green on the real API
- [x] Phase 3 — PyPI packaging readiness (names free; single dist `hydrous` ships both
      packages; license metadata + classifiers; `uv build` + twine check PASSED; not published)
- [x] Phase 3 — webhook-driven indexing (IndexingTracker + server /webhooks/hydradb/indexing;
      `HydraDBStore._wait_indexed` webhook mode with polling fallback;
      `register_indexing_webhook` + `hydra-mem serve --callback-url` / `webhook-register`)
- [x] Phase 3 — multi-corpus `type="knowledge"|"memory"` + per-scope `collections` weighting
      (SDK `Dict[str, float]` shape; local weighted RRF merge; compat `add_knowledge` + `search(collections=...)`)
- [x] Phase 3 — TypeScript SDK (`typescript/`, package hydrous-ts): mem0 surface over the
      HTTP bridge, dependency-free, strict TS, 7 E2E tests green (npm test)
- [x] Phase 3 — docs site (`docs/index.html` + `docs/comparison.md`) + README Phase-3 sections
- [ ] Full 500-instance run + score vs 0.606 baseline / 0.870 oracle (started, paused on
      endpoint 504s — resumable at data/hyp_phase3_500.jsonl; see docs/comparison.md)

## Handoff notes for agents

(Append decisions + gotchas here as they happen; newest first.)

- 2026-08-18: Phase 3 — grow it into the product (in progress). Decisions + gotchas:
  - **Full 500-instance eval PAUSED by user decision**: the LLM endpoint
    (code.in2peta.com deepseek-v4-flash) serializes under load and 504s at
    MAX_CONCURRENCY=12; measured ~17 min/question → ~25h wall for 500. Killed the
    run at 1/500 (data/hyp_phase3_500.jsonl is resumable; resume command documented
    in docs/comparison.md). Everything else in Phase 3 continues; revisit the run
    when the endpoint recovers. Comparison table (docs/comparison.md) publishes
    baseline 60.6 / oracle 87.0 / mem0 platform 94.4 (with mem0's verbatim
    platform caveat; their new algorithm is single-pass ADD-only — our append-only
    design) + methodology.
  - **Live cloud re-probe green** (scripts/probe_phase3.py): quality_score round-trips
    (0.87/None), session recorded_at + meta round-trips, delete_memory via
    context.delete works, graph_context object present (0 paths on the tiny corpus —
    expected). Probe uses its own DB + cleans up; run only against a real key.
  - **PyPI-ready**: name free (hydrous/hydra-mem confirmed 404 on PyPI). Single
    distribution "hydrous" ships BOTH packages (hydra_mem imports hydrous internals;
    two dists would force a cross-dist dep) + `hydrous`/`hydra-mem` entry points.
    `uv build` + `uvx twine check` PASSED; wheel carries LICENSE + both packages.
    Not published (no token; user-run step).
  - **Webhook-driven indexing**: hydra_db SDK has per-org `webhooks.register(
    event_types=["indexing.status_changed"], url=...)`. `hydra_mem.indexing.
    IndexingTracker` is a thread-safe {(db, coll, id): status} handshake; the
    server's /webhooks/hydradb/indexing marks it; `HydraDBStore._wait_indexed`
    clears-then-waits on it (fresh marks only; stale completed from a previous
    ingest must not satisfy the new one) and returns with NO status call when all
    sources are marked; polling is the timeout fallback. `register_indexing_webhook`
    on store + `hydra-mem serve --callback-url` / `hydra-mem webhook-register`.
    Webhook payload shape is best-effort tolerant (`ids|id|memory_id|source_id|data[]`);
    the real payload was never observed (needs a public URL) — treat field names as
    provisional until a live delivery is inspected.
  - **HTTP server is stdlib** (`hydra_mem/server.py`, ThreadingHTTPServer) — no
    fastapi/uvicorn dep. Routes: POST /add /search /get_all /update /delete /
    /delete_all /reset /chat /add_knowledge /import-mem0; GET /get/{id} /history/{id};
    POST /webhooks/hydradb/indexing; GET /health. Error mapping: 400 TypeError,
    422 ValueError (mem0-parity validation), 501 NotImplementedError, 404 unknown
    route, 404 for get() of unknown id (mem0 parity: get returns null over HTTP).
  - **Multi-corpus + collections weighting**: SDK verified — ingest
    `type=Literal['knowledge','memory']`; query `collections` accepts
    `Dict[str, float]` weighted shape and must NOT be sent with a single
    `collection` key (we pop it). Memory bucket = `qid_{scope}`, knowledge bucket =
    `qid_{scope}_knowledge`. Compat: `add_knowledge(text|list)` + `search(
    collections=[{"name": "memory"|"knowledge"|raw, "weight": w}])`; friendly names
    translate via `_collection_bucket`. Local store: knowledge docs are session rows
    in the `_knowledge` question_id bucket; weighted merge = per-bucket lane RRF
    scaled by weight, dedupe by evidence key. weight=0 excludes (BUG CAUGHT BY TEST:
    `c.get("weight") or 1.0` turned 0.0 into 1.0 — use explicit None check).
    delete_all/reset also wipe the `_knowledge` bucket.
  - **TS SDK** (typescript/, package "hydrous-ts", AGPL): dependency-free client
    (global fetch) mirroring the mem0 surface over the HTTP bridge; strict TS,
    tsc → dist (ESM + .d.ts); tests are node:test spawning the REAL python server
    (typescript/test/server_helper.py — must use .venv/bin/python, system python3
    lacks deps) and exercising the full surface E2E (7 tests green). Server needs
    the `/add_knowledge` route (added AFTER first build — POST_ROUTES must include
    every compat method the SDK exposes).
- 2026-08-17 (Phase 2.1 patch — review closure on Phase 2): Decisions + gotchas:
  - **`get(memory_id)` + `history(memory_id)` added** — mem0 parity shapes, verified
    against mem0's source: `get` returns the item dict or **None** (does NOT raise);
    `history` returns a **list** with mem0's history-table field names
    (`memory_id / old_memory / new_memory / event / created_at / updated_at /
    is_deleted / actor_id / role`), empty list when unknown. For facts, `history`
    walks the slot's FULL supersession chain via `get_timeline` (both directions):
    ADD per version + UPDATE pointing at the successor — strictly richer than
    mem0's flat log. Raw sessions: single ADD.
  - **Durable id resolution**: `_IdRegistry` JSON sidecar (atomic tmp+os.replace)
    next to the store (`{store_path}.ids.json`; hydradb: `data/hydra_mem_ids.json`,
    overridable via `ids_path=`). Every add/search/get_all/import records
    id→scope (+ scope entity ids so `get()` can rebuild user_id/agent_id/run_id).
    `delete`/`update`/`get`/`history` now resolve across process restarts; unknown
    ids pop stale registry entries before raising. `reset()` wipes EVERY scope in
    the registry (other instances' scopes included) — mem0 wipe-everything
    semantics; `delete_all` uses `drop_scope` (returns removed ids so the
    feedback-loop `_seen` map can be filtered).
  - `Memory.close()` now releases the store (sqlite connection) ON its event-loop
    thread before stopping the loop (`AsyncMemory.close()` + `_closed` flag) —
    previously the sqlite conn leaked.
  - **Search score floor**: min-max normalized to [0.1, 1.0] (single/uniform batch
    → 1.0). Rationale: with [0, 1] mapping the lowest-ranked hit normalized to 0.0
    and the DEFAULT threshold 0.1 always dropped it — a 2-result search returned 1.
    Now the default keeps every hit (mem0's default rarely trims either); threshold
    stays a relative tail-trimmer. Scores remain rank-based, not cosine.
  - **README deletion framing**: explicit `delete`/`delete_all`/`reset` are real
    hard deletes (privacy); "append-only" means supersession never SILENTLY deletes.
- 2026-08-17: Phase 2 — competitor surface. Decisions + gotchas:
  - **mem0 contract target = v1.1+** (current `mem0/memory/main.py`): `search`/`get_all` scope
    via `filters={"user_id": ...}` and REJECT top-level entity kwargs; `add` keeps top-level
    `user_id=...`; `reference_date` raises "platform-only" in mem0 OSS. We implement
    `reference_date` as as-of time travel (killer demo); we also ACCEPT the legacy top-level
    `user_id=` in `search`/`get_all` (logged) so pre-1.1 code keeps working.
  - **Compat scope mapping**: mem0 entities → sanitized store bucket
    `mem0_u_<user_id>_a_<agent_id>...` (one HydraDB collection / sqlite question_id per scope).
    Memory ids resolve within the process that created/searched them (`AsyncMemory._ids`);
    `delete()`/`update()` raise mem0's exact "Memory with id ... not found" otherwise.
  - **infer=True parity**: mem0 stores ONLY extracted memories (no raw conversation) — same here;
    quality-gated facts, `store_facts(..., session=None)`. `infer=False`: one raw session per
    non-system message (mem0's raw path), no LLM call at all.
  - **Feedback loop**: `_seen` content-hash → id, tagged on every search result; re-add of that
    content returns `event: "NOOP"` with the original id. Cross-process raw dedupe is NOT
    implemented (mem0 doesn't dedupe raw either); fact dedupe is durable via (s,p,v) check
    against `get_facts`. sha1 fact ids dedupe exact same-session re-extraction at the eval layer.
  - **Scores**: hydrous uses RRF, not cosine. Compat `search` min-max-normalizes to [0,1]
    (single-result batch → 1.0; threshold is a relative floor) and exposes the raw RRF in
    `score_details` when `explain=True`. Never emit np.float64 in results (not JSON-serializable).
  - **Sync facade**: `Memory` runs a single daemon event-loop thread (`asyncio.run_coroutine_threadsafe`);
    the store/LLM are created lazily INSIDE that loop (sqlite conns + AsyncHydraDB clients are
    loop-bound; creating them on the caller thread breaks sqlite's check_same_thread). Use
    `with Memory(...) as m` / `m.close()`.
  - **Quality gate**: `EXTRACTION_SYSTEM` v2 adds `confidence` + REJECT few-shots; changes are
    automatically cache-invalidated via `EXTRACTION_PROMPT_VERSION` (cache payload carries it).
    Old cached extractions without a version are misses → re-extract. `Fact.quality_score`,
    `Session.recorded_at`/`Session.metadata` added to schema + local sqlite
    (`ALTER TABLE` migrations; legacy rows fall back recorded_at → date) + HydraDB
    `additional_metadata` (respect 1024B cap: truncate `meta` to ≤700 chars, keep `recorded_at`
    on part 0).
  - **Accepted-but-ignored params** (log, never crash): `rerank`, `categories`, non-entity
    metadata filters, `memory_type`, `prompt`, `show_expired`; `expiration_date` is stored in
    metadata but NOT enforced yet; `timestamp` raises like mem0. `update()` on an EXTRACTED fact
    raises NotImplementedError (delete + re-add; supersession covers replacement).
  - **import-mem0**: raw replay by default (no LLM), `created_at → recorded_at`, scope per item's
    user_id/agent_id/run_id else `--user-id`; idempotent re-import via the feedback-loop hash.
  - **License**: added LICENSE (AGPL-3.0, SPDX canonical text). mem0 OSS is Apache-2.0 — README
    flags the copyleft difference for anyone vendoring as a drop-in.
  - **Demo determinism trap**: seeding the `hydra-mem demo` store bypasses the LLM (direct
    `Fact` writes with explicit valid intervals) so demos 1–3 are deterministic offline; the
    mock reader is word-overlap-based so "before"-style answers need `--real`. `store_facts`,
    `delete_question` etc. are async — any sync demo code must `asyncio.run(...)` them.
- 2026-08-14: Initial build complete, all tests green. Decisions + gotchas:
  - **Python 3.14 works**: torch 2.13.0 (CPU wheel index) + sentence-transformers 5.7.0 install
    and run cleanly on 3.14.6. `.python-version` pins 3.14 so uv doesn't pick a managed 3.11.
    torch on Linux is pinned to the pytorch-cpu index in pyproject.toml (avoids ~2.5GB CUDA wheels).
  - **Dataset quirks found in the real files**: (1) some gold `answer` values are NOT strings
    (ints, e.g. counting questions) — `EvalInstance._stringify_answer` coerces them; (2) dates
    are `2023/04/10 (Mon) 23:07` format — `text_utils.parse_date` handles it; (3) sessions are
    NOT always sorted (oracle file had descending pairs) — `sessions_from_instance` sorts
    defensively; (4) `has_answer` is stripped by the pydantic `Turn` model (extra keys ignored).
  - Async `LLMClient` protocol; store interface also async (uniform with HydraDB; local impl is
    sync sqlite inside async methods). Tests use `asyncio.run` via `tests/conftest.py:run` —
    no pytest-asyncio dependency.
  - Prompts centralized in `prompts.py` with `TASK:` marker lines; `MockLLM` dispatches on them.
    If you add a prompt, add a marker AND a mock handler branch, or tests will break.
  - Extraction cache (`cache/extractions/{qid}__{sid}.json`) is invalidated on MODEL or
    extraction-prompt-version change (Phase 2) — prompt edits re-extract automatically.
  - Fact ids are deterministic (sha1 of content) so cache reloads reproduce identical facts.
  - Changing `EMBEDDING_BACKEND`/`EMBEDDING_MODEL` on an existing store breaks (embedding dim
    mismatch) — delete `data/hydrous.sqlite3` when switching.
  - Verified default embedding path: bge-small-en-v1.5 downloads and embeds (384-d); tests use
    the stateless TfidfEmbedder so they need no network or model download.
  - MockLLM's heuristic handler is intentionally dumb (keyword overlap) — mock e2e answers are
    NOT accurate; that's expected. Accuracy comes from the real model. The mock e2e tests lock
    pipeline plumbing, not quality.
  - Runner is sequential across questions, parallel across sessions within a question
    (extraction and versioning+ingest are bounded-parallel; versioning is applied
    chronologically after extraction). For the full 500-instance run, consider adding
    question-level parallelism.
  - HydraDB adapter: `get_facts`/`get_sessions`/`get_timeline`/`find_candidates` are
    implemented via `context.list` (raw-body wire shape: `user_memories[]`, `memory_id` —
    SDK types are stale). Sessions are `user_assistant_pairs` items split into cost-bounded
    parts (server enforces text XOR pairs, a 1024-byte `additional_metadata` cap, and a
    ~1000 tokens/request ingest budget with a 5000/min rolling limit); `get_sessions`
    reconstructs turns from `context.inspect`. Supersession = `update_source_metadata` PATCH
    on the stable id (verified to sync to the vector store). Conflict detection uses
    `find_candidates` (exact list lane + semantic query lane). Parallel versioning across
    sessions may miss a conflict if the earlier session's facts aren't indexed yet (observed:
    supersessions still fire in practice on the 3-id mock run).

