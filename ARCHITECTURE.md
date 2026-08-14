# HydraMem — Architecture (THE single living doc)

Agent memory layer for cross-session continuity. Hack with Hydra hackathon, deadline 2026-08-16.
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

## Module map (`src/hydra_mem/`)

| Module | Responsibility |
|---|---|
| `config.py` | pydantic-settings, env-driven (`Settings`), provider presets, `build_llm`, `build_embedder` helpers |
| `schema.py` | `Turn`, `Session`, `Fact` (bi-temporal), `Evidence`, `EvidencePack`, `MemoryItem`, `EvalInstance`, `ABSTAIN_STRING` |
| `text_utils.py` | tokenization, tolerant date parsing/normalization, word-overlap helpers |
| `prompts.py` | ALL prompt templates (extraction, summary, query analysis, conflict, sufficiency, reader). Each starts with a stable `TASK:` marker line that `MockLLM` dispatches on. |
| `llm.py` | `LLMClient` protocol, `OpenAICompatibleLLM` (async, retry/backoff), `MockLLM` (scripted FIFO + deterministic heuristic handler) |
| `embeddings.py` | `Embedder` protocol, `SentenceTransformerEmbedder` (lazy), `TfidfEmbedder` (pure-numpy hashing, stateless, default fallback) |
| `ingest/longmemeval.py` | streaming JSON-array loader (`iter_instances`), `sessions_from_instance` (strips `has_answer`, sorts by date) |
| `ingest/extractor.py` | `FactExtractor`: LLM fact extraction + summary per session, disk cache, supersession resolution |
| `store/base.py` | `MemoryStore` ABC |
| `store/local.py` | `LocalMemoryStore`: SQLite + dense embeddings + BM25 hybrid (RRF), temporal filtering |
| `store/hydradb.py` | `HydraDBStore`: cloud adapter, same interface (import-guarded) |
| `retrieve/pipeline.py` | `RetrievalPipeline`: query analysis → hybrid search → as-of filter → chronological `EvidencePack` |
| `answer/reader.py` | `Answerer`: sufficiency gate → CoN reader → answer or exact abstention string |
| `eval/runner.py` | `EvalRunner`: iterate instances → ingest (cached) → answer → hypotheses JSONL; resumable |
| `eval/judge.py` | official judge prompts verbatim, configurable judge model, `.log` output |
| `eval/metrics.py` | overall + per-`question_type` + abstention-subset accuracy table |
| `cli.py` | typer CLI: `download-data` / `ingest` / `answer` / `eval` / `judge` / `report` |

## Data schemas

```python
Turn(role: str, content: str)                      # has_answer NEVER stored
Session(session_id, date, turns, summary=None)     # date as dataset string; normalized on compare

Fact(                                               # bi-temporal, append-only
    id, question_id, subject, predicate, value,
    fact_date,            # when the fact became true (LLM-extracted, else session date)
    session_id,           # provenance
    valid_from,           # start of validity interval (= fact_date)
    valid_to: None,       # None = currently valid; set when superseded
    recorded_at,          # wall-clock ingestion time (system time)
    superseded_by: None,  # id of the fact that replaced this one
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
   `{subject, predicate, value, fact_date}` + a 2–3 sentence summary. Results cached at
   `cache/extractions/{qid}__{sid}.json` (invalidated on model change) — reruns are free.
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
2. **Sufficiency gate** (CRAG-style, strict): LLM judges whether the evidence explicitly
   contains the asked information. "no" → return the exact abstention string.
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

SQLite at `STORE_PATH` (default `data/hydra_mem.sqlite3`). Tables: `sessions`, `turns`,
`facts`, `meta`. Embeddings stored as float32 blobs; per-question corpus is small (~40 sessions,
≤ a few hundred facts) so search loads that question's rows, computes cosine in numpy, and
builds BM25 on the fly. Hybrid merge = RRF.

Embedding backends (`EMBEDDING_BACKEND`):
- `sentence-transformers` (default when importable): `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`.
- `tfidf`: pure-numpy stateless hashing TF-IDF (4096-d, sublinear tf, L2-normalized). Used by
  tests (no network/model download) and as automatic fallback.

### HydraDB (optional extra `[hydra]`)

`HydraDBStore` implements the same `MemoryStore` interface. One collection per LongMemEval
instance (`qid_{question_id}`). Facts ingested with `id` upsert key; validity interval in
`additional_metadata` (`valid_from`/`valid_to`/`superseded_by`); supersession = re-ingest same
id with updated metadata (never delete). Query: `query_by="hybrid"`, `mode="thinking"`,
`recency_bias=0.4`, `additional_context=f"Question date: {question_date}"`. Ingestion is async —
adapter polls `databases.status` / `context.status` until ready/completed. Integration tests run
only when `HYDRA_DB_API_KEY` is set.

## How to run

```bash
uv sync                              # create venv + install deps
cp .env.example .env                 # set LLM_PROVIDER + key + model
uv run hydra-mem download-data       # fetch LongMemEval oracle + s_cleaned into data/

# deterministic smoke run, NO api key needed:
uv run hydra-mem eval --dataset data/longmemeval_oracle.json --limit 5 --mock --out hyp_mock.jsonl
uv run hydra-mem judge --hypotheses hyp_mock.jsonl --dataset data/longmemeval_oracle.json --mock
uv run hydra-mem report --log hyp_mock.jsonl.log

# real run (subset first!):
uv run hydra-mem eval --limit 10 --type knowledge-update --out hyp.jsonl
uv run hydra-mem judge --hypotheses hyp.jsonl --dataset data/longmemeval_s_cleaned.json
uv run hydra-mem report --log hyp.jsonl.log

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
- [x] `ruff check` clean; `pytest` 72 passed + 1 skipped
- [ ] Real-API subset run + prompt tuning (`--limit 10 --type knowledge-update` first)
- [ ] Full 500-instance run + score vs 0.606 baseline / 0.870 oracle
- [ ] Optional: HydraDB integration run once a HYDRA_DB_API_KEY is available

## Handoff notes for agents

(Append decisions + gotchas here as they happen; newest first.)

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
  - Extraction cache (`cache/extractions/{qid}__{sid}.json`) is invalidated on MODEL change only.
    If you edit the extraction PROMPT, delete `cache/` manually to re-extract.
  - Fact ids are deterministic (sha1 of content) so cache reloads reproduce identical facts.
  - Changing `EMBEDDING_BACKEND`/`EMBEDDING_MODEL` on an existing store breaks (embedding dim
    mismatch) — delete `data/hydra_mem.sqlite3` when switching.
  - Verified default embedding path: bge-small-en-v1.5 downloads and embeds (384-d); tests use
    the stateless TfidfEmbedder so they need no network or model download.
  - MockLLM's heuristic handler is intentionally dumb (keyword overlap) — mock e2e answers are
    NOT accurate; that's expected. Accuracy comes from the real model. The mock e2e tests lock
    pipeline plumbing, not quality.
  - Runner is sequential across questions, parallel across sessions within a question
    (extraction is order-independent; versioning is applied chronologically afterwards).
    For the full 500-instance run, consider adding question-level parallelism.
  - HydraDB adapter: `get_facts`/`get_sessions`/`get_timeline`/`find_candidates` have no native
    cloud API and return [] (logged); supersession is applied via id-upserts; conflict detection
    is delegated to HydraDB's `infer` pipeline. Revisit if the SDK grows list/filter APIs.

