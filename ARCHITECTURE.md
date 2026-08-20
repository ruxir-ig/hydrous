# Hydrous architecture

Agent memory for cross-session continuity. Bi-temporal facts, hybrid retrieval,
safe abstention, and an optional local HydraDB memory graph.

## Problem

Answer questions over chat histories spanning many sessions (LongMemEval-S scale:
~30–40 sessions / ~115K tokens per question). Requirements:

- Synthesize facts across sessions with chronological reasoning.
- Track knowledge updates (facts that were later overwritten).
- Abstain when the answer is not in the history.
- Reference baselines: GPT-4o full-history 60.6%; oracle evidence 87.0%.

## Design goals

1. End-to-end pipeline first: ingest → retrieve → answer → eval JSONL.
2. Provider-agnostic LLM client (OpenAI / OpenRouter / custom / `MockLLM`).
3. Pluggable storage: local SQLite hybrid index, HydraDB OSS graph projection,
   optional cloud adapter.
4. Bi-temporal fact versioning: never silently delete; close validity intervals.
5. Safe abstention: exact string when evidence is insufficient.
6. Cheap iteration: extraction cache, resumable eval, `--limit` / `--type` / `--ids`.

## System diagram

```
            LongMemEval JSON (data/)
                    │
            ingest/longmemeval.py
                    │
                    ▼
            ingest/extractor.py     raw turns + LLM facts + supersession + summary
                    │
                    ▼
            MemoryStore             local.py | graph/* | hydradb.py
                    │
                    ▼
            retrieve/pipeline.py    query analysis → hybrid search → as-of filter
                    │               → EvidencePack
                    ▼
            answer/reader.py        sufficiency gate → CoN reader → answer | ABSTAIN
                    │
                    ▼
            eval/runner.py → judge.py → metrics.py
```

## Module map (`src/hydrous/`)

| Module | Responsibility |
|---|---|
| `config.py` | Env-driven `Settings`, `build_llm`, `build_embedder` |
| `schema.py` | `Turn`, `Session`, `Fact`, `Evidence`, `EvidencePack`, `ABSTAIN_STRING` |
| `text_utils.py` | Tokenization, date parsing, overlap helpers |
| `prompts.py` | All prompt templates (`TASK:` markers for `MockLLM`) |
| `llm.py` | `LLMClient`, `OpenAICompatibleLLM`, `MockLLM` |
| `embeddings.py` | Sentence-transformers or TF-IDF fallback |
| `ingest/longmemeval.py` | Streaming dataset loader |
| `ingest/extractor.py` | Fact extraction, quality gate, supersession, disk cache |
| `store/base.py` | `MemoryStore` ABC |
| `store/local.py` | SQLite + dense + BM25 (RRF) |
| `store/hydradb.py` | Optional cloud adapter |
| `graph/base.py` | `MemoryGraph` named write/query seam |
| `graph/hydradb_oss.py` | Local HydraDB OSS HTTP/OpenCypher adapter |
| `graph/projector.py` | Keeps SQLite retrieval and OSS graph in sync |
| `graph/queries.py` | Named Cypher (as-of, supersession, multi-hop) |
| `retrieve/pipeline.py` | Hybrid retrieval → chronological `EvidencePack` |
| `answer/reader.py` | Sufficiency + reader → answer or abstention |
| `eval/*` | Runner, official judge prompts, metrics |
| `mem0.py` | Public `Memory` / `AsyncMemory` API |
| `importer.py` | Memory export loader |
| `server.py` | HTTP bridge + indexing webhook |
| `indexing.py` | Webhook-driven indexing tracker |
| `mcp.py` | stdio MCP server |
| `cli.py` | Typer CLI |

## Data schemas

```python
Turn(role: str, content: str)

Session(session_id, date, turns, summary=None,
        recorded_at=None,   # transaction time
        metadata={})

Fact(                       # bi-temporal, append-only on supersession
    id, question_id, subject, predicate, value,
    fact_date,              # when true in the world
    session_id,             # provenance
    valid_from,             # = fact_date
    valid_to: None,         # None = currently valid
    recorded_at,            # ingestion wall-clock
    superseded_by: None,
    quality_score: None,    # extraction confidence 0..1
)
Fact.text == f"{subject} {predicate}: {value}"
Fact.valid_as_of(date) == valid_from <= date < valid_to (or valid_to is None)

Evidence(source: "fact"|"session", text, date, session_id, score, fact_id=None)
EvidencePack(question, question_date, items)

ABSTAIN_STRING = "I don't know. The chat history does not contain this information."
```

## Algorithms

### Ingestion

1. Sessions sorted chronologically.
2. Extraction runs concurrently per session (semaphore = `MAX_CONCURRENCY`).
   Quality gate drops low-confidence / rejected facts. Results cache under
   `cache/extractions/` and invalidate on model or prompt-version change.
3. Versioning applies sessions in date order: store session → conflict-check
   currently valid facts → on conflict, set `old.valid_to` and `superseded_by`
   (never silent delete) → store new facts.

### Retrieval

1. Query analysis rewrites the question and resolves relative time.
2. Hybrid search: dense cosine + BM25, merged with RRF (k=60), over facts and sessions.
3. As-of filter: keep facts valid at `question_date`. Questions about previous
   state also include superseded facts. Sessions always searchable.
4. Dedupe, top-k, chronological order for the reader.

### Answering

1. Empty evidence pack → abstain (no LLM call).
2. Optional sufficiency gate (`USE_GATE`). Default recommendation: rely on the
   reader's own abstention instruction; the LLM gate has been unreliable on
   deepseek-v4-flash in local tests.
3. Chain-of-Note reader with `question_date`. Output containing "don't know" /
   "does not contain" normalizes to `ABSTAIN_STRING`.

### Eval

Resumable hypotheses JSONL, official LongMemEval judge prompts, overall +
per-`question_type` + abstention-subset accuracy.

## LLM configuration

From `.env` (see `.env.example`):

- `LLM_PROVIDER=openai|openrouter|custom|mock`
- `LLM_MODEL`, `JUDGE_MODEL`
- Provider keys / `LLM_BASE_URL` as needed

`MockLLM` is deterministic and key-free: scripted FIFO in tests, or heuristic
handlers keyed on each prompt's `TASK:` marker.

## Storage

### Local (default)

SQLite at `STORE_PATH` (default `data/hydrous.sqlite3`). Hybrid merge is
lane-separated RRF (facts vs sessions; facts weight 0.4). Embeddings:
`sentence-transformers` (default `BAAI/bge-small-en-v1.5`) or `tfidf` fallback.

### HydraDB OSS (`hydradb-oss`)

Composes `LocalMemoryStore` for text retrieval with `HydraDBOSSGraph` for the
explicit graph. Graph is source of truth for Entity, Fact, Session, PREDICATE,
FROM_SESSION, NEXT_SESSION, SUPERSEDED_BY. Named ops: `as_of`,
`supersession_chain`, `multi_hop`.

```bash
./scripts/hydradb_up.sh
uv run hydrous demo --mock --store hydradb-oss
```

Data under `.hydradb/store`; token in `.hydradb/auth-token` unless
`HYDRA_DB_OSS_TOKEN` is set. No cloud key required.

### HydraDB cloud (optional `[hydra]` extra)

`HydraDBStore` implements the same `MemoryStore` interface via `hydradb-sdk`.
Integration tests run only when `HYDRA_DB_API_KEY` is set.

## How to run

```bash
uv sync
cp .env.example .env

uv run hydrous eval --dataset data/longmemeval_oracle.json --limit 5 --mock \
    --out hyp_mock.jsonl
uv run hydrous judge --hypotheses hyp_mock.jsonl \
    --dataset data/longmemeval_oracle.json --mock
uv run hydrous report --log hyp_mock.jsonl.log

uv run hydrous eval --limit 10 --type knowledge-update --out hyp.jsonl
uv run pytest
uv run ruff check
```

## Public API notes

- Scopes map to store buckets (`user_id` / `agent_id` / `run_id`).
- Search results are content-hash tagged; re-add is `NOOP`.
- Fact ids are deterministic sha1 of content (exact re-extraction dedupes).
- Scores from RRF are rank-mapped to [0.1, 1.0] for the public `search` surface;
  `explain=True` exposes raw RRF in `score_details`.
- Changing embedding backend/model on an existing store requires deleting the
  SQLite file (dimension mismatch).
