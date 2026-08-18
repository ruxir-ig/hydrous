# LongMemEval comparison table

Methodology is the hard part of a benchmark table — different harnesses, judge
models, retrieval budgets, and abstention rules make raw numbers misleading. This
page states the protocol next to every number.

## The table

| System | LongMemEval score | Notes |
| --- | --- | --- |
| GPT-4o, full history (baseline) | **60.6%** | Full-context "read everything" baseline, LongMemEval's published number. |
| Oracle evidence | **87.0%** | Gold evidence given to the reader — the ceiling for retrieval. |
| mem0 (managed platform, new algorithm, Apr 2026) | **94.4** | mem0's published benchmark; their highest subscore is 98.2 assistant-memory recall. **Not 1:1 comparable** — see protocol notes. |
| **Hydrous (this repo)** | *full 500-instance run pending* | Pilot subsets: local **66.7%** (3 ids), local **60%** (5 ids), HydraDB cloud **33.3%** (3 ids, same judge). Full run was started and paused at 1/500 — the LLM endpoint degraded (504s) — see [RUN_NOTES](../RUN_NOTES.md) and the [roadmap](../ARCHITECTURE.md). |

## Protocol notes (read before comparing)

- **Judging**: Hydrous uses LongMemEval's official judge prompts, judged by the
  same model that answered ("self-judge"), restarting on disposable judge runs.
  Abstention (`_abs` ids) is scored by LongMemEval's rules.
- **Retrieval budget**: single-pass retrieval, `TOP_K=20` hybrid (dense + BM25,
  lane-separated RRF), one query-analysis round + reader. No agentic loops.
- **mem0's number** comes from [mem0's own benchmark
  table](https://github.com/mem0ai/mem0) and carries their caveat verbatim:
  *"Scores reflect Mem0's managed platform, which includes proprietary
  optimizations not available in the open-source SDK; open-source users should
  expect directionally similar gains but not identical numbers."* Their protocol
  also uses a top_200 budget and their own evaluation framework
  ([memory-benchmarks](https://github.com/mem0ai/memory-benchmarks)); a
  head-to-head under one harness is out of scope (deliberately later).
- **What mem0's new algorithm confirms**: their April-2026 rewrite is "single-pass
  ADD-only extraction — memories accumulate; nothing is overwritten", plus
  entity linking and temporal reasoning. That is the append-only bi-temporal
  design Hydrous shipped in Phase 1/2 (supersession closes `valid_to`; nothing is
  ever silently deleted; `history()` exposes the chain).

## Re-running it

```bash
# full 500-instance run (paused pending endpoint health):
.venv/bin/hydrous eval --dataset data/longmemeval_s_cleaned.json \
    --out data/hyp_phase3_500.jsonl --reingest --store local
.venv/bin/hydrous judge --hypotheses data/hyp_phase3_500.jsonl \
    --dataset data/longmemeval_s_cleaned.json
.venv/bin/hydrous report --log data/hyp_phase3_500.jsonl.log
```

The hypotheses file is resumable; the run picks up where it stopped.
