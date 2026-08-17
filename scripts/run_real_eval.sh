#!/usr/bin/env bash
# Real-API A/B on the 3-id subset already ingested on the HydraDB free accounts.
#
#   - 6a1eabeb          -> account 2 (sk_live_E5BC...)  [already ingested, no --reingest]
#   - 6d550036, gpt4_59149c77 -> account 3 (sk_live_8KRt...) [already ingested, no --reingest]
#   - local store       -> all 3 ids (SQLite, .env key irrelevant)
#
# Requires: LLM auth working (see RUN_NOTES.md) and account keys exported:
#   HYDRA_ACCT2_KEY / HYDRA_ACCT3_KEY (see local RUN_NOTES.md — never commit keys).
# os.environ beats .env in pydantic-settings, so the exports override the
# HYDRA_DB_API_KEY in .env without editing it.
set -euo pipefail
cd "$(dirname "$0")/.."

ACCT2="${HYDRA_ACCT2_KEY:?export HYDRA_ACCT2_KEY (account 2 sk_live key)}"
ACCT3="${HYDRA_ACCT3_KEY:?export HYDRA_ACCT3_KEY (account 3 sk_live key)}"

echo "== 1/4 hydradb: 6a1eabeb (account 2)"
HYDRA_DB_API_KEY="$ACCT2" .venv/bin/hydrous eval \
    --id 6a1eabeb --out data/hyp_real_hydradb.jsonl --store hydradb

echo "== 2/4 hydradb: 6d550036 + gpt4_59149c77 (account 3, appends)"
HYDRA_DB_API_KEY="$ACCT3" .venv/bin/hydrous eval \
    --id 6d550036 --id gpt4_59149c77 --out data/hyp_real_hydradb.jsonl --store hydradb

echo "== 3/4 local: all 3 ids"
.venv/bin/hydrous eval \
    --id 6a1eabeb --id 6d550036 --id gpt4_59149c77 --out data/hyp_real_local.jsonl --store local

echo "== 4/4 judge + report (both stores)"
.venv/bin/hydrous judge --hypotheses data/hyp_real_hydradb.jsonl --dataset data/longmemeval_s_cleaned.json
.venv/bin/hydrous judge --hypotheses data/hyp_real_local.jsonl --dataset data/longmemeval_s_cleaned.json
.venv/bin/hydrous report --log data/hyp_real_hydradb.jsonl.log
.venv/bin/hydrous report --log data/hyp_real_local.jsonl.log

echo "done"
