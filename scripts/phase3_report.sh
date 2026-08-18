#!/usr/bin/env bash
# Phase 3 close-out: judge the 500-instance hypotheses + print the comparison table
# rows. Idempotent — re-running appends only missing judges.
set -euo pipefail
cd "$(dirname "$0")/.."

HYP="${1:-data/hyp_phase3_500.jsonl}"
DATASET="${2:-data/longmemeval_s_cleaned.json}"
LOG="$HYP.log"

if [[ ! -s "$HYP" ]]; then
  echo "no hypotheses at $HYP — run the eval first (see docs/comparison.md)" >&2
  exit 1
fi

n=$(wc -l < "$HYP")
echo "== judging $n hypothesis(es) -> $LOG"
MAX_CONCURRENCY=6 .venv/bin/hydrous judge --hypotheses "$HYP" --dataset "$DATASET"
echo "== metrics"
.venv/bin/hydrous report --log "$LOG"
echo
echo "== comparison rows (Hydrous column)"
.venv/bin/python - "$LOG" <<'PY'
import json, sys
from hydrous.eval.metrics import compute_metrics
log = sys.argv[1]
m = compute_metrics(log)
print(f"| Hydrous | {m.get('overall')} | n={m.get('n')} judged by the same model; "
      f"abstention subset {m.get('abstention', {})}")
PY
