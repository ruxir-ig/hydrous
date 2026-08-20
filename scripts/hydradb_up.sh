#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p .hydradb/store .hydradb/cache
if [[ ! -s .hydradb/auth-token ]]; then
  printf '%s\n' 'local-development-token-32-bytes' > .hydradb/auth-token
fi
export HYDRA_DB_UID="$(id -u)"
export HYDRA_DB_GID="$(id -g)"
docker compose up -d hydradb
ready_url="http://127.0.0.1:9090/readyz"
for attempt in $(seq 1 60); do
  if curl -fsS "$ready_url" >/dev/null 2>&1; then
    echo "HydraDB OSS ready: http://127.0.0.1:8443"
    echo "Readiness: curl -fsS $ready_url"
    exit 0
  fi
  sleep 1
done

echo "HydraDB OSS did not become ready within 60 seconds: $ready_url" >&2
docker compose ps hydradb >&2
exit 1
