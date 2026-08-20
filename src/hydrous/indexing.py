"""Webhook-driven indexing for the HydraDB adapter.

HydraDB fires `indexing.status_changed` webhook events to a registered public
callback URL. When one is configured (hydrous serve --callback-url ...), the
server's `/webhooks/hydradb/indexing` endpoint marks sources in this tracker and
`HydraDBStore._wait_indexed` wakes within ~0.5s instead of a 2s polling loop —
one `context.status` confirmation (or none if every source is marked) replaces
the poll. Plain polling is kept as the fallback: no tracker, or tracker timeout.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_TERMINAL = ("completed", "failed")


class IndexingTracker:
    """Thread-safe {(database, collection, source_id): indexing_status} store.

    The webhook endpoint (server thread) marks, the store's async waiters read —
    the lock makes the handshake safe across threads/loops.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: dict[tuple[str, str, str], str] = {}

    # ---- server side (webhook endpoint) ---------------------------------

    def mark(self, database: str, collection: str, ids: list[str] | str, status: str) -> None:
        """Record indexing_status for source ids (webhook payload)."""
        if isinstance(ids, str):
            ids = [ids]
        status = str(status or "").lower()
        with self._lock:
            for sid in ids:
                if sid:
                    self._status[(database, collection, str(sid))] = status
        log.info("indexing tracker: %s/%s %s -> %s", database, collection, list(ids), status)

    # ---- store side (_wait_indexed) -------------------------------------

    def clear(self, database: str, collection: str, ids: list[str]) -> None:
        """Forget prior statuses for these ids (a re-ingest starts fresh)."""
        with self._lock:
            for sid in ids:
                self._status.pop((database, collection, sid), None)

    def done(self, database: str, collection: str, ids: set[str]) -> set[str]:
        """Subset of ids already reported terminal (completed/failed)."""
        with self._lock:
            return {sid for sid in ids if self._status.get((database, collection, sid)) in _TERMINAL}

    def failed(self, database: str, collection: str, ids: set[str]) -> set[str]:
        with self._lock:
            return {sid for sid in ids if self._status.get((database, collection, sid)) == "failed"}

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {"/".join(k): v for k, v in self._status.items()}
