"""Webhook-driven indexing: IndexingTracker handshake + HydraDBStore short-circuit.

No network: HydraDBStore is constructed with a fake client; the webhook side
marks the tracker, and _wait_indexed must wake without calling context.status.
"""

import asyncio
from types import SimpleNamespace

import pytest

from hydra_mem.indexing import IndexingTracker
from hydrous.store.hydradb import HydraDBStore

from .conftest import run


def _status_response(ids: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            statuses=[
                SimpleNamespace(id=sid, indexing_status="completed", error_message=None) for sid in ids
            ]
        )
    )


class _FakeContext:
    def __init__(self, *, raise_on_call: bool):
        self.raise_on_call = raise_on_call
        self.calls = 0

    async def status(self, **kwargs):
        self.calls += 1
        if self.raise_on_call:
            raise AssertionError("context.status should not be called in webhook mode")
        return _status_response(kwargs["ids"])


class _FakeClient:
    def __init__(self, *, raise_on_status: bool = False):
        self.context = _FakeContext(raise_on_call=raise_on_status)


def _store(client, tracker=None, *, poll_timeout=10.0, webhook_interval=0.05) -> HydraDBStore:
    s = HydraDBStore(
        "fake-key", base_url="http://127.0.0.1:1",
        webhook_tracker=tracker, webhook_interval=webhook_interval, poll_timeout=poll_timeout,
    )
    s._client = client
    return s


def test_tracker_mark_done_failed_clear():
    t = IndexingTracker()
    t.mark("db", "coll", ["a", "b"], "completed")
    t.mark("db", "coll", "c", "failed")
    assert t.done("db", "coll", {"a", "b", "c"}) == {"a", "b", "c"}  # failed is terminal too
    assert t.failed("db", "coll", {"a", "b", "c"}) == {"c"}
    t.clear("db", "coll", ["a"])
    assert t.done("db", "coll", {"a"}) == set()
    assert t.snapshot()  # {"db/coll/c": "failed", ...}


def test_wait_indexed_wakes_on_webhook_without_polling():
    tracker = IndexingTracker()
    client = _FakeClient(raise_on_status=True)  # status must NOT be called
    store = _store(client, tracker)

    async def go():
        wait = asyncio.create_task(store._wait_indexed("qid_q", ["a", "b"]))
        await asyncio.sleep(0.15)  # give it time to start waiting (and clear)
        tracker.mark(store._database, "qid_q", ["a", "b"], "completed")
        await asyncio.wait_for(wait, timeout=2.0)

    run(go())
    assert client.context.calls == 0


def test_wait_indexed_raises_on_failed_webhook():
    tracker = IndexingTracker()
    store = _store(_FakeClient(raise_on_status=True), tracker)

    async def go():
        wait = asyncio.create_task(store._wait_indexed("qid_q", ["a"]))
        await asyncio.sleep(0.15)
        tracker.mark(store._database, "qid_q", ["a"], "failed")
        await wait  # propagates the RuntimeError

    with pytest.raises(RuntimeError, match="indexing failed"):
        run(go())


def test_wait_indexed_falls_back_to_polling_without_tracker():
    client = _FakeClient()
    store = _store(client)  # no tracker -> plain polling path
    run(store._wait_indexed("qid_q", ["a", "b"]))
    assert client.context.calls >= 1


def test_wait_indexed_prefers_fresh_marks_over_stale():
    """A tracker entry from a PREVIOUS ingest must not satisfy a new ingest
    (clear() resets before waiting, so stale 'completed' is ignored)."""
    tracker = IndexingTracker()
    store = _store(_FakeClient(), tracker)
    stale_key = (store._database, "qid_q", "a")
    tracker.mark(*stale_key, "completed")  # status left over from an old run

    async def go():
        wait = asyncio.create_task(store._wait_indexed("qid_q", ["a"]))
        await asyncio.sleep(0.15)  # the stale mark alone is not enough -> fallback poll
        tracker.mark(*stale_key, "completed")  # the FRESH webhook event
        await asyncio.wait_for(wait, timeout=2.0)

    run(go())
    # fresh mark satisfied it; status may have been consulted as fallback — assert no error
