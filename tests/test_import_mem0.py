"""hydrous import-mem0: replay a mem0 get_all export, preserving created_at -> recorded_at."""

import json

import pytest

from hydrous import Memory
from hydrous.embeddings import TfidfEmbedder
from hydrous.importer import load_mem0_export
from hydrous.llm import MockLLM

_CREATED = "2025-01-15T10:30:00Z"
_CREATED2 = "2024-11-02T08:00:00.000Z"


@pytest.fixture
def export(tmp_path):
    payload = {
        "results": [
            {
                "id": "mem_alpha",
                "memory": "Likes hiking on weekends",
                "hash": "h1",
                "metadata": {"category": "hobbies"},
                "score": 0.9,
                "created_at": _CREATED,
                "updated_at": _CREATED,
                "user_id": "alice",
            },
            {
                "id": "mem_beta",
                "memory": "Prefers dark mode and vim keybindings",
                "metadata": {},
                "created_at": _CREATED2,
                "updated_at": _CREATED2,
                "user_id": "alice",
            },
            {
                "id": "mem_gamma",
                "memory": "Plans a trip to Japan in April",
                "created_at": "2025-02-01T12:00:00",
                "updated_at": "2025-02-01T12:00:00",
                # no user_id -> default scope
            },
            {"id": "mem_empty", "memory": "", "created_at": _CREATED},  # skipped
            "garbage",  # skipped (non-dict)
        ]
    }
    p = tmp_path / "mem0_export.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.fixture
def mem(tmp_path) -> Memory:
    m = Memory(
        store="local",
        store_path=str(tmp_path / "import.sqlite3"),
        llm=MockLLM(),
        embedder=TfidfEmbedder(),
    )
    yield m
    m.close()


def test_load_mem0_export_shapes(export, tmp_path):
    items = load_mem0_export(export)
    assert len(items) == 4  # "garbage" dropped; malformed entries filtered after
    assert items[0]["id"] == "mem_alpha"

    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_mem0_export(bad)

    shape = tmp_path / "shape.json"
    shape.write_text('{"nope": 42}', encoding="utf-8")
    with pytest.raises(ValueError, match="results"):
        load_mem0_export(shape)


def test_import_raw_replay_preserves_created_at(export, mem):
    stats = mem.import_mem0(load_mem0_export(export), default_user_id="default")
    assert stats["imported"] == 3  # 4 item rows - 1 empty-memory skip... see below
    # mem_empty has no "memory" -> skipped; garbage skipped -> 3 replayed units

    alice = mem.get_all(filters={"user_id": "alice"})["results"]
    assert len(alice) == 2
    by_mem0 = {i["metadata"]["mem0_id"]: i for i in alice}
    assert by_mem0["mem_alpha"]["memory"] == "Likes hiking on weekends"
    # created_at -> recorded_at: raw replay carries the export timestamp exactly
    assert by_mem0["mem_alpha"]["created_at"].startswith("2025-01-15T10:30:00")
    assert by_mem0["mem_beta"]["created_at"].startswith("2024-11-02T08:00:00")
    assert by_mem0["mem_alpha"]["metadata"]["category"] == "hobbies"

    default_scope = mem.get_all(filters={"user_id": "default"})["results"]
    assert len(default_scope) == 1
    assert default_scope[0]["memory"] == "Plans a trip to Japan in April"  # set to "default" by importer
    assert default_scope[0]["metadata"]["mem0_id"] == "mem_gamma"


def test_import_infer_re_extracts_facts_with_created_at(export, mem):
    stats = mem.import_mem0(load_mem0_export(export), infer=True, default_user_id="default")
    assert stats["imported"] >= 1  # MockLLM extracts one fact per non-empty memory
    alice = mem.get_all(filters={"user_id": "alice"})["results"]
    assert alice and all(i["metadata"]["kind"] == "fact" for i in alice)
    # facts preserve the export created_at as recorded_at
    assert any(i["created_at"].startswith("2025-01-15T10:30:00") for i in alice)


def test_import_is_idempotent(export, mem):
    first = mem.import_mem0(load_mem0_export(export), default_user_id="default")
    assert first["imported"] == 3
    second = mem.import_mem0(load_mem0_export(export), default_user_id="default")
    # feedback-loop: exact re-import of the same raw texts is a no-op, no dupes
    assert second["imported"] == 0
    assert second["noop"] == 3
    assert len(mem.get_all(filters={"user_id": "alice"})["results"]) == 2
