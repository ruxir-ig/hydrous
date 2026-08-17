import importlib.util

import pytest

from hydrous.schema import Fact, Session, Turn
from hydrous.store.hydradb import (
    HydraDBStore,
    _fact_valid_as_of,
    _parse_content_turns,
    fact_to_memory_item,
    session_to_memory_item,
    session_to_memory_items,
)


def _fact() -> Fact:
    return Fact(
        id="f1", question_id="q1", subject="user", predicate="lives_in", value="Berlin",
        fact_date="2023-05-01", session_id="s1", valid_from="2023-05-01",
        valid_to="2023-06-15", superseded_by="f2",
    )


def test_fact_item_mapping():
    item = fact_to_memory_item(_fact())
    assert item["id"] == "f1"
    assert item["text"] == "user lives_in: Berlin"
    assert "user_assistant_pairs" not in item
    assert item["metadata"] == {"question_id": "q1", "kind": "fact"}
    assert item["additional_metadata"]["valid_from"] == "2023-05-01"
    assert item["additional_metadata"]["valid_to"] == "2023-06-15"
    assert item["additional_metadata"]["superseded_by"] == "f2"


def test_session_item_mapping_pairs():
    session = Session(
        session_id="s1",
        date="2023/05/01 (Mon) 10:00",
        turns=[
            Turn(role="user", content="hi"),
            Turn(role="assistant", content="hello"),
            Turn(role="user", content="bye"),  # trailing user turn without assistant reply
        ],
    )
    items = session_to_memory_items("q1", session)
    assert len(items) == 1  # short pairs fit under the ingest budget -> single part
    item = items[0]
    assert item["id"] == "q1:s1"
    assert item["user_assistant_pairs"] == [{"user": "hi", "assistant": "hello"}]
    assert "text" not in item  # server requires text XOR user_assistant_pairs
    assert "turns" not in item["additional_metadata"]  # 1024-byte cap on additional_metadata
    assert item["additional_metadata"]["session_id"] == "s1"
    assert item["additional_metadata"]["part"] == 0
    # inspect derives the pair text back from pairs (get_sessions path)
    assert _parse_content_turns(
        "USER: hi\nASSISTANT: hello\nUSER: bye"
    ) == [Turn(role="user", content="hi"), Turn(role="assistant", content="hello"), Turn(role="user", content="bye")]


def test_session_item_mapping_splits_oversized_sessions():
    long = "x" * 5000  # ~714 cost units per pair -> a single pair exceeds the 600 budget
    session = Session(
        session_id="s3",
        date="2023/05/01 (Mon) 10:00",
        turns=[
            Turn(role="user", content=long),
            Turn(role="assistant", content=long),
            Turn(role="user", content=long),
            Turn(role="assistant", content=long),
            Turn(role="user", content=long),
            Turn(role="assistant", content=long),
        ],
    )
    items = session_to_memory_items("q1", session)
    assert len(items) > 1
    ids = [i["id"] for i in items]
    assert ids[0] == "q1:s3"
    assert ids[1] == "q1:s3#1"
    parts = [i["additional_metadata"]["part"] for i in items]
    assert parts == sorted(parts)
    for i in items:
        assert i["additional_metadata"]["session_id"] == "s3"
        assert "text" not in i
        total_pairs = sum(len(i["user_assistant_pairs"]) for i in items)
        assert total_pairs == 3  # trailing user turn dropped; 3 complete pairs total
        # pairs are capped so a single part always fits one ingest request
        for p in i["user_assistant_pairs"]:
            assert len(p["user"]) <= 4000 and len(p["assistant"]) <= 4000
    assert session_to_memory_item("q1", session)["id"] == "q1:s3"  # alias = first part


def test_session_item_mapping_text_fallback():
    session = Session(
        session_id="s2",
        date="2023/05/01 (Mon) 10:00",
        turns=[Turn(role="user", content="only user messages here")],
    )
    items = session_to_memory_items("q1", session)
    assert len(items) == 1
    item = items[0]
    assert "user_assistant_pairs" not in item
    assert item["text"] == session.text


def test_fact_valid_as_of():
    am = {"valid_from": "2023-05-01", "valid_to": "2023-06-15"}
    assert _fact_valid_as_of(am, None) is False  # superseded = not currently valid
    assert _fact_valid_as_of(am, "2023-06-01") is True
    assert _fact_valid_as_of(am, "2023-06-15") is False  # [valid_from, valid_to)
    assert _fact_valid_as_of(am, "2023-04-30") is False
    assert _fact_valid_as_of({"valid_from": "2023-05-01"}, None) is True  # open interval


def test_parse_content_turns():
    # single pair: "ROLE: content" lines
    turns = _parse_content_turns("USER: hi there\nASSISTANT: hello\nand a follow-up line\nUSER: bye")
    assert [t.role for t in turns] == ["user", "assistant", "user"]
    assert turns[0].content == "hi there"
    assert turns[1].content == "hello\nand a follow-up line"
    assert turns[2].content == "bye"
    # multi pair: server returns the {"pairs": [...]} JSON envelope
    envelope = '{"pairs":[{"user":"what\'s the weather?","assistant":"sunny"},{"user":"and tomorrow?","assistant":"rain"}],"user_name":"User"}'
    turns = _parse_content_turns(envelope)
    assert [(t.role, t.content) for t in turns] == [
        ("user", "what's the weather?"), ("assistant", "sunny"),
        ("user", "and tomorrow?"), ("assistant", "rain"),
    ]


def test_store_requires_sdk_with_clear_error():
    if importlib.util.find_spec("hydra_db") is not None:
        pytest.skip("hydradb-sdk is installed")
    with pytest.raises(RuntimeError, match="hydradb-sdk"):
        HydraDBStore("fake-key")
