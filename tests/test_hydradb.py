import importlib.util
import json
import os

import pytest

from hydra_mem.schema import Fact, Session, Turn
from hydra_mem.store.hydradb import HydraDBStore, fact_to_memory_item, session_to_memory_item

from .conftest import run


def _fact() -> Fact:
    return Fact(
        id="f1", question_id="q1", subject="user", predicate="lives_in", value="Berlin",
        fact_date="2023-05-01", session_id="s1", valid_from="2023-05-01",
        valid_to="2023-06-15", superseded_by="f2",
    )


def test_fact_item_mapping():
    item = fact_to_memory_item(_fact())
    assert item.id == "f1"
    assert item.text == "user lives_in: Berlin"
    assert item.user_assistant_pairs is None
    meta = json.loads(item.metadata)
    assert meta == {"question_id": "q1", "kind": "fact"}
    assert item.additional_metadata["valid_from"] == "2023-05-01"
    assert item.additional_metadata["valid_to"] == "2023-06-15"
    assert item.additional_metadata["superseded_by"] == "f2"
    dumped = item.model_dump(exclude_none=True)
    assert "user_assistant_pairs" not in dumped  # text XOR pairs


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
    item = session_to_memory_item("q1", session)
    assert item.id == "q1:s1"
    assert item.user_assistant_pairs == [{"user": "hi", "assistant": "hello"}]
    assert item.text is None
    assert item.additional_metadata["session_id"] == "s1"


def test_session_item_mapping_text_fallback():
    session = Session(
        session_id="s2",
        date="2023/05/01 (Mon) 10:00",
        turns=[Turn(role="user", content="only user messages here")],
    )
    item = session_to_memory_item("q1", session)
    assert item.user_assistant_pairs is None
    assert item.text == session.text


def test_store_requires_sdk_with_clear_error():
    if importlib.util.find_spec("hydra_db") is not None:
        pytest.skip("hydradb-sdk is installed")
    with pytest.raises(RuntimeError, match="hydradb-sdk"):
        HydraDBStore("fake-key")


@pytest.mark.skipif(not os.environ.get("HYDRA_DB_API_KEY"), reason="HYDRA_DB_API_KEY not set")
def test_hydradb_integration_roundtrip():
    store = HydraDBStore(os.environ["HYDRA_DB_API_KEY"], database="hydra_mem_test")
    run(store.add_facts("qtest", [_fact()]))
    results = run(store.search("qtest", "where does the user live", as_of_date="2023-06-01"))
    assert results
