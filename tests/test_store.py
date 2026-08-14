from hydra_mem.schema import Fact, Session, Turn
from hydra_mem.store.local import LocalMemoryStore

from .conftest import run


def _fact(id, value, valid_from, valid_to=None, predicate="favorite_color") -> Fact:
    return Fact(
        id=id,
        question_id="q1",
        subject="user",
        predicate=predicate,
        value=value,
        fact_date=valid_from,
        session_id="s1",
        valid_from=valid_from,
        valid_to=valid_to,
    )


def _session(sid="s1", date="2023/05/01 (Mon) 10:00", content="My favorite color is blue.") -> Session:
    return Session(
        session_id=sid,
        date=date,
        turns=[Turn(role="user", content=content), Turn(role="assistant", content="ok")],
    )


def test_roundtrip_and_persistence(tmp_path, embedder):
    path = tmp_path / "persist.sqlite3"
    store = LocalMemoryStore(path, embedder)
    run(store.add_session("q1", _session()))
    run(store.add_facts("q1", [_fact("f1", "blue", "2023-05-01")]))
    store.close()

    reopened = LocalMemoryStore(path, embedder)
    facts = run(reopened.get_facts("q1"))
    sessions = run(reopened.get_sessions("q1"))
    assert [f.value for f in facts] == ["blue"]
    assert sessions[0].turns[0].content == "My favorite color is blue."
    reopened.close()


def test_search_ranks_relevant_fact_first(store):
    run(store.add_facts("q1", [
        _fact("f1", "blue", "2023-05-01"),
        _fact("f2", "sourdough bread", "2023-05-02", predicate="favorite_food"),
    ]))
    results = run(store.search("q1", "favorite color", as_of_date="2023-06-01", include_sessions=False))
    assert results[0].fact_id == "f1"
    assert results[0].source == "fact"


def test_search_as_of_and_superseded(store):
    run(store.add_facts("q1", [
        _fact("f1", "blue", "2023-05-01", valid_to="2023-06-01"),
        _fact("f2", "green", "2023-06-01"),
    ]))
    current = run(store.search("q1", "color", as_of_date="2023-06-20", include_sessions=False))
    assert {e.fact_id for e in current} == {"f2"}

    past = run(store.search("q1", "color", as_of_date="2023-05-10", include_sessions=False))
    assert {e.fact_id for e in past} == {"f1"}

    everything = run(
        store.search("q1", "color", as_of_date="2023-06-20", include_sessions=False, include_superseded=True)
    )
    assert {e.fact_id for e in everything} == {"f1", "f2"}


def test_search_includes_sessions(store):
    run(store.add_session("q1", _session()))
    results = run(store.search("q1", "favorite color", as_of_date="2023-06-01", include_facts=False))
    assert len(results) == 1
    assert results[0].source == "session"
    assert results[0].session_id == "s1"


def test_fact_augmented_session_indexing(store):
    # a fact's text is indexed together with its session -> session findable by fact terms
    session = _session(content="we chatted about weekend plans")
    run(store.add_session("q1", session, facts=[_fact("f1", "blue", "2023-05-01")]))
    results = run(store.search("q1", "favorite_color blue", as_of_date="2023-06-01", include_facts=False))
    assert results and results[0].session_id == "s1"


def test_has_and_delete_question(store):
    assert not run(store.has_question("q1"))
    run(store.add_session("q1", _session()))
    run(store.add_facts("q1", [_fact("f1", "blue", "2023-05-01")]))
    assert run(store.has_question("q1"))
    run(store.delete_question("q1"))
    assert not run(store.has_question("q1"))
    assert run(store.get_facts("q1")) == []


def test_empty_store_search(store):
    assert run(store.search("q1", "anything")) == []


def test_questions_are_isolated(store):
    run(store.add_facts("q1", [_fact("f1", "blue", "2023-05-01")]))
    assert run(store.search("q2", "color")) == []
    assert run(store.get_facts("q2")) == []
