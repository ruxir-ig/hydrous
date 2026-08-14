from hydra_mem.llm import MockLLM, default_mock_handler
from hydra_mem.retrieve.pipeline import RetrievalPipeline, parse_json_object
from hydra_mem.schema import Fact, Session, Turn

from .conftest import run


def _fact(id, value, valid_from, valid_to=None) -> Fact:
    return Fact(
        id=id, question_id="q1", subject="user", predicate="favorite_color", value=value,
        fact_date=valid_from, session_id="s1", valid_from=valid_from, valid_to=valid_to,
    )


def _seed(store):
    run(store.add_facts("q1", [
        _fact("f_old", "blue", "2023-05-01", valid_to="2023-06-01"),
        _fact("f_new", "green", "2023-06-01"),
    ]))
    run(store.add_session("q1", Session(
        session_id="s1", date="2023/05/01 (Mon) 10:00",
        turns=[Turn(role="user", content="My favorite color is blue.")],
    )))


def test_parse_json_object():
    assert parse_json_object('prefix {"a": 1} suffix') == {"a": 1}
    assert parse_json_object("junk") == {}


def test_retrieve_chronological_and_as_of(store, mock_llm):
    _seed(store)
    pipeline = RetrievalPipeline(mock_llm, store, top_k=10)
    pack = run(pipeline.retrieve("q1", "What is my favorite color?", "2023/06/20 (Tue) 12:00"))
    assert pack.items
    # superseded fact excluded by as-of filter
    assert all(ev.fact_id != "f_old" for ev in pack.items)
    assert any(ev.fact_id == "f_new" for ev in pack.items)
    # chronological order (dates non-decreasing)
    from hydra_mem.text_utils import date_key

    dates = [date_key(ev.date) for ev in pack.items]
    assert dates == sorted(dates)


def test_previous_intent_includes_superseded(store, mock_llm):
    _seed(store)
    pipeline = RetrievalPipeline(mock_llm, store, top_k=10)
    pack = run(pipeline.retrieve("q1", "What did I previously say my favorite color was?", "2023/06/20 (Tue) 12:00"))
    assert any(ev.fact_id == "f_old" for ev in pack.items)


def test_query_analysis_failure_falls_back(store):
    _seed(store)

    def garbage_handler(system, user):
        if "analyze_query" in system:
            return "not json at all"
        return default_mock_handler(system, user)

    pipeline = RetrievalPipeline(MockLLM(handler=garbage_handler), store, top_k=10)
    pack = run(pipeline.retrieve("q1", "favorite color?", "2023/06/20 (Tue) 12:00"))
    assert pack.items  # identity rewrite still retrieves


def test_query_analysis_subqueries_merged(store):
    _seed(store)

    def handler(system, user):
        if "analyze_query" in system:
            return '{"rewritten_query": "color preference", "subqueries": ["green color"], "time_range": null, "question_type_hint": null}'
        return default_mock_handler(system, user)

    pipeline = RetrievalPipeline(MockLLM(handler=handler), store, top_k=10)
    pack = run(pipeline.retrieve("q1", "color preference?", "2023/06/20 (Tue) 12:00"))
    assert any(ev.fact_id == "f_new" for ev in pack.items)
