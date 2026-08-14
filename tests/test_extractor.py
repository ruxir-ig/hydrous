import json
import re

from hydrous.ingest.extractor import FactExtractor, parse_json_array
from hydrous.llm import MockLLM, default_mock_handler

from .conftest import run

FACTS = {
    "s1": [{"subject": "user", "predicate": "lives_in", "value": "Berlin", "fact_date": "2023-05-01"}],
    "s2": [{"subject": "user", "predicate": "lives_in", "value": "Munich", "fact_date": "2023-06-15"}],
}


def _handler(system: str, user: str) -> str:
    task = system.split("\n")[0]
    if "extract_facts" in task:
        sid = re.search(r"Session ID: (\S+)", user).group(1)
        return json.dumps(FACTS[sid])
    if "summarize_session" in task:
        return f"summary for {re.search(r'Session date: (.*)', user).group(1)}"
    return default_mock_handler(system, user)  # conflict: values differ -> yes


def _extractor(llm, store, tmp_path) -> FactExtractor:
    return FactExtractor(llm, store, tmp_path / "cache", model_name="test-model")


def test_parse_json_array_robust():
    assert parse_json_array('Here you go: [{"a": 1}] hope it helps') == [{"a": 1}]
    assert parse_json_array("no json here") == []
    assert parse_json_array("[1, 2, 3]") == []  # not objects
    assert parse_json_array('[{"a": 1}, "junk", {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_ingest_builds_versioned_facts(store, tmp_path, sample_sessions):
    llm = MockLLM(handler=_handler)
    extractor = _extractor(llm, store, tmp_path)
    run(extractor.ingest_question("q1", sample_sessions))

    facts = run(store.get_facts("q1"))
    assert len(facts) == 2
    by_value = {f.value: f for f in facts}
    old, new = by_value["Berlin"], by_value["Munich"]
    assert old.valid_to == "2023-06-15"
    assert old.superseded_by == new.id
    assert new.valid_to is None

    timeline = run(store.get_timeline("q1", "user", "lives_in"))
    assert [f.value for f in timeline] == ["Berlin", "Munich"]

    sessions = run(store.get_sessions("q1"))
    assert sessions[0].summary is not None


def test_ingest_is_idempotent_via_has_question(store, tmp_path, sample_sessions):
    llm = MockLLM(handler=_handler)
    extractor = _extractor(llm, store, tmp_path)
    run(extractor.ingest_question("q1", sample_sessions))
    calls_after_first = len(llm.calls)
    run(extractor.ingest_question("q1", sample_sessions))  # skip_if_ingested default
    assert len(llm.calls) == calls_after_first  # no new LLM calls


def test_extraction_cache_avoids_llm(store, tmp_path, sample_sessions):
    llm = MockLLM(handler=_handler)
    extractor = _extractor(llm, store, tmp_path)
    facts1, summary1 = run(extractor.extract_session("q1", sample_sessions[0]))
    assert facts1 and summary1

    def boom(system, user):
        raise AssertionError("LLM must not be called — cache should hit")

    extractor2 = FactExtractor(MockLLM(handler=boom), store, tmp_path / "cache", model_name="test-model")
    facts2, summary2 = run(extractor2.extract_session("q1", sample_sessions[0]))
    assert [f.id for f in facts2] == [f.id for f in facts1]  # deterministic ids
    assert summary2 == summary1


def test_cache_invalidated_on_model_change(store, tmp_path, sample_sessions):
    llm = MockLLM(handler=_handler)
    extractor = _extractor(llm, store, tmp_path)
    run(extractor.extract_session("q1", sample_sessions[0]))

    llm2 = MockLLM(handler=_handler)
    extractor2 = FactExtractor(llm2, store, tmp_path / "cache", model_name="other-model")
    run(extractor2.extract_session("q1", sample_sessions[0]))
    assert len(llm2.calls) > 0  # cache miss -> LLM called


def test_as_of_search_reflects_supersession(store, tmp_path, sample_sessions):
    llm = MockLLM(handler=_handler)
    extractor = _extractor(llm, store, tmp_path)
    run(extractor.ingest_question("q1", sample_sessions))

    current = run(store.search("q1", "where does the user live", as_of_date="2023-06-20", include_sessions=False))
    assert current and all("Munich" in e.text for e in current)

    past = run(store.search("q1", "where does the user live", as_of_date="2023-05-10", include_sessions=False))
    assert past and all("Berlin" in e.text for e in past)
