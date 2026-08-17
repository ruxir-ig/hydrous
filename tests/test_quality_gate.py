"""Extraction quality gate: scored facts, junk REJECTed before storage, score kept in metadata.

The gate is the answer to mem0 issue #4573's "98% junk extraction" complaint:
every extracted fact carries an LLM confidence, facts below `quality_min_score`
(or explicitly flagged) never reach the store, and the score survives storage
(local sqlite + HydraDB additional_metadata) for audit / later use.
"""

import json
from collections import deque

from hydrous import prompts
from hydrous.ingest.extractor import ExtractionCache, FactExtractor
from hydrous.llm import MockLLM
from hydrous.schema import Session, Turn

from .conftest import run

_MIN = 0.4


def _session() -> Session:
    return Session(
        session_id="s1",
        date="2023/05/01 (Mon) 10:00",
        turns=[Turn(role="user", content="I love hiking every Saturday morning.")],
    )


def _extractor(store, llm, tmp_path, script: list[str]):
    ex = FactExtractor(llm, store, tmp_path, model_name="mock-model", quality_min_score=_MIN)
    llm._script = deque(script)
    return ex


def test_low_confidence_and_reject_flag_dropped(store, tmp_path):
    llm = MockLLM()
    raw = json.dumps(
        [
            {"subject": "user", "predicate": "enjoys_hiking", "value": "every Saturday morning",
             "fact_date": "2023-05-01", "confidence": 0.9},
            {"subject": "user", "predicate": "small_talk", "value": "How are you?",
             "fact_date": "2023-05-01", "confidence": 0.1},
            {"subject": "user", "predicate": "transient", "value": "user is tired right now",
             "fact_date": "2023-05-01", "confidence": 0.8, "reject": True},
        ]
    )
    ex = _extractor(store, llm, tmp_path, [raw, "summary"])
    facts, _summary = run(ex.extract_session("q1", _session()))
    assert [f.predicate for f in facts] == ["enjoys_hiking"]
    assert facts[0].quality_score == 0.9


def test_missing_confidence_is_kept_ungated(store, tmp_path):
    # models that don't emit confidence: legacy pass-through (score None, kept)
    llm = MockLLM()
    raw = json.dumps(
        [{"subject": "user", "predicate": "lives_in", "value": "Paris", "fact_date": "2023-05-01"}]
    )
    ex = _extractor(store, llm, tmp_path, [raw, "summary"])
    facts, _ = run(ex.extract_session("q1", _session()))
    assert [f.predicate for f in facts] == ["lives_in"]
    assert facts[0].quality_score is None


def test_quality_score_persists_in_local_store(store, tmp_path):
    llm = MockLLM()
    raw = json.dumps(
        [{"subject": "user", "predicate": "enjoys_hiking", "value": "every Saturday morning",
          "fact_date": "2023-05-01", "confidence": 0.87}]
    )
    ex = _extractor(store, llm, tmp_path, [raw, "summary"])
    facts, _ = run(ex.extract_session("q1", _session()))
    run(ex.store_facts("q1", facts))
    stored = run(store.get_facts("q1"))
    assert [(f.predicate, f.quality_score) for f in stored] == [("enjoys_hiking", 0.87)]


def test_hydradb_item_carries_quality_score():
    from hydrous.schema import Fact
    from hydrous.store.hydradb import fact_to_memory_item

    fact = Fact(
        id="f1", question_id="q1", subject="user", predicate="enjoys_hiking",
        value="hiking", fact_date="2023-05-01", session_id="s1", valid_from="2023-05-01",
        quality_score=0.82,
    )
    am = fact_to_memory_item(fact)["additional_metadata"]
    assert am["quality_score"] == 0.82

    unscored = fact.model_copy(update={"quality_score": None})
    assert "quality_score" not in fact_to_memory_item(unscored)["additional_metadata"]


def test_v2_prompt_mentions_rejection_and_scoring():
    sys = prompts.EXTRACTION_SYSTEM
    assert "confidence" in sys
    assert "REJECT" in sys
    assert "small talk" in sys.lower() and "boilerplate" in sys.lower()
    assert "0..1" in sys
    assert prompts.EXTRACTION_PROMPT_VERSION == 2


def test_extraction_cache_invalidates_on_prompt_version(store, tmp_path):
    # cache entry written under an older prompt version is a miss (re-extract)
    ExtractionCache(tmp_path, prompt_version=1).save(
        "q1", "s1", "mock-model",
        [{"subject": "user", "predicate": "old", "value": "v", "fact_date": "2023-05-01"}],
        "old summary",
    )
    llm = MockLLM()
    ex = FactExtractor(llm, store, tmp_path, model_name="mock-model")  # current version (2)
    assert ex._cache.load("q1", "s1", "mock-model") is None
    raw = json.dumps([{"subject": "user", "predicate": "new", "value": "v",
                       "fact_date": "2023-05-01", "confidence": 1.0}])
    llm._script = deque([raw, "summary"])
    facts, _ = run(ex.extract_session("q1", _session()))
    assert [f.predicate for f in facts] == ["new"]


def test_rejects_malformed_fact_missing_fields(store, tmp_path):
    llm = MockLLM()
    # quality gate MUST NOT rescue structurally invalid facts (no value to score)
    raw = json.dumps(
        [
            {"subject": "user", "predicate": "", "value": "x", "fact_date": "2023-05-01", "confidence": 0.9},
            {"subject": "", "predicate": "p", "value": "", "fact_date": "2023-05-01", "confidence": 0.9},
        ]
    )
    ex = _extractor(store, llm, tmp_path, [raw, "summary"])
    facts, _ = run(ex.extract_session("q1", _session()))
    assert facts == []
