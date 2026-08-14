from hydra_mem import prompts
from hydra_mem.answer.reader import Answerer
from hydra_mem.llm import MockLLM, default_mock_handler
from hydra_mem.retrieve.pipeline import RetrievalPipeline
from hydra_mem.schema import ABSTAIN_STRING, Fact

from .conftest import run


def _color_fact() -> Fact:
    return Fact(
        id="f1", question_id="q1", subject="user", predicate="favorite_color",
        value="My favorite color is blue.", fact_date="2023-05-01", session_id="s1",
        valid_from="2023-05-01",
    )


def _answerer(llm, store, **kwargs) -> Answerer:
    return Answerer(llm, RetrievalPipeline(llm, store, top_k=10), **kwargs)


def test_empty_evidence_abstains_without_gate_call(store, mock_llm):
    result = run(_answerer(mock_llm, store).answer("q1", "What is my favorite color?", "2023-06-01"))
    assert result.abstained
    assert result.answer == ABSTAIN_STRING
    gate_calls = [c for c in mock_llm.calls if prompts.TASK_SUFFICIENCY in c["messages"][0]["content"]]
    assert gate_calls == []  # no evidence -> gate skipped entirely


def test_gate_rejects_insufficient_evidence(store):
    run(store.add_facts("q1", [_color_fact()]))

    def handler(system, user):
        if prompts.TASK_SUFFICIENCY in system:
            return "no"
        return default_mock_handler(system, user)

    llm = MockLLM(handler=handler)
    result = run(_answerer(llm, store).answer("q1", "What is my favorite color?", "2023-06-01"))
    assert result.abstained
    assert result.answer == ABSTAIN_STRING


def test_gate_passes_and_reader_answers(store, mock_llm):
    run(store.add_facts("q1", [_color_fact()]))
    result = run(_answerer(mock_llm, store).answer("q1", "What is my favorite color?", "2023-06-01"))
    assert not result.abstained
    assert "blue" in result.answer


def test_reader_idk_output_normalized(store):
    run(store.add_facts("q1", [_color_fact()]))

    def handler(system, user):
        if prompts.TASK_ANSWER in system:
            return "I don't know based on this."
        return default_mock_handler(system, user)

    llm = MockLLM(handler=handler)
    result = run(_answerer(llm, store).answer("q1", "What is my favorite color?", "2023-06-01"))
    assert result.abstained
    assert result.answer == ABSTAIN_STRING


def test_gate_disabled(store):
    run(store.add_facts("q1", [_color_fact()]))
    llm = MockLLM()
    result = run(_answerer(llm, store, use_gate=False).answer("q1", "color?", "2023-06-01"))
    gate_calls = [c for c in llm.calls if prompts.TASK_SUFFICIENCY in c["messages"][0]["content"]]
    assert gate_calls == []
    assert result.answer  # reader still ran
