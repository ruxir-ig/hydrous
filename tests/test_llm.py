import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, RateLimitError

from hydrous import prompts
from hydrous.llm import MockLLM, OpenAICompatibleLLM, default_mock_handler
from hydrous.schema import ABSTAIN_STRING, Session, Turn

from .conftest import run


def _session() -> Session:
    return Session(
        session_id="s1",
        date="2023/05/01 (Mon) 10:00",
        turns=[
            Turn(role="user", content="My favorite color is blue."),
            Turn(role="assistant", content="Nice!"),
        ],
    )


# ------------------------------------------------------------------ MockLLM


def test_mock_script_fifo_and_recording():
    llm = MockLLM(script=["first", "second"])
    msgs = [{"role": "user", "content": "hi"}]
    assert run(llm.chat(msgs)) == "first"
    assert run(llm.chat(msgs, model="gpt-x")) == "second"
    assert len(llm.calls) == 2
    assert llm.calls[1]["model"] == "gpt-x"
    # script exhausted -> falls back to default handler
    assert run(llm.chat(msgs)) == "ok"


def test_mock_extraction_handler():
    llm = MockLLM()
    resp = run(llm.chat(prompts.extraction_messages(_session())))
    facts = json.loads(resp)
    assert facts == [
        {
            "subject": "user",
            "predicate": "my_favorite_color",
            "value": "My favorite color is blue.",
            "fact_date": "2023/05/01 (Mon) 10:00",
        }
    ]


def test_mock_conflict_handler():
    differ = prompts.conflict_messages("user", "lives_in", "Berlin", "2023-05-01", "Munich", "2023-06-15")
    same = prompts.conflict_messages("user", "lives_in", "Berlin", "2023-05-01", "Berlin", "2023-06-15")
    assert default_mock_handler(differ[0]["content"], differ[1]["content"]) == "yes"
    assert default_mock_handler(same[0]["content"], same[1]["content"]) == "no"


def test_mock_sufficiency_handler():
    enough = prompts.sufficiency_messages(
        "What is my favorite color?", "2023-06-01", "[2023-05-01] (fact, session s1) user favorite_color: blue"
    )
    missing = prompts.sufficiency_messages(
        "What is my shoe size?", "2023-06-01", "[2023-05-01] (fact, session s1) user hobby: cooking"
    )
    assert default_mock_handler(enough[0]["content"], enough[1]["content"]) == "yes"
    assert default_mock_handler(missing[0]["content"], missing[1]["content"]) == "no"


def test_mock_reader_handler_answers_and_abstains():
    evidence = "[2023-05-01] (fact, session s1) user favorite_color: blue"
    msgs = prompts.reader_messages("What is my favorite color?", "2023-06-01", evidence)
    answer = default_mock_handler(msgs[0]["content"], msgs[1]["content"])
    assert "blue" in answer

    msgs = prompts.reader_messages("What is my shoe size?", "2023-06-01", evidence)
    assert default_mock_handler(msgs[0]["content"], msgs[1]["content"]) == ABSTAIN_STRING


def test_mock_judge_handler():
    correct = "Question: q\nCorrect Answer: blue\nModel Response: the color is blue\nIs the model response correct?"
    wrong = "Question: q\nCorrect Answer: blue\nModel Response: it is red\nIs the model response correct?"
    abstained = "Question: q\nExplanation: never discussed\nModel Response: " + ABSTAIN_STRING
    hallucinated = "Question: q\nExplanation: never discussed\nModel Response: size 42"
    assert default_mock_handler("", correct) == "yes"
    assert default_mock_handler("", wrong) == "no"
    assert default_mock_handler("", "Does the model correctly identify the question as unanswerable?\n" + abstained) == "yes"
    assert default_mock_handler("", "unanswerable\n" + hallucinated) == "no"


# ------------------------------------------------------------------ OpenAICompatibleLLM retry


def _fake_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    return RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)


class _FakeCompletions:
    def __init__(self, failures: list[Exception], content: str = "ok"):
        self._failures = list(failures)
        self._content = content
        self.attempts = 0

    async def create(self, **kwargs):
        self.attempts += 1
        if self._failures:
            raise self._failures.pop(0)
        return _fake_response(self._content)


class _FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def test_retry_recovers_from_rate_limit():
    completions = _FakeCompletions([_rate_limit_error(), _rate_limit_error()], content="fine")
    llm = OpenAICompatibleLLM(
        api_key="x", base_url=None, model="m", max_retries=5, client=_FakeClient(completions)
    )
    llm_chat = llm.chat([{"role": "user", "content": "hi"}])
    # patch out the backoff sleep so the test is instant
    import hydrous.llm as llm_mod

    orig_sleep = llm_mod.asyncio.sleep

    async def no_sleep(_):
        return None

    llm_mod.asyncio.sleep = no_sleep
    try:
        assert run(llm_chat) == "fine"
    finally:
        llm_mod.asyncio.sleep = orig_sleep
    assert completions.attempts == 3


def test_client_error_4xx_not_retried():
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    err = APIStatusError("bad request", response=httpx.Response(400, request=request), body=None)
    completions = _FakeCompletions([err])
    llm = OpenAICompatibleLLM(
        api_key="x", base_url=None, model="m", max_retries=5, client=_FakeClient(completions)
    )
    with pytest.raises(APIStatusError):
        run(llm.chat([{"role": "user", "content": "hi"}]))
    assert completions.attempts == 1
