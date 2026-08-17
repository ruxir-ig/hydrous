"""The one LLM client: OpenAI-compatible async client + deterministic MockLLM.

All LLM calls in the codebase go through `LLMClient.chat`. Swap providers via env config;
swap intelligence off entirely with MockLLM (tests, dry runs).
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections import deque
from collections.abc import Callable
from typing import Protocol

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from . import prompts
from .schema import ABSTAIN_STRING
from .text_utils import content_words, tokenize, word_overlap

Message = dict[str, str]


class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str: ...


class OpenAICompatibleLLM:
    """Async client for any OpenAI-compatible endpoint (OpenAI / OpenRouter / custom).

    Hardening notes (see RUN_NOTES "endpoint hangs"): the custom endpoint silently
    swallows some requests under HTTP/2 multiplexing — the connection stays ESTAB
    but the response never arrives and httpx's read timeout never fires. So every
    call gets a hard wall-clock deadline via asyncio.wait_for (independent of httpx),
    and every call runs on a brand-new httpx transport (HTTP/1.1, zero keep-alive):
    fresh TLS connection per request, exactly mirroring the burst tests that never
    hang. The AsyncOpenAI object is rebuilt per call too: long-lived clients have
    shown to stall after a few requests in this pipeline."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 5,
        client: AsyncOpenAI | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._client = client
        self._deadline = timeout
        self.model = model
        self.max_retries = max_retries

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await asyncio.wait_for(
                    self._call(
                        model or self.model, messages, temperature, max_tokens
                    ),
                    timeout=self._deadline,
                )
                return resp
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_exc = e
            except TimeoutError as e:
                last_exc = e
            except APIStatusError as e:
                last_exc = e
                if e.status_code < 500:
                    raise
            await asyncio.sleep(min(2**attempt, 30) + random.random())
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_exc

    async def _call(
        self, model: str, messages: list[Message], temperature: float, max_tokens: int | None
    ) -> str:
        """One attempt on a brand-new transport: fresh client + fresh connection."""
        if self._client is not None:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        http_client = httpx.AsyncClient(
            timeout=self._deadline + 15,
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=8),
            http2=False,
        )
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._deadline + 5,
            max_retries=0,
            http_client=http_client,
        )
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        finally:
            await client.close()


MockHandler = Callable[[str, str], str]


class MockLLM:
    """Deterministic LLM. No network, no key.

    - `script`: FIFO list of canned responses, consumed first (unit tests).
    - `handler`: custom (system, user) -> str; defaults to `default_mock_handler`,
      which dispatches heuristically on the `TASK:` marker of each prompt.
    - `calls`: every invocation is recorded for assertions.
    """

    def __init__(
        self,
        script: list[str] | None = None,
        handler: MockHandler | None = None,
        model: str = "mock-model",
    ):
        self._script: deque[str] = deque(script or [])
        self._handler = handler or default_mock_handler
        self.model = model
        self.calls: list[dict] = []

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        self.calls.append({"messages": messages, "model": model or self.model})
        if self._script:
            return self._script.popleft()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        return self._handler(system, user)


# ---------------------------------------------------------------------------
# Default deterministic handler
# ---------------------------------------------------------------------------

_FIELD_RES = {
    name: re.compile(rf"(?m)^{re.escape(name)}:\s*(.*)$")
    for name in ("Session date", "Question", "Correct Answer", "Rubric", "Explanation", "Model Response")
}


def _field(text: str, name: str) -> str:
    m = _FIELD_RES[name].search(text)
    return m.group(1).strip() if m else ""


def _task_of(system: str) -> str | None:
    m = re.match(r"^TASK:\s*(\w+)", system)
    return m.group(1) if m else None


def default_mock_handler(system: str, user: str) -> str:
    task = _task_of(system)
    if task == prompts.TASK_EXTRACT:
        return _mock_extract(user)
    if task == prompts.TASK_SUMMARIZE:
        return _mock_summarize(user)
    if task == prompts.TASK_ANALYZE:
        return _mock_analyze(user)
    if task == prompts.TASK_CONFLICT:
        return _mock_conflict(user)
    if task == prompts.TASK_SUFFICIENCY:
        return _mock_sufficiency(user)
    if task == prompts.TASK_ANSWER:
        return _mock_answer(user)
    if "unanswerable" in user or "Is the model response correct?" in user:
        return _mock_judge(user)
    return "ok"


def _mock_extract(user: str) -> str:
    """One fact per USER turn; predicate = first 3 content tokens (stable, deterministic)."""
    date = _field(user, "Session date")
    facts = []
    seen = set()
    for line in user.splitlines():
        if not line.startswith("USER:"):
            continue
        content = line[len("USER:") :].strip()
        if len(content) < 5:
            continue
        predicate = "_".join(tokenize(content)[:3]) or "stated"
        key = (predicate, content[:300])
        if key in seen:
            continue
        seen.add(key)
        facts.append({"subject": "user", "predicate": predicate, "value": content[:300], "fact_date": date})
    return json.dumps(facts)


def _mock_summarize(user: str) -> str:
    date = _field(user, "Session date")
    users = [line[len("USER:") :].strip() for line in user.splitlines() if line.startswith("USER:")]
    if not users:
        return f"Session on {date} with no user content."
    return f"On {date}, the user discussed: " + "; ".join(u[:100] for u in users[:3])


def _mock_analyze(user: str) -> str:
    question = _field(user, "Question")
    return json.dumps(
        {"rewritten_query": question, "subqueries": [], "time_range": None, "question_type_hint": None}
    )


def _mock_conflict(user: str) -> str:
    old = re.search(r"(?m)^Existing fact:\s*(.*?)\s*\(date:.*\)$", user)
    new = re.search(r"(?m)^New fact:\s*(.*?)\s*\(date:.*\)$", user)
    if old and new:
        return "yes" if old.group(1) != new.group(1) else "no"
    return "no"


def _mock_sufficiency(user: str) -> str:
    question = _field(user, "Question")
    evidence = user.split("Evidence:", 1)[1] if "Evidence:" in user else ""
    qw = content_words(question)
    if not qw:
        return "no"
    overlap = len(qw & content_words(evidence)) / len(qw)
    return "yes" if overlap >= 0.3 else "no"


def _mock_answer(user: str) -> str:
    question = _field(user, "Question")
    lines = [line for line in user.splitlines() if line.startswith("[")]
    best, best_score = "", 0.0
    for line in lines:
        score = word_overlap(question, line)
        if score > best_score:
            best, best_score = line, score
    if not best or best_score < 0.2:
        return ABSTAIN_STRING
    m = re.match(r"^\[[^\]]*\]\s*\([^)]*\)\s*(.*)$", best)
    snippet = m.group(1) if m else best
    return f"Based on the chat history, {snippet}"[:600]


def _mock_judge(user: str) -> str:
    response = _field(user, "Model Response")
    if "unanswerable" in user:
        lowered = response.lower()
        identified = "don't know" in lowered or "does not contain" in lowered or "cannot answer" in lowered
        return "yes" if identified else "no"
    gold = _field(user, "Correct Answer") or _field(user, "Rubric")
    if not gold or not response:
        return "no"
    return "yes" if word_overlap(gold, response) >= 0.5 else "no"
