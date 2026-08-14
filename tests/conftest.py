from __future__ import annotations

import asyncio

import pytest

from hydra_mem.embeddings import TfidfEmbedder
from hydra_mem.llm import MockLLM
from hydra_mem.schema import Session, Turn
from hydra_mem.store.local import LocalMemoryStore


def run(coro):
    """Run an async coroutine in a fresh event loop (avoids a pytest-asyncio dependency)."""
    return asyncio.run(coro)


@pytest.fixture
def embedder() -> TfidfEmbedder:
    return TfidfEmbedder(dim=256)


@pytest.fixture
def store(tmp_path, embedder):
    s = LocalMemoryStore(tmp_path / "test.sqlite3", embedder)
    yield s
    s.close()


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def sample_sessions() -> list[Session]:
    return [
        Session(
            session_id="s1",
            date="2023/05/01 (Mon) 10:00",
            turns=[
                Turn(role="user", content="I live in Berlin and work as a data scientist."),
                Turn(role="assistant", content="Berlin is a great city for data science!"),
            ],
        ),
        Session(
            session_id="s2",
            date="2023/06/15 (Thu) 09:00",
            turns=[
                Turn(role="user", content="I moved to Munich last week for my new job."),
                Turn(role="assistant", content="Congrats on the move!"),
            ],
        ),
    ]
