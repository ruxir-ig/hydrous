"""End-to-end smoke test on real LongMemEval oracle instances with MockLLM.

Deterministic, no network, no API key. Skipped only if the dataset is not downloaded.
Asserts pipeline plumbing (ingest → retrieve → answer → JSONL → judge → metrics),
not answer quality — that's the real model's job.
"""

import json
from pathlib import Path

import pytest

from hydra_mem.answer.reader import Answerer
from hydra_mem.embeddings import TfidfEmbedder
from hydra_mem.eval.judge import judge_hypotheses
from hydra_mem.eval.metrics import compute_metrics
from hydra_mem.eval.runner import EvalRunner
from hydra_mem.ingest.extractor import FactExtractor
from hydra_mem.llm import MockLLM
from hydra_mem.retrieve.pipeline import RetrievalPipeline
from hydra_mem.store.local import LocalMemoryStore

from .conftest import run

ORACLE = Path("data/longmemeval_oracle.json")


@pytest.mark.skipif(not ORACLE.exists(), reason="oracle dataset not downloaded")
def test_e2e_oracle_with_mock(tmp_path):
    llm = MockLLM()
    store = LocalMemoryStore(tmp_path / "e2e.sqlite3", TfidfEmbedder(dim=512))
    extractor = FactExtractor(llm, store, tmp_path / "cache", model_name="mock-model")
    answerer = Answerer(llm, RetrievalPipeline(llm, store, top_k=10))
    runner = EvalRunner(extractor, answerer)
    out = tmp_path / "hyp.jsonl"

    results = run(runner.run(ORACLE, out, limit=3))
    assert len(results) == 3
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
    assert all(set(line) == {"question_id", "hypothesis"} for line in lines)
    assert all(isinstance(line["hypothesis"], str) and line["hypothesis"] for line in lines)

    # resume is a no-op
    assert run(runner.run(ORACLE, out, limit=3)) == []

    # extraction cache was written
    assert list((tmp_path / "cache" / "extractions").glob("*.json"))

    records = run(judge_hypotheses(out, ORACLE, llm=MockLLM(), judge_model="mock-model"))
    assert len(records) == 3
    metrics = compute_metrics(str(out) + ".log")
    assert metrics["n"] == 3
    assert metrics["overall"] is not None
    store.close()
