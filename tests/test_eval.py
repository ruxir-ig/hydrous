import json

from hydra_mem.answer.reader import Answerer
from hydra_mem.embeddings import TfidfEmbedder
from hydra_mem.eval.judge import judge_hypotheses, select_template
from hydra_mem.eval.metrics import compute_metrics, format_metrics
from hydra_mem.eval.runner import EvalRunner, load_completed_ids
from hydra_mem.ingest.extractor import FactExtractor
from hydra_mem.llm import MockLLM
from hydra_mem.retrieve.pipeline import RetrievalPipeline
from hydra_mem.schema import ABSTAIN_STRING
from hydra_mem.store.local import LocalMemoryStore

from .conftest import run


def _instance(qid, qtype, question, answer, qdate, sessions):
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": question,
        "answer": answer,
        "question_date": qdate,
        "haystack_session_ids": [s[0] for s in sessions],
        "haystack_dates": [s[1] for s in sessions],
        "haystack_sessions": [
            [{"role": role, "content": content, "has_answer": True} for role, content in s[2]]
            for s in sessions
        ],
        "answer_session_ids": [sessions[0][0]],
    }


def _dataset(tmp_path):
    instances = [
        _instance(
            "q_color", "single-session-user", "What is my favorite color?", "blue",
            "2023/05/10 (Wed) 12:00",
            [("sess1", "2023/05/01 (Mon) 10:00",
              [("user", "My favorite color is blue."), ("assistant", "Nice choice!")])],
        ),
        _instance(
            "q_shoesize_abs", "single-session-user", "What is my shoe size?",
            "Shoe size was never discussed in the chat history.",
            "2023/05/10 (Wed) 12:00",
            [("sess1", "2023/05/02 (Tue) 11:00",
              [("user", "I enjoy cooking Italian food."), ("assistant", "Great!")])],
        ),
        _instance(
            "q_move", "knowledge-update", "Where do I live now?", "Munich",
            "2023/06/10 (Sat) 12:00",
            [("s1", "2023/05/01 (Mon) 10:00", [("user", "I live in Berlin."), ("assistant", "ok")]),
             ("s2", "2023/06/01 (Thu) 09:00", [("user", "I moved to Munich last month."), ("assistant", "ok")])],
        ),
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(instances))
    return path


def _runner(tmp_path):
    llm = MockLLM()
    store = LocalMemoryStore(tmp_path / "eval.sqlite3", TfidfEmbedder(dim=512))
    extractor = FactExtractor(llm, store, tmp_path / "cache", model_name="mock-model")
    answerer = Answerer(llm, RetrievalPipeline(llm, store, top_k=10))
    return EvalRunner(extractor, answerer), store


def test_template_selection():
    assert "unanswerable" in select_template("x_abs", "single-session-user")
    assert "off-by-one" in select_template("x", "temporal-reasoning")
    assert "updated answer" in select_template("x", "knowledge-update")
    assert "rubric" in select_template("x", "single-session-preference")
    assert "correct answer" in select_template("x", "multi-session")


def test_runner_judge_metrics_end_to_end(tmp_path):
    dataset = _dataset(tmp_path)
    runner, store = _runner(tmp_path)
    out = tmp_path / "hyp.jsonl"

    results = run(runner.run(dataset, out))
    assert len(results) == 3
    by_qid = {r.question_id: r for r in results}
    assert not by_qid["q_color"].abstained
    assert by_qid["q_shoesize_abs"].abstained
    assert by_qid["q_shoesize_abs"].answer == ABSTAIN_STRING

    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
    assert all(set(line) == {"question_id", "hypothesis"} for line in lines)

    # resume: everything already done -> no new results, file unchanged
    assert load_completed_ids(out) == {"q_color", "q_shoesize_abs", "q_move"}
    assert run(runner.run(dataset, out)) == []
    assert len(out.read_text().splitlines()) == 3

    # judge with MockLLM: color -> yes, abstention -> yes, ku -> no (mock reader is dumb)
    records = run(judge_hypotheses(out, dataset, llm=MockLLM(), judge_model="mock-model"))
    judgments = {r["question_id"]: r["judgment"] for r in records}
    assert judgments == {"q_color": True, "q_shoesize_abs": True, "q_move": False}

    metrics = compute_metrics(str(out) + ".log")
    assert metrics["n"] == 3
    assert metrics["overall"] == 2 / 3
    assert metrics["abstention"] == {"n": 1, "acc": 1.0}
    assert metrics["by_type"]["knowledge-update"] == {"n": 1, "acc": 0.0}
    table = format_metrics(metrics)
    assert "OVERALL" in table and "knowledge-update" in table
    store.close()


def test_runner_limit_and_type_filter(tmp_path):
    dataset = _dataset(tmp_path)
    runner, store = _runner(tmp_path)
    out = tmp_path / "hyp.jsonl"
    results = run(runner.run(dataset, out, limit=1))
    assert len(results) == 1

    out2 = tmp_path / "hyp2.jsonl"
    results = run(runner.run(dataset, out2, question_types={"knowledge-update"}))
    assert [r.question_id for r in results] == ["q_move"]

    out3 = tmp_path / "hyp3.jsonl"
    results = run(runner.run(dataset, out3, exclude_abs=True))
    assert {r.question_id for r in results} == {"q_color", "q_move"}
    store.close()
