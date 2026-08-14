import json

from hydra_mem.ingest.longmemeval import (
    iter_instances,
    iter_json_array,
    load_gold_index,
    sessions_from_instance,
)


def _instance(qid, qtype="single-session-user", abs_id=False):
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": f"question for {qid}",
        "answer": "some answer",
        "question_date": "2023/06/10 (Sat) 12:00",
        # deliberately out of chronological order
        "haystack_session_ids": ["s2", "s1"],
        "haystack_dates": ["2023/06/01 (Thu) 09:00", "2023/05/01 (Mon) 10:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "later", "has_answer": False}],
            [{"role": "user", "content": "earlier", "has_answer": True}],
        ],
        "answer_session_ids": ["s1"],
    }


def _write_dataset(tmp_path, instances) -> str:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(instances, indent=2))  # pretty-printed: multi-line objects
    return str(path)


def test_iter_json_array_tiny_chunks(tmp_path):
    path = _write_dataset(tmp_path, [_instance("a"), _instance("b")])
    objs = list(iter_json_array(path, chunk_size=64))  # stress incremental parsing
    assert [o["question_id"] for o in objs] == ["a", "b"]


def test_iter_instances_and_filters(tmp_path):
    path = _write_dataset(tmp_path, [_instance("a"), _instance("b", qtype="knowledge-update"), _instance("c_abs")])
    assert [i.question_id for i in iter_instances(path)] == ["a", "b", "c_abs"]
    assert [i.question_id for i in iter_instances(path, limit=2)] == ["a", "b"]
    assert [i.question_id for i in iter_instances(path, ids={"b"})] == ["b"]
    assert [i.question_id for i in iter_instances(path, question_types={"knowledge-update"})] == ["b"]
    assert [i.question_id for i in iter_instances(path, exclude_abs=True)] == ["a", "b"]


def test_sessions_sorted_and_has_answer_stripped(tmp_path):
    path = _write_dataset(tmp_path, [_instance("a")])
    inst = next(iter_instances(path))
    sessions = sessions_from_instance(inst)
    assert [s.session_id for s in sessions] == ["s1", "s2"]  # chronological
    for s in sessions:
        for t in s.turns:
            assert "has_answer" not in t.model_dump()


def test_load_gold_index(tmp_path):
    path = _write_dataset(tmp_path, [_instance("a"), _instance("b")])
    index = load_gold_index(path)
    assert set(index) == {"a", "b"}
    assert index["a"]["question"] == "question for a"
    assert index["a"]["question_type"] == "single-session-user"
