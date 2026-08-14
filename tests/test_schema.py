from hydrous.schema import EvalInstance, Evidence, EvidencePack, Fact


def _fact(valid_from="2023-05-01", valid_to=None) -> Fact:
    return Fact(
        question_id="q1",
        subject="user",
        predicate="lives_in",
        value="Berlin",
        fact_date=valid_from,
        session_id="s1",
        valid_from=valid_from,
        valid_to=valid_to,
    )


def test_fact_valid_as_of_open_interval():
    f = _fact()
    assert f.valid_as_of("2023-05-01")
    assert f.valid_as_of("2024-01-01")
    assert not f.valid_as_of("2023-04-01")
    assert f.valid_as_of(None)


def test_fact_valid_as_of_closed_interval():
    f = _fact(valid_to="2023-06-15")
    assert f.valid_as_of("2023-05-10")
    assert not f.valid_as_of("2023-06-15")  # valid_to is exclusive
    assert not f.valid_as_of("2023-06-20")
    assert not f.valid_as_of(None)  # not currently valid


def test_fact_text():
    assert _fact().text == "user lives_in: Berlin"


def test_evidence_pack_chronological_and_prompt():
    pack = EvidencePack(
        question="q",
        question_date="2023-06-20",
        items=[
            Evidence(source="fact", text="later\nfact", date="2023/06/15 (Thu) 09:00", session_id="s2", fact_id="f2"),
            Evidence(source="fact", text="earlier fact", date="2023/05/01 (Mon) 10:00", session_id="s1", fact_id="f1"),
        ],
    )
    chrono = pack.chronological()
    assert [e.fact_id for e in chrono.items] == ["f1", "f2"]
    prompt = pack.to_prompt()
    lines = prompt.splitlines()
    assert len(lines) == 2  # whitespace-collapsed: one line per item
    assert lines[0].startswith("[2023-05-01 10:00] (fact, session s1) earlier fact")
    assert lines[1].startswith("[2023-06-15 09:00] (fact, session s2) later fact")


def test_eval_instance_strips_has_answer():
    inst = EvalInstance.model_validate(
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "q",
            "answer": "a",
            "question_date": "2023/05/10 (Wed) 12:00",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/05/01 (Mon) 10:00"],
            "haystack_sessions": [[{"role": "user", "content": "hi", "has_answer": True}]],
            "answer_session_ids": ["s1"],
        }
    )
    assert inst.haystack_sessions[0][0].role == "user"
    assert "has_answer" not in inst.haystack_sessions[0][0].model_dump()


def test_is_abstention():
    base = {
        "question_type": "single-session-user",
        "question": "q",
        "answer": "a",
        "question_date": "2023/05/10 (Wed) 12:00",
        "haystack_session_ids": [],
        "haystack_dates": [],
        "haystack_sessions": [],
    }
    assert EvalInstance.model_validate({**base, "question_id": "x_abs"}).is_abstention
    assert not EvalInstance.model_validate({**base, "question_id": "x"}).is_abstention
