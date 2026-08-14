"""Streaming loader for LongMemEval dataset files (top-level JSON array of instances).

The s_cleaned file is ~277MB; we stream one instance at a time instead of json.load.
`has_answer` is stripped implicitly: turns are parsed into `Turn(role, content)` and
pydantic ignores every other key, so no model ever sees answer markers.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterator
from pathlib import Path

from ..schema import EvalInstance, Session
from ..text_utils import compare_dates


def iter_json_array(path: str | Path, *, chunk_size: int = 1 << 22) -> Iterator[dict]:
    """Yield objects from a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buf = ""
    started = False
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            eof = chunk == ""
            buf += chunk
            while True:
                i, n = 0, len(buf)
                while i < n and buf[i] in " \t\r\n,":
                    i += 1
                if not started:
                    if i < n and buf[i] == "[":
                        i += 1
                        started = True
                        while i < n and buf[i] in " \t\r\n":
                            i += 1
                    elif i < n:
                        raise ValueError(f"{path}: expected a top-level JSON array")
                if i >= n:
                    buf = ""
                    break
                if buf[i] == "]":
                    return
                buf = buf[i:]
                try:
                    obj, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break  # incomplete object — read more
                yield obj
                buf = buf[end:]
            if eof:
                return


def iter_instances(
    path: str | Path,
    *,
    limit: int | None = None,
    ids: set[str] | None = None,
    question_types: set[str] | None = None,
    exclude_abs: bool = False,
) -> Iterator[EvalInstance]:
    """Stream validated EvalInstance objects with optional filtering."""
    seen = 0
    for obj in iter_json_array(path):
        inst = EvalInstance.model_validate(obj)
        if ids is not None and inst.question_id not in ids:
            continue
        if question_types is not None and inst.question_type not in question_types:
            continue
        if exclude_abs and inst.is_abstention:
            continue
        yield inst
        seen += 1
        if limit is not None and seen >= limit:
            return


def sessions_from_instance(inst: EvalInstance) -> list[Session]:
    """Build Sessions (has_answer stripped via Turn), sorted chronologically."""
    sessions = [
        Session(session_id=sid, date=date, turns=turns)
        for sid, date, turns in zip(
            inst.haystack_session_ids,
            inst.haystack_dates,
            inst.haystack_sessions,
            strict=False,
        )
    ]
    return sorted(sessions, key=functools.cmp_to_key(lambda a, b: compare_dates(a.date, b.date)))


def load_gold_index(path: str | Path) -> dict[str, dict[str, str]]:
    """qid -> {question, answer, question_type, question_date} for the judge."""
    index: dict[str, dict[str, str]] = {}
    for inst in iter_instances(path):
        index[inst.question_id] = {
            "question": inst.question,
            "answer": inst.answer,
            "question_type": inst.question_type,
            "question_date": inst.question_date,
        }
    return index
