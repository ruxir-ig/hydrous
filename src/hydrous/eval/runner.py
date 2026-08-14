"""Eval runner: instances → ingest (cached) → answer → hypotheses JSONL. Resumable."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from ..answer.reader import Answerer, AnswerResult
from ..ingest.extractor import FactExtractor
from ..ingest.longmemeval import iter_instances, sessions_from_instance
from ..schema import EvalInstance

log = logging.getLogger(__name__)


def load_completed_ids(out_path: str | Path) -> set[str]:
    """question_ids already present in the hypotheses file (for resume)."""
    ids: set[str] = set()
    path = Path(out_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


class EvalRunner:
    def __init__(self, extractor: FactExtractor, answerer: Answerer):
        self._extractor = extractor
        self._answerer = answerer

    async def run(
        self,
        dataset_path: str | Path,
        out_path: str | Path,
        *,
        limit: int | None = None,
        question_types: set[str] | None = None,
        ids: set[str] | None = None,
        exclude_abs: bool = False,
        reingest: bool = False,
        on_result: Callable[[EvalInstance, AnswerResult], None] | None = None,
    ) -> list[AnswerResult]:
        """Write {"question_id", "hypothesis"} JSONL. Skips question_ids already in out_path."""
        done = load_completed_ids(out_path)
        if done:
            log.info("resuming: %d question_ids already in %s", len(done), out_path)
        results: list[AnswerResult] = []
        considered = 0
        with open(out_path, "a", encoding="utf-8") as f:
            for inst in iter_instances(
                dataset_path, ids=ids, question_types=question_types, exclude_abs=exclude_abs
            ):
                if limit is not None and considered >= limit:
                    break
                considered += 1
                if inst.question_id in done:
                    continue
                if reingest:
                    await self._extractor.store.delete_question(inst.question_id)
                sessions = sessions_from_instance(inst)
                await self._extractor.ingest_question(
                    inst.question_id, sessions, skip_if_ingested=not reingest
                )
                result = await self._answerer.answer(
                    inst.question_id, inst.question, inst.question_date
                )
                f.write(
                    json.dumps({"question_id": inst.question_id, "hypothesis": result.answer}) + "\n"
                )
                f.flush()
                results.append(result)
                if on_result:
                    on_result(inst, result)
        return results
