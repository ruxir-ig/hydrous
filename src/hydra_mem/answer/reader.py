"""Answering: strict evidence-sufficiency gate (CRAG-style) → Chain-of-Note reader.

Abstention strategy: if the evidence pack is empty OR the gate judges it insufficient,
return the exact abstention string — never hallucinate. The reader is also instructed
to abstain, and any "don't know"-style output is normalized to the exact string.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from .. import prompts
from ..llm import LLMClient
from ..retrieve.pipeline import RetrievalPipeline
from ..schema import ABSTAIN_STRING, EvidencePack

log = logging.getLogger(__name__)

_IDK_MARKERS = ("don't know", "do not know", "does not contain", "not contain this information")


class AnswerResult(BaseModel):
    question_id: str
    answer: str
    abstained: bool
    evidence: EvidencePack


class Answerer:
    def __init__(
        self,
        llm: LLMClient,
        retriever: RetrievalPipeline,
        *,
        use_gate: bool = True,
        abstain_string: str = ABSTAIN_STRING,
    ):
        self._llm = llm
        self._retriever = retriever
        self._use_gate = use_gate
        self._abstain = abstain_string

    async def answer(self, question_id: str, question: str, question_date: str) -> AnswerResult:
        pack = await self._retriever.retrieve(question_id, question, question_date)

        if not pack.items:
            log.info("question %s: empty evidence pack — abstaining", question_id)
            return self._abstain_result(question_id, pack)

        evidence_text = pack.to_prompt()

        if self._use_gate:
            verdict = await self._llm.chat(
                prompts.sufficiency_messages(question, question_date, evidence_text)
            )
            if "yes" not in verdict.lower():
                log.info("question %s: evidence judged insufficient — abstaining", question_id)
                return self._abstain_result(question_id, pack)

        raw = (await self._llm.chat(prompts.reader_messages(question, question_date, evidence_text))).strip()
        if any(marker in raw.lower() for marker in _IDK_MARKERS):
            return self._abstain_result(question_id, pack)
        return AnswerResult(question_id=question_id, answer=raw, abstained=False, evidence=pack)

    def _abstain_result(self, question_id: str, pack: EvidencePack) -> AnswerResult:
        return AnswerResult(
            question_id=question_id, answer=self._abstain, abstained=True, evidence=pack
        )
