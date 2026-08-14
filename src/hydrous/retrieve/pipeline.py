"""Retrieval: query analysis → hybrid search → as-of-date filter → chronological EvidencePack."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from .. import prompts
from ..llm import LLMClient
from ..schema import Evidence, EvidencePack
from ..store.base import MemoryStore

log = logging.getLogger(__name__)

_PREVIOUS_INTENT_RE = re.compile(
    r"\b(previously|before|used to|originally|earlier|in the past|did i (think|say|believe|have))\b",
    re.IGNORECASE,
)


@dataclass
class QueryAnalysis:
    rewritten_query: str
    subqueries: list[str] = field(default_factory=list)
    time_range: dict | None = None
    question_type_hint: str | None = None


def parse_json_object(text: str) -> dict:
    """Robustly extract a JSON object from an LLM response."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


class RetrievalPipeline:
    def __init__(
        self,
        llm: LLMClient,
        store: MemoryStore,
        *,
        top_k: int = 12,
        use_query_analysis: bool = True,
    ):
        self._llm = llm
        self._store = store
        self._top_k = top_k
        self._use_query_analysis = use_query_analysis

    async def analyze_query(self, question: str, question_date: str) -> QueryAnalysis:
        """LLM rewrite with time expressions resolved against question_date.

        Any failure (bad JSON, LLM error) falls back to the identity rewrite."""
        if not self._use_query_analysis:
            return QueryAnalysis(rewritten_query=question)
        try:
            resp = await self._llm.chat(prompts.query_analysis_messages(question, question_date))
            data = parse_json_object(resp)
            if not data:
                raise ValueError("empty analysis")
            return QueryAnalysis(
                rewritten_query=str(data.get("rewritten_query") or question),
                subqueries=[str(s) for s in (data.get("subqueries") or [])][:3],
                time_range=data.get("time_range"),
                question_type_hint=data.get("question_type_hint"),
            )
        except Exception:
            log.warning("query analysis failed; using identity rewrite", exc_info=True)
            return QueryAnalysis(rewritten_query=question)

    async def retrieve(
        self, question_id: str, question: str, question_date: str, *, top_k: int | None = None
    ) -> EvidencePack:
        top_k = top_k or self._top_k
        analysis = await self.analyze_query(question, question_date)
        queries = [analysis.rewritten_query, *analysis.subqueries]
        # "What did I previously think..." needs superseded versions; current-state
        # questions get them filtered out by the as-of-date validity check.
        include_superseded = bool(_PREVIOUS_INTENT_RE.search(question))

        best: dict[str, Evidence] = {}
        for q in queries:
            if not q or not q.strip():
                continue
            results = await self._store.search(
                question_id, q,
                as_of_date=question_date,
                top_k=top_k,
                include_superseded=include_superseded,
            )
            for ev in results:
                if ev.key not in best or ev.score > best[ev.key].score:
                    best[ev.key] = ev

        items = sorted(best.values(), key=lambda e: e.score, reverse=True)[:top_k]
        pack = EvidencePack(question=question, question_date=question_date, items=items)
        return pack.chronological()
