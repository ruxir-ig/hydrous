"""Core data models for Hydrous."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .text_utils import compare_dates, date_key

ABSTAIN_STRING = "I don't know. The chat history does not contain this information."


class Turn(BaseModel):
    """One chat turn. Extra keys (e.g. `has_answer`) are dropped on parse — by design."""

    role: str
    content: str


class Session(BaseModel):
    session_id: str
    date: str
    turns: list[Turn]
    summary: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(f"{t.role.upper()}: {t.content}" for t in self.turns)


class Fact(BaseModel):
    """Bi-temporal atomic fact. Append-only: supersession closes valid_to, never deletes."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    question_id: str
    subject: str
    predicate: str
    value: str
    fact_date: str
    session_id: str
    valid_from: str
    valid_to: str | None = None
    recorded_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    superseded_by: str | None = None

    @property
    def text(self) -> str:
        return f"{self.subject} {self.predicate}: {self.value}"

    def valid_as_of(self, date: str | None) -> bool:
        """True if the fact's validity interval covers `date` (None = 'currently valid')."""
        if date is None:
            return self.valid_to is None
        if compare_dates(self.valid_from, date) > 0:
            return False
        return self.valid_to is None or compare_dates(date, self.valid_to) < 0


class Evidence(BaseModel):
    source: str  # "fact" | "session"
    text: str
    date: str
    session_id: str
    score: float = 0.0
    fact_id: str | None = None

    @property
    def key(self) -> str:
        if self.source == "fact" and self.fact_id:
            return f"fact:{self.fact_id}"
        return f"{self.source}:{self.session_id}"


class EvidencePack(BaseModel):
    question: str
    question_date: str
    items: list[Evidence] = Field(default_factory=list)
    question_type_hint: str | None = None

    def chronological(self) -> EvidencePack:
        ordered = sorted(self.items, key=lambda e: (date_key(e.date), e.score))
        return EvidencePack(
            question=self.question,
            question_date=self.question_date,
            items=ordered,
            question_type_hint=self.question_type_hint,
        )

    def to_prompt(self, max_chars_per_item: int = 1500, max_total_chars: int = 24000) -> str:
        """One line per item, whitespace-collapsed, prefixed with its date.

        Compact, high-signal window: a small per-item cap keeps the total input
        inside the reader's reliable attention range. v2's ~14KB compact input
        answered correctly where the 59KB dump abstained (attention dilution —
        see RUN_NOTES "reader budget"). Selected by relevance into the budget,
        then presented chronologically (dates drive temporal reasoning)."""
        top = sorted(self.items, key=lambda e: e.score, reverse=True)
        lines: list[tuple[str, str]] = []
        used = 0
        for ev in top:
            text = ev.text if len(ev.text) <= max_chars_per_item else ev.text[:max_chars_per_item] + "..."
            text = " ".join(text.split())
            used += len(text)
            if used > max_total_chars and lines:
                break
            prefix = f"[{date_key(ev.date) or ev.date}] ({ev.source}, session {ev.session_id}) "
            lines.append((date_key(ev.date) or ev.date, prefix + text))
        lines.sort(key=lambda p: p[0])
        return "\n".join(line for _, line in lines)

    def gate_prompt(self, max_items: int = 12, max_chars_per_item: int = 8000) -> str:
        """Projection for the sufficiency gate: top-scored items, same per-item
        window as the reader's `to_prompt` (8000 chars).

        Per-item truncation must match the reader view — a narrower window made
        the gate abstain on answerable questions (gold statements sit deep in
        long sessions; see RUN_NOTES "gate projection"). The gate audits exactly
        what the reader would see."""
        top = sorted(self.items, key=lambda e: e.score, reverse=True)[:max_items]
        lines = []
        for ev in top:
            text = ev.text if len(ev.text) <= max_chars_per_item else ev.text[:max_chars_per_item] + "..."
            text = " ".join(text.split())
            lines.append(f"[{date_key(ev.date) or ev.date}] ({ev.source}, session {ev.session_id}) {text}")
        return "\n".join(lines)


class MemoryItem(BaseModel):
    """HydraDB ingest item shape (see store/hydradb.py).

    Note: the v2 server rejects string `metadata` (must be a map), so the wire
    adapter builds plain dicts; this class documents the shape."""

    id: str
    title: str
    text: str | None = None
    user_assistant_pairs: list[dict[str, str]] | None = None
    infer: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    additional_metadata: dict[str, Any] = Field(default_factory=dict)


class EvalInstance(BaseModel):
    """One LongMemEval row. `has_answer` inside sessions is dropped via Turn."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_session_ids: list[str]
    haystack_dates: list[str]
    haystack_sessions: list[list[Turn]]
    answer_session_ids: list[str] = Field(default_factory=list)

    @field_validator("answer", mode="before")
    @classmethod
    def _stringify_answer(cls, v: Any) -> str:
        # some gold answers are numbers/lists in the dataset (e.g. counting questions)
        if isinstance(v, str):
            return v
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")
