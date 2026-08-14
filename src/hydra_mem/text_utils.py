"""Shared text helpers: tokenization, tolerant date parsing, word overlap."""

from __future__ import annotations

import re
from datetime import datetime

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DATE_RE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")

STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have how i if in into is it its "
    "me my of on or so that the their them they this to was what when which who with you your "
    "did do does were was will would could should about tell know".split()
)

_DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def content_words(text: str) -> set[str]:
    """Tokens minus stopwords and very short tokens."""
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2}


def word_overlap(a: str, b: str) -> float:
    """Fraction of a's content words that appear in b (0.0 if a has none)."""
    ca = content_words(a)
    if not ca:
        return 0.0
    cb = content_words(b)
    return len(ca & cb) / len(ca)


def parse_date(value: object) -> datetime | None:
    """Tolerantly parse LongMemEval-style dates like '2023/05/20 (Sat) 16:35'."""
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    m = _DATE_RE.search(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        t = _TIME_RE.search(s)
        hh, mm = (int(t.group(1)), int(t.group(2))) if t else (0, 0)
        try:
            return datetime(y, mo, d, hh, mm)
        except ValueError:
            return None
    return None


def date_key(value: object) -> str:
    """Normalized 'YYYY-MM-DD HH:MM' (or 'YYYY-MM-DD') string; '' if unparseable."""
    dt = parse_date(value)
    if dt is None:
        return str(value).strip() if value else ""
    if dt.hour or dt.minute:
        return dt.strftime("%Y-%m-%d %H:%M")
    return dt.strftime("%Y-%m-%d")


def compare_dates(a: object, b: object) -> int:
    """-1/0/1 comparison; parsed datetimes when possible, else normalized-string compare."""
    da, db = parse_date(a), parse_date(b)
    if da is not None and db is not None:
        return (da > db) - (da < db)
    ka, kb = date_key(a), date_key(b)
    return (ka > kb) - (ka < kb)
