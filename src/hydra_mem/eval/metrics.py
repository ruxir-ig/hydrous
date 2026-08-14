"""Accuracy report from a judge .log: overall, per question_type, abstention subset."""

from __future__ import annotations

import json
from pathlib import Path


def compute_metrics(log_path: str | Path) -> dict:
    records = [
        json.loads(line)
        for line in Path(log_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    judged = [r for r in records if r.get("judgment") is not None]

    def acc(rows: list[dict]) -> float | None:
        return (sum(bool(r["judgment"]) for r in rows) / len(rows)) if rows else None

    by_type: dict[str, list[dict]] = {}
    for r in judged:
        by_type.setdefault(r.get("question_type") or "unknown", []).append(r)

    abs_rows = [r for r in judged if r["question_id"].endswith("_abs")]
    non_abs = [r for r in judged if not r["question_id"].endswith("_abs")]
    return {
        "n": len(judged),
        "overall": acc(judged),
        "by_type": {t: {"n": len(rows), "acc": acc(rows)} for t, rows in sorted(by_type.items())},
        "abstention": {"n": len(abs_rows), "acc": acc(abs_rows)},
        "non_abstention": {"n": len(non_abs), "acc": acc(non_abs)},
    }


def format_metrics(metrics: dict) -> str:
    def pct(x: float | None) -> str:
        return f"{x * 100:5.1f}%" if x is not None else "  n/a"

    lines = [
        f"{'category':<28} {'n':>4} {'acc':>7}",
        "-" * 42,
        f"{'OVERALL':<28} {metrics['n']:>4} {pct(metrics['overall']):>7}",
        f"{'abstention (_abs)':<28} {metrics['abstention']['n']:>4} {pct(metrics['abstention']['acc']):>7}",
        (
            f"{'non-abstention':<28} {metrics['non_abstention']['n']:>4} "
            f"{pct(metrics['non_abstention']['acc']):>7}"
        ),
        "-" * 42,
    ]
    for qtype, row in metrics["by_type"].items():
        lines.append(f"{qtype:<28} {row['n']:>4} {pct(row['acc']):>7}")
    return "\n".join(lines)
