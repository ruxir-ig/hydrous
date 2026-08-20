"""mem0 export loading for `hydrous import-mem0`.

A mem0 `get_all` export looks like:

    {"results": [
        {"id": "...", "memory": "Likes hiking on weekends", "hash": "...",
         "metadata": {...}, "score": 0.9, "created_at": "2025-01-15T10:30:00Z",
         "updated_at": "...", "user_id": "alice", ...},
        ...
    ]}

The replay itself lives in `AsyncMemory.import_mem0` (created_at -> recorded_at);
this module only loads + validates the file shape.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_mem0_export(path: str | Path) -> list[dict]:
    """Read a mem0 export file and return the list of memory items."""
    raw_path = Path(path)
    try:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{raw_path} is not valid JSON: {e}") from e
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise ValueError(
            f"{raw_path}: expected a mem0 get_all export "
            '(JSON object with a "results" list, or a bare list of memories)'
        )
    return [r for r in results if isinstance(r, dict)]
