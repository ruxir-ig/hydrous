"""hydra-mem: mem0-compatible memory API backed by the hydrous engine.

`from hydra_mem import Memory` is a drop-in for `from mem0 import Memory` — same
method surface (add / search / get_all / update / delete / delete_all / reset),
same result shapes, plus the temporal machinery mem0's OSS structurally lacks:
bi-temporal supersession, `reference_date` time-travel (mem0 raises on it),
exact abstention instead of hallucination, and HydraDB graph-context lanes.

Contract tests live in tests/test_mem0_compat.py and are lifted from mem0's own
documented examples and signatures (mem0 v1.1+).
"""

from .api import AsyncMemory, Memory

__version__ = "0.1.0"

__all__ = ["Memory", "AsyncMemory", "__version__"]
