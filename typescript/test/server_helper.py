"""Test fixture: spawn a hydrous HTTP server with MockLLM on an ephemeral port.

Prints {"port": N} on stdout once serving; the TS test parses it and SIGTERMs
this process when done. Run from the repo root (src/ is put on sys.path).
"""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hydrous import Memory  # noqa: E402
from hydrous.embeddings import TfidfEmbedder  # noqa: E402
from hydrous.indexing import IndexingTracker  # noqa: E402
from hydrous.llm import MockLLM  # noqa: E402
from hydrous.server import build_server  # noqa: E402


def main() -> None:
    tmp = tempfile.mkdtemp()
    memory = Memory(
        store="local",
        store_path=os.path.join(tmp, "ts.sqlite3"),
        llm=MockLLM(),
        embedder=TfidfEmbedder(),
    )
    server = build_server(memory, host="127.0.0.1", port=0, tracker=IndexingTracker(), database="hydrous")
    print(json.dumps({"port": server.server_address[1]}), flush=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def stop(*_):
        server.shutdown()
        server.server_close()
        memory.close()
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
