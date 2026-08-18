# hydrous-ts — TypeScript SDK

Mem0-compatible memory API over the [`hydra-mem serve`](../src/hydra_mem/server.py) HTTP
bridge: `add` / `search` / `get_all` / `get` / `history` / `update` / `delete` /
`delete_all` / `reset`, plus the hydrous extensions (`chat`, `add_knowledge`
multi-corpus, `reference_date` time-travel).

Dependency-free — uses the global `fetch` (Node ≥ 18, browsers, Deno, Bun).

```ts
import { Memory } from "hydrous-ts";

const m = new Memory({ baseUrl: "http://127.0.0.1:8765" }); // `hydra-mem serve`

await m.add("I am planning to go to the market tomorrow", { user_id: "alice" });
const hits = await m.search("what is alice planning to do tomorrow?", {
  filters: { user_id: "alice" }, top_k: 3,
});
// hits.results: [{ id, memory, score, ... }]

await m.get(memory_id);          // item | null (mem0 parity)
await m.history(memory_id);      // full supersession chain (mem0 field names)
await m.chat("what laptop did I use before the MacBook?", { user_id: "alice" });

// RAG + memory in one database (hydrous extension):
await m.add_knowledge("The hydrous engine stores bi-temporal facts", { user_id: "alice" });
await m.search("bi-temporal", {
  filters: { user_id: "alice" },
  collections: [{ name: "knowledge", weight: 0.5 }],
});
```

## Running the server

```bash
cd ..   # repo root
.venv/bin/hydra-mem serve --mock --port 8765          # MockLLM, offline
.venv/bin/hydra-mem serve --port 8765                 # real configured LLM
.venv/bin/hydra-mem serve --store hydradb --callback-url https://your-tunnel.example/webhooks/hydradb/indexing
```

## Develop / test

```bash
npm run build   # tsc -> dist/
npm test        # spawns the real Python server (MockLLM) and runs the E2E suite
```

Tests (`test/sdk.test.mjs`, node:test, no framework) cover the full surface:
add/search/get/history/update/delete round-trip, exact abstention, error mapping,
quality-gated `infer=true`, `add_knowledge` + weighted collections, time-travel,
and `reset`.
