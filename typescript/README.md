# hydrous-ts

TypeScript client for Hydrous over the [`hydrous serve`](../src/hydrous/server.py)
HTTP bridge.

Methods: `add` / `search` / `get_all` / `get` / `history` / `update` / `delete` /
`delete_all` / `reset`, plus `chat`, `add_knowledge`, `reference_date`, and
weighted `collections`.

Dependency-free — uses global `fetch` (Node ≥ 18, browsers, Deno, Bun).

```ts
import { Memory } from "hydrous-ts";

const m = new Memory({ baseUrl: "http://127.0.0.1:8765" });

await m.add("I am planning to go to the market tomorrow", { user_id: "alice" });
const hits = await m.search("what is alice planning to do tomorrow?", {
  filters: { user_id: "alice" }, top_k: 3,
});

await m.get(memory_id);
await m.history(memory_id);
await m.chat("what laptop did I use before the MacBook?", { user_id: "alice" });

await m.add_knowledge("The hydrous engine stores bi-temporal facts", { user_id: "alice" });
await m.search("bi-temporal", {
  filters: { user_id: "alice" },
  collections: [{ name: "knowledge", weight: 0.5 }],
});
```

## Running the server

```bash
cd ..   # repo root
.venv/bin/hydrous serve --mock --port 8765
.venv/bin/hydrous serve --store hydradb-oss --mock --port 8765
```

## Develop / test

```bash
npm run build
npm test        # spawns the real Python server (MockLLM) and runs E2E
```

Tests cover add/search/get/history/update/delete, exact abstention, error
mapping, quality-gated `infer=true`, `add_knowledge` + weighted collections,
time-travel, and `reset`.
