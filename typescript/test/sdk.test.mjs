/**
 * End-to-end SDK test: spawns the real hydra-mem Python server (MockLLM + a
 * throwaway sqlite), then exercises the whole TypeScript surface over HTTP.
 *
 * Run: npm test  (from typescript/)
 */

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import readline from "node:readline";

import { Memory } from "../dist/index.js";

const ABSTAIN = "I don't know. The chat history does not contain this information.";

let server;
let client;

before(async () => {
  // project venv python (carries openai/httpx/pydantic)
  const py = new URL("../../.venv/bin/python", import.meta.url).pathname;
  server = spawn(py, ["test/server_helper.py"], {
    cwd: new URL("..", import.meta.url).pathname,
    stdio: ["ignore", "pipe", "inherit"],
  });
  const lines = readline.createInterface({ input: server.stdout });
  const port = await new Promise((resolve, reject) => {
    lines.on("line", (line) => {
      try {
        resolve(JSON.parse(line).port);
      } catch {
        /* keep reading */
      }
    });
    server.once("exit", (code) => reject(new Error(`server exited early: ${code}`)));
    setTimeout(() => reject(new Error("server port timeout")), 15000);
  });
  client = new Memory({ baseUrl: `http://127.0.0.1:${port}` });
});

after(async () => {
  if (server) server.kill("SIGTERM");
  await once(server, "exit").catch(() => {});
});

test("add / search / get / history / update / delete roundtrip", async () => {
  const added = await client.add("hello from the TS sdk", { user_id: "alice", infer: false });
  assert.equal(added.results[0].event, "ADD");
  const id = added.results[0].id;

  const hits = await client.search("hello sdk", { filters: { user_id: "alice" } });
  assert.ok(hits.results.some((r) => r.id === id));

  const item = await client.get(id);
  assert.equal(item.memory, "hello from the TS sdk");

  const history = await client.history(id);
  assert.equal(history[0].event, "ADD");
  assert.equal(history[0].memory_id, id);

  await client.update(id, { text: "hello from the TS sdk (v2)" });
  assert.equal((await client.get(id)).memory, "hello from the TS sdk (v2)");

  assert.equal((await client.delete(id)).message, "Memory deleted successfully!");
  assert.equal(await client.get(id), null); // mem0 parity: null, not a throw
});

test("chat abstains exactly when there is no evidence", async () => {
  const answer = await client.chat("what did I eat on 3 March 1999?", { user_id: "nobody" });
  assert.equal(answer, ABSTAIN);
});

test("errors map to thrown Errors with the server message", async () => {
  await assert.rejects(() => client.search("x"), /user_id, agent_id, run_id/);
  await assert.rejects(() => client.delete("nope"), /not found/);
});

test("infer=true extracts quality-gated facts", async () => {
  const added = await client.add("my favorite city is Lisbon", { user_id: "traveller" });
  assert.ok(added.results.length > 0);
  const all = await client.get_all({ filters: { user_id: "traveller" } });
  assert.ok(all.results.every((r) => r.metadata?.kind === "fact"));
});

test("add_knowledge + weighted collections search (RAG + memory)", async () => {
  await client.add_knowledge("The hydrous engine stores bi-temporal facts", { user_id: "rag" });
  const found = await client.search("bi-temporal facts", {
    filters: { user_id: "rag" },
    collections: [{ name: "knowledge", weight: 1.0 }],
  });
  assert.ok(found.results.some((r) => r.memory.includes("bi-temporal")));

  // memory bucket alone has no knowledge -> empty
  const memoryOnly = await client.search("bi-temporal facts", { filters: { user_id: "rag" } });
  assert.equal(memoryOnly.results.length, 0);
});

test("reference_date is accepted (time-travel; mem0 OSS raises here)", async () => {
  // seeded via raw add (today); searching as-of today still finds it
  await client.add("I run a daily standup at 9am", { user_id: "alice", infer: false });
  const hits = await client.search("daily standup", {
    filters: { user_id: "alice" },
    reference_date: new Date().toISOString().slice(0, 10),
  });
  assert.ok(hits.results.length > 0);
});

test("reset wipes scopes", async () => {
  await client.add("transient", { user_id: "wipe_me", infer: false });
  assert.equal((await client.reset()).message, "Memory reset successfully!");
  const items = await client.get_all({ filters: { user_id: "wipe_me" } });
  assert.equal(items.results.length, 0);
});
