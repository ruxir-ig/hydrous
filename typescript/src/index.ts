/**
 * hydrous-ts — TypeScript SDK for the hydrous memory engine.
 *
 * Mirrors the mem0 SDK surface (add / search / get_all / get / history / update /
 * delete / delete_all / reset) plus the hydrous extensions (chat, add_knowledge,
 * reference_date time-travel) over the `hydrous serve` HTTP bridge:
 *
 * ```ts
 * import { Memory } from "hydrous-ts";
 * const m = new Memory({ baseUrl: "http://127.0.0.1:8765" });
 * await m.add("I love hiking on weekends", { user_id: "alice" });
 * const hits = await m.search("hiking", { filters: { user_id: "alice" } });
 * ```
 *
 * Dependency-free: uses the global fetch (Node >= 18, browsers, Deno, Bun).
 */

export interface EntityScope {
  user_id?: string;
  agent_id?: string;
  run_id?: string;
}

export interface MemoryItem {
  id: string;
  memory: string;
  hash?: string;
  metadata?: Record<string, unknown>;
  score?: number;
  created_at?: string;
  updated_at?: string;
  event?: "ADD" | "NOOP" | "UPDATE" | "DELETE";
  role?: string | null;
  actor_id?: string | null;
  user_id?: string;
  agent_id?: string;
  run_id?: string;
  score_details?: Record<string, unknown>;
}

export interface AddResult {
  results: MemoryItem[];
}

export interface SearchQuery {
  query: string;
  top_k?: number;
  filters?: EntityScope;
  threshold?: number;
  rerank?: boolean;
  explain?: boolean;
  reference_date?: string | null; // as-of time-travel over validity intervals
  collections?: { name: string; weight: number }[]; // hydrous: multi-corpus weighting
}

export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface HistoryRecord {
  id: string;
  memory_id: string;
  old_memory: string | null;
  new_memory: string | null;
  event: "ADD" | "UPDATE" | "DELETE";
  created_at: string;
  updated_at: string;
  is_deleted: boolean;
  actor_id: string | null;
  role: string | null;
}

export interface Messaging {
  message: string;
}

export interface MemoryOptions {
  /** hydrous server base URL (the `hydrous serve` bridge). */
  baseUrl?: string;
  /** Optional Authorization bearer token (server may be behind a proxy). */
  apiKey?: string;
  /** Custom fetch implementation (tests, proxies). */
  fetch?: typeof globalThis.fetch;
}

export interface AddOptions {
  user_id?: string;
  agent_id?: string;
  run_id?: string;
  metadata?: Record<string, unknown>;
  expiration_date?: string | null;
  /** true = quality-gated fact extraction (LLM); false = raw store, no LLM. */
  infer?: boolean;
}

export interface ImportOptions {
  infer?: boolean;
  default_user_id?: string;
}

const DEFAULT_BASE = "http://127.0.0.1:8765";

export class Memory {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(options: MemoryOptions = {}) {
    this.baseUrl = (options.baseUrl ?? DEFAULT_BASE).replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  // ------------------------------------------------------------------ io

  private async request<T>(
    method: "GET" | "POST",
    path: string,
    body?: unknown,
  ): Promise<T> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;
    const res = await this.fetchImpl(this.baseUrl + path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (res.status === 404 && path.startsWith("/get/")) {
      return null as T; // mem0 parity: get() of an unknown id is null
    }
    if (!res.ok) {
      let message = `HTTP ${res.status}`;
      try {
        const json = (await res.json()) as { error?: string };
        if (json.error) message = json.error;
      } catch {
        /* keep the status fallback */
      }
      throw new Error(message);
    }
    return (await res.json()) as T;
  }

  // -------------------------------------------------------------- surface

  /** Create memory(ies). `infer: false` stores raw messages (no LLM call). */
  add(messages: string | Message[], options: AddOptions = {}): Promise<AddResult> {
    return this.request("POST", "/add", { messages, ...options });
  }

  /** Hybrid search; results are tagged so re-`add`ing them is a NOOP. */
  search(q: string | SearchQuery, options: Omit<SearchQuery, "query"> = {}): Promise<AddResult> {
    const query = typeof q === "string" ? { query: q } : q;
    return this.request("POST", "/search", { ...query, ...options });
  }

  /** List everything in a scope (facts + raw sessions), newest first. */
  get_all(options: { filters?: EntityScope; top_k?: number } = {}): Promise<AddResult> {
    return this.request("POST", "/get_all", options);
  }

  /** One memory by id; null when unknown (mem0 parity). */
  get(memory_id: string): Promise<MemoryItem | null> {
    return this.request("GET", `/get/${encodeURIComponent(memory_id)}`);
  }

  /** Change records for a memory — for facts, the full supersession chain. */
  history(memory_id: string): Promise<HistoryRecord[]> {
    return this.request("GET", `/history/${encodeURIComponent(memory_id)}`);
  }

  /** Replace a raw memory's text / merge metadata. */
  update(memory_id: string, args: { text?: string; metadata?: Record<string, unknown>; data?: string } = {}): Promise<Messaging> {
    return this.request("POST", "/update", { memory_id, ...args });
  }

  /** Delete one memory (an explicit user-initiated hard delete). */
  delete(memory_id: string): Promise<Messaging> {
    return this.request("POST", "/delete", { memory_id });
  }

  /** Delete every memory in one scope. */
  delete_all(scope: EntityScope = {}): Promise<Messaging> {
    return this.request("POST", "/delete_all", scope);
  }

  /** Wipe every scope known to the store. */
  reset(): Promise<Messaging> {
    return this.request("POST", "/reset", {});
  }

  /** Answer a question from memory; exact abstention string when unanswerable. */
  chat(query: string, filters?: EntityScope): Promise<string> {
    return this.request<string>("POST", "/chat", { query, filters });
  }

  /** RAG corpus (hydrous extension): docs stored on the same database in the
   * scope's knowledge bucket; retrieve with search({ collections: [...] }). */
  add_knowledge(documents: string | string[] | { text: string }[], scope: EntityScope = {}): Promise<AddResult> {
    return this.request("POST", "/add_knowledge", { documents, ...scope });
  }

  /** Replay a mem0 `get_all` export, preserving created_at -> recorded_at. */
  import_mem0(items: Record<string, unknown>[], options: ImportOptions = {}): Promise<Record<string, unknown>> {
    return this.request("POST", "/import-mem0", { items, ...options });
  }
}

/** AsyncMemory alias: the hydrous server is async-backed end to end, so one
 * class mirrors both mem0 `Memory` and `AsyncMemory` surfaces. */
export { Memory as AsyncMemory };
