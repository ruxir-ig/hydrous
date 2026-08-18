"""hydra-mem CLI: add / search / get-all / import-mem0 / demo.

The `import-mem0` command is the migration tool: point it at a mem0 `get_all`
export and it replays every memory into hydrous, preserving created_at as
recorded_at.

    hydra-mem import-mem0 mem0_export.json --user-id alice
    hydra-mem add "I love hiking on weekends" --user-id alice
    hydra-mem search "what does alice like?" --user-id alice
    hydra-mem demo                      # the four "mem0 cannot do this" demos
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer

from hydra_mem import Memory
from hydra_mem.api import _Scope
from hydra_mem.importer import load_mem0_export
from hydrous.embeddings import TfidfEmbedder
from hydrous.llm import MockLLM
from hydrous.schema import ABSTAIN_STRING

app = typer.Typer(
    help="hydra-mem — mem0-compatible memory API (drop-in for 'from mem0 import Memory')",
    no_args_is_help=True,
)


def _memory(
    store: str,
    *,
    mock: bool = False,
    store_path: Path | None = None,
    embedder: Any | None = None,
    **extra: Any,
) -> Memory:
    kwargs: dict = {"store": store, **extra}
    if store_path is not None:
        kwargs["store_path"] = str(store_path)
    if mock:
        kwargs["llm"] = MockLLM()
    if embedder is not None:
        kwargs["embedder"] = embedder
    return Memory(**kwargs)


@app.command("add")
def add_cmd(
    text: str,
    user_id: str = typer.Option("default", "--user-id"),
    agent_id: str | None = typer.Option(None, "--agent-id"),
    infer: bool = typer.Option(True, "--infer/--raw", help="--raw stores the message as-is (no LLM)"),
    store: str = typer.Option("local", "--store", help="local | hydradb"),
    store_path: Path | None = typer.Option(None, "--store-path", help="Local sqlite path"),
    mock: bool = typer.Option(True, "--mock/--real", help="--real uses the configured LLM for infer"),
) -> None:
    """Add a memory (mem0: `mem0 add "..." --user-id alice`)."""
    with _memory(store, mock=mock, store_path=store_path) as m:
        results = m.add(text, user_id=user_id, agent_id=agent_id, infer=infer)["results"]
    for r in results:
        typer.echo(f"{r.get('event')} {r.get('id')}: {r.get('memory')}")
    typer.echo(f"added {len(results)} memory(ies)")


@app.command("search")
def search_cmd(
    query: str,
    user_id: str = typer.Option("default", "--user-id"),
    top_k: int = typer.Option(10, "--top-k"),
    reference_date: str | None = typer.Option(
        None, "--as-of", help="time-travel: only facts valid on YYYY-MM-DD (mem0 raises here)"
    ),
    store: str = typer.Option("local", "--store", help="local | hydradb"),
    store_path: Path | None = typer.Option(None, "--store-path", help="Local sqlite path"),
) -> None:
    """Hybrid search over a scope (mem0: `mem0 search "..." --user-id alice`)."""
    with _memory(store, store_path=store_path) as m:
        results = m.search(query, filters={"user_id": user_id}, top_k=top_k, reference_date=reference_date)[
            "results"
        ]
    for r in results:
        typer.echo(f"[{r.get('score'):.3f}] {r.get('id')}: {r.get('memory')}")
    typer.echo(f"{len(results)} result(s)")


@app.command("get-all")
def get_all_cmd(
    user_id: str = typer.Option("default", "--user-id"),
    store: str = typer.Option("local", "--store", help="local | hydradb"),
    store_path: Path | None = typer.Option(None, "--store-path", help="Local sqlite path"),
) -> None:
    """List every memory in a scope (mem0: `mem0 get_all --user-id alice`)."""
    with _memory(store, store_path=store_path) as m:
        for r in m.get_all(filters={"user_id": user_id})["results"]:
            typer.echo(f"{r.get('created_at')} {r.get('id')}: {r.get('memory')}")


@app.command("import-mem0")
def import_mem0_cmd(
    export: Path,
    user_id: str = typer.Option("default", "--user-id", help="Fallback scope when an item has no user_id"),
    infer: bool = typer.Option(
        False, "--infer", help="Re-extract facts with the LLM instead of replaying raw text"
    ),
    store: str = typer.Option("local", "--store", help="local | hydradb"),
    store_path: Path | None = typer.Option(
        None, "--store-path", help="Local sqlite path (default: STORE_PATH env)"
    ),
    mock: bool = typer.Option(False, "--mock", help="Use MockLLM for --infer (offline)"),
) -> None:
    """Import a mem0 `get_all` export, preserving created_at -> recorded_at."""
    items = load_mem0_export(export)
    typer.echo(f"{export}: {len(items)} memory item(s)")
    with _memory(store, mock=mock, store_path=store_path) as m:
        stats = m.import_mem0(items, infer=infer, default_user_id=user_id)
    typer.echo(
        f"imported {stats['imported']} memory unit(s), {stats['noop']} duplicate(s), "
        f"{stats['skipped']} skipped — scopes: {', '.join(stats['scopes']) or '(none)'}"
    )


@app.command("demo")
def demo_cmd(
    user_id: str = typer.Option("demo", "--user-id"),
    store: str = typer.Option("local", "--store", help="local | hydradb"),
    store_path: Path = typer.Option(Path("data/hydrous.sqlite3"), "--store-path", help="Local sqlite path"),
    real: bool = typer.Option(False, "--real", help="Use the configured LLM for chat answers"),
) -> None:
    """Run the four positioning demos mem0 structurally cannot do.

    1. "what laptop did I use BEFORE the MacBook?"  (superseded facts)
    2. as_of=2024-01-01 time-travel                  (mem0 raises on reference_date)
    3. no evidence -> exact abstention               (mem0 has no chat at all)
    4. multi-hop graph question                      (HydraDB graph_context lanes)
    """
    qid = _Scope(user_id=user_id).question_id
    # Seed deterministic bi-temporal facts (no LLM needed for seeding).
    from hydrous.schema import Fact
    from hydrous.store.local import LocalMemoryStore

    tmp_store = LocalMemoryStore(store_path, TfidfEmbedder())
    asyncio.run(tmp_store.delete_question(qid))  # idempotent re-runs
    dell = Fact(
        id="demo_dell_xps", question_id=qid, subject="user", predicate="uses_laptop",
        value="Dell XPS 13", fact_date="2023-06-01", session_id="demo_s1", valid_from="2023-06-01",
    )
    macbook = Fact(
        id="demo_macbook_pro", question_id=qid, subject="user", predicate="uses_laptop",
        value="MacBook Pro", fact_date="2024-03-15", session_id="demo_s2", valid_from="2024-03-15",
    )
    dell.valid_to = "2024-03-15"
    dell.superseded_by = macbook.id
    asyncio.run(tmp_store.add_facts(qid, [dell, macbook]))
    asyncio.run(tmp_store.update_fact(dell))
    tmp_store.close()

    with _memory(store, mock=not real, store_path=store_path, embedder=TfidfEmbedder()) as m:
        typer.echo("\n-- 1) superseded-fact question: 'what laptop did I use before the MacBook?'")
        r1 = m.search("laptop before the MacBook", filters={"user_id": user_id}, top_k=3)
        for r in r1["results"]:
            typer.echo(f"   search: [{r.get('score'):.3f}] {r.get('memory')}  (id={r.get('id')})")
        before_answer = m.chat(
            "what laptop did I use before the MacBook?", filters={"user_id": user_id}
        )
        typer.echo(f"   chat:  {before_answer}")
        if not real:
            typer.echo(
                "   (mock reader is word-overlap based; with a real LLM this answers "
                "from the superseded fact: 'Dell XPS 13' — see demo 2 for the retrievable view)"
            )

        typer.echo("\n-- 2) time travel: search(as_of='2024-01-01')  (mem0 OSS raises ValueError)")
        r2 = m.search("laptop", filters={"user_id": user_id}, top_k=3, reference_date="2024-01-01")
        for r in r2["results"]:
            typer.echo(f"   [{r.get('score'):.3f}] {r.get('memory')}  (id={r.get('id')})")

        typer.echo("\n-- 3) no evidence -> exact abstention (no hallucination)")
        answer = m.chat("what did I eat for breakfast on 3 March 1999?", filters={"user_id": user_id})
        typer.echo(f"   chat answer: {answer!r}")
        typer.echo(f"   is exact abstention string: {answer == ABSTAIN_STRING}")

        typer.echo("\n-- 4) multi-hop graph question (HydraDB graph_context lanes)")
        if store == "hydradb":
            # graph paths are merged into search evidence as source="graph" items
            r4 = m.search(
                "what city does my sister's partner work in?", filters={"user_id": user_id}
            )
            for r in r4["results"]:
                kind = r.get("metadata", {}).get("kind")
                typer.echo(f"   [{r.get('score'):.3f}] {kind}: {r.get('memory')}")
        else:
            graph_scope = _Scope(user_id=f"{user_id}_graph").question_id
            graph_store = LocalMemoryStore(store_path, TfidfEmbedder())
            asyncio.run(graph_store.delete_question(graph_scope))
            anna = Fact(
                id="g_sister_anna", question_id=graph_scope, subject="user", predicate="has_sister",
                value="Anna", fact_date="2023-01-05", session_id="g1", valid_from="2023-01-05",
            )
            london = Fact(
                id="g_anna_works", question_id=graph_scope, subject="Anna", predicate="works_in_city",
                value="London", fact_date="2023-02-10", session_id="g2", valid_from="2023-02-10",
            )
            asyncio.run(graph_store.add_facts(graph_scope, [anna, london]))
            graph_store.close()
            q = "what city does my sister work in?"
            r4 = m.search(q, filters={"user_id": f"{user_id}_graph"}, top_k=3)
            for r in r4["results"]:
                typer.echo(f"   [{r.get('score'):.3f}] {r.get('memory')}  (id={r.get('id')})")
            typer.echo(
                "   local store: multi-hop needs an entity chain; the reader connects "
                "'sister=Anna' + 'Anna works in London' (full graph lanes need --store hydradb)"
            )
            typer.echo(
                f"   chat: {m.chat(q, filters={'user_id': f'{user_id}_graph'})}"
            )
    typer.echo(f"\ndemo done (scoped to {qid!r})")


@app.command("webhook-register")
def webhook_register_cmd(
    url: str,
    signing_secret: str | None = typer.Option(
        None, "--signing-secret", help="Secret HydraDB signs payloads with (>=32 chars)"
    ),
    mock: bool = typer.Option(False, "--mock", help="Dry-run against MockLLM (no network)"),
) -> None:
    """Register the HydraDB `indexing.status_changed` webhook for the configured key.

    Requires a PUBLIC url (tunneled `hydra-mem serve`) and HYDRA_DB_API_KEY."""
    with _memory("hydradb", mock=mock) as m:
        m.register_indexing_webhook(url, signing_secret=signing_secret)
    typer.echo(f"registered: {url} -> ['indexing.status_changed']")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    store: str = typer.Option("local", "--store", help="local | hydradb"),
    store_path: Path | None = typer.Option(None, "--store-path", help="Local sqlite path"),
    mock: bool = typer.Option(False, "--mock", help="Use MockLLM for extraction/chat"),
    callback_url: str | None = typer.Option(
        None,
        "--callback-url",
        help="Public callback URL to register for HydraDB indexing.status_changed webhooks",
    ),
    signing_secret: str | None = typer.Option(
        None, "--signing-secret", help="Webhook signing secret (>=32 chars)"
    ),
) -> None:
    """Serve the mem0-compatible API over HTTP (+ HydraDB indexing webhooks).

    `hydra-mem serve --store hydradb --callback-url https://my-tunnel.example/...`
    stops ingest polling: HydraDB POSTs indexing.status_changed to this server's
    /webhooks/hydradb/indexing and the tracker wakes the store instead."""
    from hydra_mem.indexing import IndexingTracker
    from hydra_mem.server import build_server
    from hydrous.config import Settings

    tracker = IndexingTracker()
    with _memory(store, mock=mock, store_path=store_path, webhook_tracker=tracker) as m:
        database = Settings().hydra_db_database
        if callback_url:
            try:
                m.register_indexing_webhook(callback_url, signing_secret=signing_secret)
                typer.echo(f"webhook registered: {callback_url}")
            except RuntimeError as e:
                typer.secho(f"warning: {e}", fg=typer.colors.YELLOW, err=True)
        server = build_server(m, host=host, port=port, tracker=tracker, database=database)
        typer.echo(
            f"hydra-mem serving on http://{host}:{port} "
            f"(store={store}, webhooks={'on' if callback_url else 'off'}) — Ctrl+C to stop"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            typer.echo("\nstopped")
        finally:
            server.server_close()


if __name__ == "__main__":
    app()
