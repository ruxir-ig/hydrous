"""hydrous CLI: download-data / ingest / answer / eval / judge / report +
add / search / get-all / import-mem0 / demo / serve / webhook-register."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any

import typer

from .answer.reader import Answerer
from .config import Settings, build_default_embedder, build_llm
from .embeddings import TfidfEmbedder, build_embedder
from .eval.judge import judge_hypotheses
from .eval.metrics import compute_metrics, format_metrics
from .eval.runner import EvalRunner
from .importer import load_mem0_export
from .ingest.extractor import FactExtractor
from .ingest.longmemeval import iter_instances, sessions_from_instance
from .llm import MockLLM
from .mem0 import Memory, _Scope
from .retrieve.pipeline import RetrievalPipeline
from .schema import ABSTAIN_STRING
from .store.base import MemoryStore
from .store.local import LocalMemoryStore

app = typer.Typer(
    help="Hydrous — bi-temporal agent memory for cross-session continuity",
    no_args_is_help=True,
)

ORACLE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
S_CLEANED_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

_ACCOUNT_MAP_PATH = Path("data/hydradb_accounts.json")


def _check_hydradb_accounts(ids: list[str], api_key: str) -> None:
    """Fail loudly if a qid's data lives on a different HydraDB account.

    Per-account collections mean `has_question` is silently False on the wrong
    account — which would trigger a fresh extraction+ingest (quota burn) instead
    of an error. The map is data/hydradb_accounts.json: qid -> api-key prefix."""
    import json

    if not _ACCOUNT_MAP_PATH.exists():
        return
    try:
        mapping = json.loads(_ACCOUNT_MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    key_prefix = api_key[:20] if api_key else ""
    for qid in ids:
        owner = mapping.get(qid)
        if owner and owner != key_prefix:
            raise typer.Exit(
                f"error: {qid} was ingested on account {owner!r} but the configured key is {key_prefix!r}. "
                f"Set HYDRA_DB_API_KEY to the owning account (see data/hydradb_accounts.json)."
            )


def _setup_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def _llm(settings: Settings, *, mock: bool):
    try:
        return build_llm(settings, force_mock=mock)
    except ValueError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e


def _build_store(
    settings: Settings, *, mock: bool, store_backend: str, store_path: Path | None = None
) -> MemoryStore:
    """Build a retrieval store, optionally projected into local HydraDB OSS."""
    if store_backend == "hydradb":
        from .store.hydradb import HydraDBStore

        if not settings.hydra_db_api_key:
            typer.secho(
                "error: --store hydradb requires HYDRA_DB_API_KEY in .env", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=2)
        return HydraDBStore(
            settings.hydra_db_api_key,
            database=settings.hydra_db_database,
            base_url=settings.hydra_db_base_url,
        )
    if store_backend == "hydradb-oss":
        from .graph import GraphBackedStore, HydraDBOSSGraph

        try:
            token = settings.resolve_hydradb_oss_token()
        except ValueError as e:
            typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from e
        graph = HydraDBOSSGraph(
            token,
            base_url=settings.hydra_db_oss_base_url,
            graph_id=settings.hydra_db_oss_graph_id,
            namespace=settings.hydra_db_oss_namespace,
            cell_id=settings.hydra_db_oss_cell_id,
            timeout=settings.request_timeout,
        )
        embedder = build_embedder("tfidf") if mock else build_default_embedder(settings)
        return GraphBackedStore.local(store_path or settings.store_path, embedder, graph)
    if store_backend != "local":
        raise typer.BadParameter(
            f"unknown store backend: {store_backend} (expected local|hydradb|hydradb-oss)"
        )
    embedder = build_embedder("tfidf") if mock else build_default_embedder(settings)
    return LocalMemoryStore(store_path or settings.store_path, embedder)


def _build_stack(settings: Settings, *, mock: bool, store: str = "local", store_path: Path | None = None):
    """Wire llm + store + extractor + retriever + answerer. --mock forces MockLLM + tfidf."""
    llm = _llm(settings, mock=mock)
    store_ = _build_store(settings, mock=mock, store_backend=store, store_path=store_path)
    extractor = FactExtractor(
        llm,
        store_,
        settings.cache_dir,
        model_name=("mock-model" if mock else settings.llm_model),
        max_concurrency=settings.max_concurrency,
        quality_min_score=settings.quality_min_score,
    )
    retriever = RetrievalPipeline(llm, store_, top_k=settings.top_k)
    answerer = Answerer(llm, retriever, use_gate=settings.use_gate)
    return llm, store_, extractor, retriever, answerer


@app.command("download-data")
def download_data(
    data_dir: Path = Path("data"),
    oracle_only: bool = typer.Option(False, "--oracle-only", help="Skip the 277MB s_cleaned file"),
) -> None:
    """Download LongMemEval datasets into data/ (resumable via curl -C -)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    targets = [(ORACLE_URL, data_dir / "longmemeval_oracle.json")]
    if not oracle_only:
        targets.append((S_CLEANED_URL, data_dir / "longmemeval_s_cleaned.json"))
    for url, dest in targets:
        typer.echo(f"downloading -> {dest}")
        subprocess.run(["curl", "-fSL", "-C", "-", "-o", str(dest), url], check=True)
    typer.echo("done.")


@app.command()
def ingest(
    dataset: Path = typer.Option(Path("data/longmemeval_s_cleaned.json"), help="Dataset JSON"),
    limit: int | None = typer.Option(None, help="Max instances to ingest"),
    qtype: list[str] = typer.Option([], "--type", help="Filter by question_type"),
    ids: list[str] = typer.Option([], "--id", help="Filter by question_id"),
    mock: bool = typer.Option(False, "--mock", help="Use MockLLM + tfidf (no API key)"),
    store_backend: str = typer.Option("local", "--store", help="local | hydradb | hydradb-oss"),
    reingest: bool = typer.Option(False, "--reingest", help="Drop stored data per instance first"),
) -> None:
    """Ingest instances (extract facts + summaries, build the memory store)."""
    settings = Settings()
    _setup_logging(settings)
    if store_backend == "hydradb" and ids:
        _check_hydradb_accounts(ids, settings.hydra_db_api_key)
    _, store, extractor, _, _ = _build_stack(settings, mock=mock, store=store_backend)

    async def main() -> int:
        n = 0
        for inst in iter_instances(
            dataset, limit=limit, ids=set(ids) or None, question_types=set(qtype) or None
        ):
            if reingest:
                await store.delete_question(inst.question_id)
            await extractor.ingest_question(
                inst.question_id, sessions_from_instance(inst), skip_if_ingested=not reingest
            )
            n += 1
            typer.echo(f"ingested {inst.question_id} [{inst.question_type}]")
        return n

    n = asyncio.run(main())
    store.close()
    typer.echo(f"done: {n} instance(s)")


@app.command()
def answer(
    question_id: str,
    dataset: Path = typer.Option(Path("data/longmemeval_s_cleaned.json"), help="Dataset JSON"),
    mock: bool = typer.Option(False, "--mock", help="Use MockLLM + tfidf (no API key)"),
    store_backend: str = typer.Option("local", "--store", help="local | hydradb | hydradb-oss"),
    show_evidence: bool = typer.Option(False, "--evidence", help="Print the evidence pack"),
) -> None:
    """Answer one dataset instance's question (ingests it first if needed)."""
    settings = Settings()
    _setup_logging(settings)
    if store_backend == "hydradb":
        _check_hydradb_accounts([question_id], settings.hydra_db_api_key)
    _, store, extractor, _, answerer = _build_stack(settings, mock=mock, store=store_backend)

    async def main() -> None:
        inst = next(iter(iter_instances(dataset, ids={question_id})), None)
        if inst is None:
            raise typer.BadParameter(f"{question_id} not found in {dataset}")
        await extractor.ingest_question(inst.question_id, sessions_from_instance(inst))
        result = await answerer.answer(inst.question_id, inst.question, inst.question_date)
        typer.echo(f"Q: {inst.question}\nA: {result.answer}\nabstained: {result.abstained}")
        if show_evidence:
            typer.echo("--- evidence ---")
            typer.echo(result.evidence.to_prompt())

    asyncio.run(main())
    store.close()


@app.command("eval")
def eval_cmd(
    dataset: Path = typer.Option(Path("data/longmemeval_s_cleaned.json"), help="Dataset JSON"),
    out: Path = typer.Option(Path("hypotheses.jsonl"), help="Hypotheses JSONL (appended, resumable)"),
    limit: int | None = typer.Option(None, help="Max instances to consider"),
    qtype: list[str] = typer.Option([], "--type", help="Filter by question_type"),
    ids: list[str] = typer.Option([], "--id", help="Filter by question_id"),
    exclude_abs: bool = typer.Option(False, "--exclude-abs", help="Skip _abs abstention items"),
    mock: bool = typer.Option(False, "--mock", help="Use MockLLM + tfidf (no API key)"),
    store_backend: str = typer.Option("local", "--store", help="local | hydradb | hydradb-oss"),
    reingest: bool = typer.Option(False, "--reingest", help="Drop stored data per instance first"),
) -> None:
    """Run the full pipeline (ingest → retrieve → answer) and write hypotheses JSONL."""
    settings = Settings()
    _setup_logging(settings)
    if store_backend == "hydradb" and ids:
        _check_hydradb_accounts(ids, settings.hydra_db_api_key)
    _, store, extractor, _, answerer = _build_stack(settings, mock=mock, store=store_backend)
    runner = EvalRunner(extractor, answerer)

    def progress(inst, result) -> None:
        typer.echo(f"{inst.question_id} [{inst.question_type}] abstained={result.abstained}")

    results = asyncio.run(
        runner.run(
            dataset,
            out,
            limit=limit,
            question_types=set(qtype) or None,
            ids=set(ids) or None,
            exclude_abs=exclude_abs,
            reingest=reingest,
            on_result=progress,
        )
    )
    store.close()
    typer.echo(f"wrote {len(results)} new hypotheses -> {out}")


@app.command()
def judge(
    hypotheses: Path = typer.Option(..., help="Hypotheses JSONL from `eval`"),
    dataset: Path = typer.Option(Path("data/longmemeval_s_cleaned.json"), help="Dataset JSON (gold answers)"),
    out: Path | None = typer.Option(None, help="Log output path (default: <hypotheses>.log)"),
    mock: bool = typer.Option(False, "--mock", help="Use MockLLM as judge (no API key)"),
) -> None:
    """Judge hypotheses with the official LongMemEval prompts."""
    settings = Settings()
    _setup_logging(settings)
    llm = _llm(settings, mock=mock)
    records = asyncio.run(
        judge_hypotheses(
            hypotheses,
            dataset,
            out,
            llm=llm,
            judge_model=("mock-model" if mock else settings.judge_model),
            max_concurrency=settings.max_concurrency,
        )
    )
    typer.echo(f"judged {len(records)} hypotheses -> {out or str(hypotheses) + '.log'}")


@app.command()
def report(
    log: Path = typer.Option(..., "--log", help="Judge .log file from `judge`"),
) -> None:
    """Print overall / per-type / abstention accuracy table from a judge log."""
    typer.echo(format_metrics(compute_metrics(log)))


# ---------------------------------------------------------------------------
# Public Memory API (add / search / get-all / import-mem0 / demo / serve)
# ---------------------------------------------------------------------------


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
    store: str = typer.Option("local", "--store", help="local | hydradb | hydradb-oss"),
    store_path: Path | None = typer.Option(None, "--store-path", help="Local sqlite path"),
    mock: bool = typer.Option(True, "--mock/--real", help="--real uses the configured LLM for infer"),
) -> None:
    """Add a memory."""
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
        None, "--as-of", help="time-travel: only facts valid on YYYY-MM-DD"
    ),
    store: str = typer.Option("local", "--store", help="local | hydradb | hydradb-oss"),
    store_path: Path | None = typer.Option(None, "--store-path", help="Local sqlite path"),
) -> None:
    """Hybrid search over a scope."""
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
    store: str = typer.Option("local", "--store", help="local | hydradb | hydradb-oss"),
    store_path: Path | None = typer.Option(None, "--store-path", help="Local sqlite path"),
) -> None:
    """List every memory in a scope."""
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
    store: str = typer.Option("local", "--store", help="local | hydradb | hydradb-oss"),
    store_path: Path | None = typer.Option(
        None, "--store-path", help="Local sqlite path (default: STORE_PATH env)"
    ),
    mock: bool = typer.Option(False, "--mock", help="Use MockLLM for --infer (offline)"),
) -> None:
    """Import a memory `get_all` export, preserving created_at -> recorded_at."""
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
    store: str = typer.Option("hydradb-oss", "--store", help="local | hydradb | hydradb-oss"),
    store_path: Path = typer.Option(Path("data/hydrous.sqlite3"), "--store-path", help="Local sqlite path"),
    real: bool = typer.Option(
        False, "--real/--mock", help="Use the configured LLM, or deterministic MockLLM by default"
    ),
) -> None:
    """Run the four Hydrous demos.

    1. "what laptop did I use BEFORE the MacBook?"  (superseded facts)
    2. as_of=2024-01-01 time-travel                  (reference_date)
    3. no evidence -> exact abstention
    4. multi-hop graph question                      (HydraDB path evidence)
    """
    qid = _Scope(user_id=user_id).question_id
    # Seed deterministic bi-temporal facts (no LLM needed for seeding).
    from .schema import Fact

    settings = Settings()
    tmp_store = _build_store(settings, mock=True, store_backend=store, store_path=store_path)
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
    if store == "hydradb-oss":
        sister = Fact(
            id="demo_sister_anna", question_id=qid, subject="user", predicate="has_sister",
            value="Anna", fact_date="2023-01-05", session_id="demo_s3", valid_from="2023-01-05",
        )
        city = Fact(
            id="demo_anna_city", question_id=qid, subject="Anna", predicate="works_in_city",
            value="London", fact_date="2023-02-10", session_id="demo_s4", valid_from="2023-02-10",
        )
        asyncio.run(tmp_store.add_facts(qid, [sister, city]))
    tmp_store.close()

    oss_graph = None
    if store == "hydradb-oss":
        from .graph import HydraDBOSSGraph

        oss_graph = HydraDBOSSGraph(
            settings.resolve_hydradb_oss_token(),
            base_url=settings.hydra_db_oss_base_url,
            graph_id=settings.hydra_db_oss_graph_id,
            namespace=settings.hydra_db_oss_namespace,
            cell_id=settings.hydra_db_oss_cell_id,
            timeout=settings.request_timeout,
        )

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
        if oss_graph is not None:
            chain = asyncio.run(oss_graph.supersession_chain(qid, "user", "uses_laptop"))
            from .graph.queries import supersession_chain as supersession_cypher

            typer.echo("   HydraDB supersession Cypher:")
            typer.echo(f"   {supersession_cypher()}")
            typer.echo(f"   HydraDB graph chain: {[f.value for f in chain]}")

        typer.echo("\n-- 2) time travel: search(as_of='2024-01-01')")
        r2 = m.search("laptop", filters={"user_id": user_id}, top_k=3, reference_date="2024-01-01")
        for r in r2["results"]:
            typer.echo(f"   [{r.get('score'):.3f}] {r.get('memory')}  (id={r.get('id')})")
        if oss_graph is not None:
            facts = asyncio.run(oss_graph.as_of(qid, "2024-01-01"))
            typer.echo(f"   HydraDB graph as-of: {[f.value for f in facts]}")

        typer.echo("\n-- 3) no evidence -> exact abstention (no hallucination)")
        answer = m.chat("what did I eat for breakfast on 3 March 1999?", filters={"user_id": user_id})
        typer.echo(f"   chat answer: {answer!r}")
        typer.echo(f"   is exact abstention string: {answer == ABSTAIN_STRING}")

        typer.echo("\n-- 4) multi-hop graph question (HydraDB path evidence in search/chat)")
        if store == "hydradb":
            # graph paths are merged into search evidence as source="graph" items
            r4 = m.search(
                "what city does my sister's partner work in?", filters={"user_id": user_id}
            )
            for r in r4["results"]:
                kind = r.get("metadata", {}).get("kind")
                typer.echo(f"   [{r.get('score'):.3f}] {kind}: {r.get('memory')}")
        elif store == "hydradb-oss":
            paths = asyncio.run(oss_graph.multi_hop(qid, "user", ["has_sister", "works_in_city"]))
            typer.echo("   Cypher:")
            typer.echo(f"   {paths[0].cypher if paths else '(no path)'}")
            for path in paths:
                typer.echo(f"   path: {path.text}")
            q = "what city does my sister work in?"
            r4 = m.search(q, filters={"user_id": user_id}, top_k=3)
            typer.echo("   search evidence (the same path is now reader input):")
            for r in r4["results"]:
                kind = r.get("metadata", {}).get("kind")
                typer.echo(f"   [{r.get('score'):.3f}] {kind}: {r.get('memory')}")
            typer.echo(f"   chat: {m.chat(q, filters={'user_id': user_id})}")
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
    if oss_graph is not None:
        oss_graph.close()
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

    Requires a PUBLIC url (tunneled `hydrous serve`) and HYDRA_DB_API_KEY."""
    with _memory("hydradb", mock=mock) as m:
        m.register_indexing_webhook(url, signing_secret=signing_secret)
    typer.echo(f"registered: {url} -> ['indexing.status_changed']")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    store: str = typer.Option("local", "--store", help="local | hydradb | hydradb-oss"),
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
    """Serve the Hydrous HTTP API (+ HydraDB indexing webhooks + playground).

    `hydrous serve --store hydradb --callback-url https://my-tunnel.example/...`
    stops ingest polling: HydraDB POSTs indexing.status_changed to this server's
    /webhooks/hydradb/indexing and the tracker wakes the store instead."""
    from .indexing import IndexingTracker
    from .server import build_server

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
            f"hydrous serving on http://{host}:{port} "
            f"(store={store}, webhooks={'on' if callback_url else 'off'}) — Ctrl+C to stop"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            typer.echo("\nstopped")
@app.command("mcp")
def mcp_cmd(
    store: str = typer.Option("local", "--store", help="local | hydradb | hydradb-oss"),
    store_path: Path | None = typer.Option(None, "--store-path", help="Local sqlite path"),
    user_id: str = typer.Option("default", "--user-id", help="Default user/project scope ID"),
    mock: bool = typer.Option(False, "--mock", help="Use MockLLM for extraction/chat"),
) -> None:
    """Run the Model Context Protocol (MCP) server over stdio for Cursor/Claude integration."""
    from .mcp import MCPServer

    with _memory(store, mock=mock, store_path=store_path) as m:
        server = MCPServer(m, default_user_id=user_id)
        server.serve_forever()


if __name__ == "__main__":
    app()
