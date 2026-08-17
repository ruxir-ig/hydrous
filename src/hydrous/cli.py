"""hydrous CLI: download-data / ingest / answer / eval / judge / report."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

import typer

from .answer.reader import Answerer
from .config import Settings, build_default_embedder, build_llm
from .embeddings import build_embedder
from .eval.judge import judge_hypotheses
from .eval.metrics import compute_metrics, format_metrics
from .eval.runner import EvalRunner
from .ingest.extractor import FactExtractor
from .ingest.longmemeval import iter_instances, sessions_from_instance
from .retrieve.pipeline import RetrievalPipeline
from .store.base import MemoryStore
from .store.local import LocalMemoryStore

app = typer.Typer(
    help="Hydrous — agent memory layer for cross-session continuity", no_args_is_help=True
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
    """local = SQLite (+ embedder); hydradb = cloud adapter (needs HYDRA_DB_API_KEY)."""
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
    if store_backend != "local":
        raise typer.BadParameter(f"unknown store backend: {store_backend} (expected local|hydradb)")
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
    store_backend: str = typer.Option("local", "--store", help="local | hydradb"),
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
    store_backend: str = typer.Option("local", "--store", help="local | hydradb"),
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
    store_backend: str = typer.Option("local", "--store", help="local | hydradb"),
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
