"""Command-line entry point for the production repository corpus pipeline.

The command performs a bounded, resumable pass through discovery, enrichment,
quality filtering, Postgres persistence, and Qdrant indexing. Postgres is the
source of truth; indexing without it requires an explicit development flag.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("pipeline.acquisition")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def run_acquisition(
    token: str,
    *,
    limit: int = 150,
    batch_size: int = 15,
    workers: int = 4,
    existing_repos: set[str] | None = None,
) -> list[Any]:
    """Backward-compatible list-returning acquisition wrapper."""
    from acquisition.pipeline import run_acquisition as _run_acquisition

    return _run_acquisition(
        token,
        limit=limit,
        batch_size=batch_size,
        workers=workers,
        existing_repos=existing_repos,
    )


def filter_enriched(
    enriched: list[Any],
    *,
    min_readme_chars: int = 200,
) -> tuple[list[Any], list[tuple[Any, list[str]]]]:
    """Split enriched repositories into approved and rejected audit groups."""
    if min_readme_chars < 1:
        raise ValueError("min_readme_chars must be a positive integer")

    kept: list[Any] = []
    dropped: list[tuple[Any, list[str]]] = []
    for repository in enriched:
        payload = repository.payload
        reasons: list[str] = []
        readme_length = payload.get("readme_length", 0)
        if readme_length == 0:
            reasons.append("no README")
        elif readme_length < min_readme_chars:
            reasons.append(
                f"README too thin ({readme_length} chars < {min_readme_chars})"
            )

        if not any(
            (
                bool((payload.get("description") or "").strip()),
                bool(payload.get("languages")),
                bool(payload.get("topics")),
            )
        ):
            reasons.append("shell repo: no description, languages, or topics")

        if reasons:
            dropped.append((repository, reasons))
        else:
            kept.append(repository)
    return kept, dropped


def index_approved_repositories(
    approved: list[Any],
    *,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_collection: str | None = None,
    embedding_model: str | None = None,
) -> list[Any]:
    """Embed approved repositories and persist their vectors to Qdrant."""
    if not approved:
        return []

    from config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from embedding.qdrant_store import QdrantRepositoryStore
    from embedding.repository_embedding import RepositoryEmbeddingConfig

    embedding_config = RepositoryEmbeddingConfig(
        model_name=embedding_model
        or os.getenv("EMBEDDING_MODEL")
        or "all-MiniLM-L6-v2",
    )
    store = QdrantRepositoryStore(
        url=qdrant_url or QDRANT_URL,
        api_key=qdrant_api_key or QDRANT_API_KEY,
        collection_name=qdrant_collection or QDRANT_COLLECTION_NAME,
        vector_size=embedding_config.embedding_dim,
    )
    pipeline = RepositoryEmbeddingPipeline(config=embedding_config, store=store)
    return pipeline.index_batch(approved)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer (>= 1), got {number}"
        )
    return number


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from acquisition.config import CorpusPipelineSettings

    defaults = CorpusPipelineSettings.from_environment()
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Corpus pipeline: Discovery → Enrichment → Quality Filter → "
            "Postgres → Qdrant"
        ),
    )
    parser.add_argument("--limit", type=_positive_int, default=150)
    parser.add_argument("--batch-size", type=_positive_int, default=15)
    parser.add_argument("--workers", type=_positive_int, default=4)
    parser.add_argument("--min-readme-chars", type=_positive_int, default=200)
    parser.add_argument(
        "--corpus-target", type=_positive_int, default=defaults.target_count
    )
    parser.add_argument("--max-cycles", type=_positive_int, default=defaults.max_cycles)
    parser.add_argument(
        "--checkpoint-path", type=Path, default=defaults.checkpoint_path
    )
    parser.add_argument(
        "--index-qdrant",
        action="store_true",
        help="Deprecated compatibility flag; indexing already runs by default",
    )
    parser.add_argument("--no-index-qdrant", action="store_true")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--qdrant-collection", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--allow-qdrant-without-postgres",
        action="store_true",
        help="Development only: permit indexing when Postgres is unavailable",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one bounded corpus-ingestion invocation and return an exit code."""
    args = _parse_args(argv)
    _setup_logging(args.log_level)

    token = os.getenv("GITHUB_TOKEN")
    if not token or token == "your_github_token_here":
        logger.error("Set GITHUB_TOKEN in your environment or .env file first.")
        return 1

    from acquisition.checkpoint import CorpusCheckpoint
    from acquisition.config import CorpusPipelineSettings
    from acquisition.corpus_pipeline import CorpusPipeline
    from acquisition.pipeline import enrich_repository_ids, run_acquisition_detailed
    from database import PostgreSQLConnector

    settings = CorpusPipelineSettings(
        target_count=args.corpus_target,
        max_cycles=args.max_cycles,
        checkpoint_path=args.checkpoint_path,
    ).validated()

    def acquire(**kwargs: Any):
        return run_acquisition_detailed(token, **kwargs)

    def retry(names: list[str]):
        return enrich_repository_ids(
            token,
            names,
            batch_size=args.batch_size,
            workers=args.workers,
        )

    def index(sources: list[Any]):
        return index_approved_repositories(
            sources,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            qdrant_collection=args.qdrant_collection,
            embedding_model=args.embedding_model,
        )

    pipeline = CorpusPipeline(
        database=PostgreSQLConnector(),
        acquire=acquire,
        acquire_retries=retry,
        quality_filter=filter_enriched,
        indexer=index,
        settings=settings,
        checkpoint=CorpusCheckpoint(settings.checkpoint_path),
        allow_qdrant_without_postgres=args.allow_qdrant_without_postgres,
        indexing_enabled=not args.no_index_qdrant,
    )
    try:
        report = pipeline.run(
            limit=args.limit,
            batch_size=args.batch_size,
            workers=args.workers,
            min_readme_chars=args.min_readme_chars,
        )
    except Exception as exc:
        logger.error("Corpus pipeline failed: %s", exc, exc_info=True)
        return 1

    logger.info("Corpus run report: %s", report.as_dict())
    return 2 if report.failures and not (report.persisted or report.indexed) else 0


if __name__ == "__main__":
    sys.exit(main())
