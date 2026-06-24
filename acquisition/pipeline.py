"""Acquisition pipeline logic for discovering and enriching GitHub repositories."""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger("pipeline.acquisition")


def run_acquisition(
    token: str,
    *,
    limit: int = 150,
    batch_size: int = 15,
    workers: int = 4,
    existing_repos: set[str] | None = None,
) -> list[Any]:
    """
    Discover and enrich GitHub repositories via GraphQL only.

    Returns a list of EnrichmentResult objects. Each carries:
      .repo_id          — "owner/repo"
      .payload          — Osiris-compatible dict (star_count, language, topics, …)
      .raw_repository   — raw GraphQL response fields
      .readme           — ReadmeDocument (clean_text, extracted_paragraphs, …)
      .topics           — list[str]
      .languages        — dict[str, int]  (language → bytes)
    """
    from acquisition.github_graphql_client import GitHubGraphQLClient
    from acquisition.github_discovery import GitHubDiscoveryEngine, DiscoveryConfig
    from acquisition.repository_enricher import RepositoryEnricher

    client   = GitHubGraphQLClient(token=token)
    # Fetch a larger buffer of candidate repositories to account for filtering duplicates
    discovery_limit = limit + 50 if existing_repos else limit + 20
    config   = DiscoveryConfig(total_limit=discovery_limit)
    discovery = GitHubDiscoveryEngine(client, config=config)
    enricher  = RepositoryEnricher(graphql_client=client)

    # ── Step 1: Discovery ─────────────────────────────────────────────────────
    logger.info("Discovering repositories …")
    discovered = discovery.discover(limit=discovery_limit)
    logger.info("Discovered %d candidate repos", len(discovered))

    if existing_repos:
        new_discovered = []
        for r in discovered:
            full_name = r if isinstance(r, str) else r.get("full_name", "")
            if full_name not in existing_repos:
                new_discovered.append(r)
        logger.info(
            "Filtered out %d already existing repos from candidates. %d new candidates remain.",
            len(discovered) - len(new_discovered),
            len(new_discovered),
        )
        discovered = new_discovered

    # ── Step 2: Concurrent enrichment ─────────────────────────────────────────
    targets = discovered[:limit]
    logger.info(
        "Enriching %d repos with %d concurrent worker(s) …",
        len(targets), workers,
    )
    enriched: list = []

    # Each worker thread gets its own GitHubGraphQLClient (and requests.Session)
    # so concurrent threads never race on a shared Session object.
    _thread_local = threading.local()

    def _get_thread_enricher() -> "RepositoryEnricher":
        if not hasattr(_thread_local, "enricher"):
            from acquisition.github_graphql_client import GitHubGraphQLClient
            from acquisition.repository_enricher import RepositoryEnricher
            _thread_local.enricher = RepositoryEnricher(
                graphql_client=GitHubGraphQLClient(token=token)
            )
        return _thread_local.enricher

    def _enrich_one(repo: Any) -> Any:
        return _get_thread_enricher().enrich(repo)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all enrich() calls concurrently; each is an independent GraphQL
        # round-trip so workers spend their time waiting on network, not the CPU.
        futures = {
            executor.submit(_enrich_one, repo): repo
            for repo in targets
        }
        for future in as_completed(futures):
            repo = futures[future]
            full_name = repo if isinstance(repo, str) else repo.get("full_name", "")
            try:
                result = future.result()
                if result:
                    enriched.append(result)
                    logger.info("  ✓  %-44s (total enriched: %d)", full_name, len(enriched))
                else:
                    logger.warning("  ✗  %s: enricher returned None", full_name)
            except Exception as exc:
                logger.warning("  ✗  %s: %s", full_name, exc)

    logger.info("Acquisition complete — %d / %d repos enriched", len(enriched), limit)
    return enriched

