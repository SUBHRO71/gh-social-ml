"""Failure-path tests for production corpus acquisition orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from acquisition.checkpoint import CorpusCheckpoint
from acquisition.config import CorpusPipelineSettings
from acquisition.corpus_pipeline import CorpusPipeline
from acquisition.identity import deduplicate_candidates, normalize_repository_name
from acquisition.models import AcquisitionFailure, AcquisitionRunResult
from acquisition.pipeline import enrich_repositories
from acquisition.repository_enricher import RepositoryEnricher
from database.connector import RepositoryUpsertResult


def _source(name: str):
    return SimpleNamespace(repo_id=name, payload={"id": name})


class _Database:
    enabled = True

    def __init__(self, outcome: RepositoryUpsertResult | None = None) -> None:
        self.count = 0
        self.outcome = outcome or RepositoryUpsertResult()

    def verify_connection(self) -> bool:
        return True

    def init_db(self) -> None:
        pass

    def get_repo_count(self) -> int:
        return self.count

    def get_existing_repository_names(self) -> set[str]:
        return set()

    def upsert_repositories_detailed(self, sources):
        self.count += len(self.outcome.succeeded)
        return self.outcome

    def get_repositories_by_full_names(self, names):
        return [{"id": name, "full_name": name} for name in names]


def _settings(path: Path, *, target: int = 50_000) -> CorpusPipelineSettings:
    return CorpusPipelineSettings(
        target_count=target,
        max_cycles=1,
        checkpoint_path=path,
    )


def test_identity_normalization_and_case_insensitive_deduplication():
    assert (
        normalize_repository_name(" https://github.com/OpenAI/Codex/ ")
        == "OpenAI/Codex"
    )
    unique, removed = deduplicate_candidates(
        ["OpenAI/Codex", {"full_name": "openai/codex"}, "invalid"]
    )
    assert unique == ["OpenAI/Codex"]
    assert removed == 2


def test_transient_readme_failure_is_reported_for_retry(monkeypatch):
    def return_warning(_self, batch):
        name = batch[0]
        return [SimpleNamespace(repo_id=name, warnings=["README fetch failed"])]

    monkeypatch.setattr(RepositoryEnricher, "get_repositories_batch", return_warning)
    result = enrich_repositories("test-token", ["owner/repo"], batch_size=1, workers=1)

    assert result.repositories == []
    assert len(result.failures) == 1
    assert result.failures[0].stage == "readme_enrichment"


def test_pipeline_indexes_only_successfully_persisted_subset(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    database = _Database(
        RepositoryUpsertResult(
            succeeded=["owner/good"],
            failed={"owner/bad": "constraint failure"},
        )
    )
    indexed = []
    pipeline = CorpusPipeline(
        database=database,
        acquire=lambda **_: AcquisitionRunResult(
            repositories=[_source("owner/good"), _source("owner/bad")],
            discovered_count=2,
        ),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: indexed.extend(values) or values,
        settings=_settings(path),
        checkpoint=checkpoint,
    )

    report = pipeline.run(limit=2, batch_size=1, workers=1, min_readme_chars=1)

    assert [item.repo_id for item in indexed] == ["owner/good"]
    assert report.persisted == 1
    assert report.indexed == 1
    assert checkpoint.pending_persistence == ["owner/bad"]
    assert checkpoint.pending_index == []


def test_enrichment_failure_is_checkpointed_for_identity_only_retry(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    pipeline = CorpusPipeline(
        database=_Database(),
        acquire=lambda **_: AcquisitionRunResult(
            failures=[
                AcquisitionFailure(
                    "owner/retry", "readme_enrichment", "temporary timeout"
                )
            ],
            discovered_count=1,
        ),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: values,
        settings=_settings(path),
        checkpoint=checkpoint,
    )

    report = pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert checkpoint.pending_persistence == ["owner/retry"]
    assert "owner/retry" in report.failures
    assert "temporary timeout" in path.read_text(encoding="utf-8")


def test_pending_index_is_resumed_before_new_discovery(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    checkpoint.add_pending_index(["owner/repo"])
    checkpoint.save()
    database = _Database()
    database.count = 1
    indexed = []
    pipeline = CorpusPipeline(
        database=database,
        acquire=lambda **_: pytest.fail("discovery should not run after target"),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: indexed.extend(values) or values,
        settings=_settings(path, target=1),
        checkpoint=CorpusCheckpoint(path),
    )

    report = pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert indexed == [{"id": "owner/repo", "full_name": "owner/repo"}]
    assert report.resumed_indexing == 1
    assert pipeline.checkpoint.pending_index == []


def test_pipeline_refuses_implicit_qdrant_only_production_run(tmp_path):
    pipeline = CorpusPipeline(
        database=SimpleNamespace(enabled=False),
        acquire=lambda **_: [],
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: values,
        settings=_settings(tmp_path / "checkpoint.json"),
    )

    with pytest.raises(RuntimeError, match="requires a verified Postgres"):
        pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)


def test_checkpoint_round_trip_is_atomic_and_contains_only_retry_state(tmp_path):
    path = tmp_path / "nested" / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    checkpoint.add_pending_persistence(["Owner/Repo", "owner/repo"])
    checkpoint.record_failure("Owner/Repo", "persistence", "failed")
    checkpoint.save()

    loaded = CorpusCheckpoint(path)
    assert loaded.pending_persistence == ["Owner/Repo"]
    assert "readme" not in path.read_text(encoding="utf-8").lower()
