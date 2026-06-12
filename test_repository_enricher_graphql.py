"""Tests for RepositoryEnricher GraphQL integration."""

from __future__ import annotations

from typing import Any
from acquisition.repository_enricher import RepositoryEnricher
from acquisition.github_graphql_client import GitHubGraphQLClient


class DummyGraphQLClient(GitHubGraphQLClient):
    def __init__(self):
        super().__init__(token="dummy")
        self.called = False
        self.should_fail = False

    def get_repository(self, owner: str, name: str) -> dict[str, Any] | None:
        self.called = True
        if self.should_fail:
            raise Exception("GraphQL error")
            
        return {
            "nameWithOwner": f"{owner}/{name}",
            "name": name,
            "description": "A graphql repo",
            "url": f"https://github.com/{owner}/{name}",
            "stargazerCount": 200,
            "forkCount": 20,
            "languages": {
                "edges": [{"size": 1000, "node": {"name": "Python"}}]
            },
            "repositoryTopics": {
                "nodes": [{"topic": {"name": "graphql"}}]
            },
            "readme1": {"text": "# Title\n\nThis is a GraphQL Readme paragraph."},
            "watchers": {"totalCount": 50},
            "issues": {"totalCount": 10},
            "owner": {"login": owner},
        }

    def get_repositories_batch(self, repos: list[tuple[str, str]]) -> dict[str, dict[str, Any]]:
        results = {}
        for owner, name in repos:
            results[f"{owner}/{name}"] = self.get_repository(owner, name)
        return results


def test_enricher_uses_graphql():
    gql_client = DummyGraphQLClient()
    
    enricher = RepositoryEnricher(gql_client)
    result = enricher.enrich("test/repo")
    
    assert result is not None
    assert gql_client.called is True
    assert result.payload["star_count"] == 200
    assert result.payload["topics"] == ["graphql"]
    assert result.payload["primary_language"] == "Python"
    assert result.payload["readme_length"] > 0


def test_enricher_returns_none_on_graphql_error():
    gql_client = DummyGraphQLClient()
    gql_client.should_fail = True
    
    enricher = RepositoryEnricher(gql_client)
    result = enricher.enrich("test/repo")
    
    assert result is None
    assert gql_client.called is True


def test_enricher_batch_processing():
    gql_client = DummyGraphQLClient()
    
    enricher = RepositoryEnricher(gql_client)
    repos = [{"full_name": "test/repo1"}, "test/repo2"]
    
    results = enricher.get_repositories_batch(repos)
    assert len(results) == 2
    assert results[0].payload["id"] == "test/repo1"
    assert results[1].payload["id"] == "test/repo2"


def test_contributor_extraction_behavior():
    gql_client = DummyGraphQLClient()
    enricher = RepositoryEnricher(gql_client)
    result = enricher.enrich("test/repo")
    assert result is not None
    # mentionable_users_count should fall back to 1 (owner present) since contributors is empty list
    assert result.payload["mentionable_users_count"] == 1


def test_partial_graphql_error_logging(caplog):
    import logging
    from unittest.mock import MagicMock, patch
    
    client = GitHubGraphQLClient(token="dummy")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {"repository": {"name": "test-repo"}},
        "errors": [{"message": "Some field failed to fetch"}]
    }
    
    with patch.object(client.session, "post", return_value=mock_response):
        with caplog.at_level(logging.WARNING):
            res = client.execute("query { dummy }")
            assert res is not None
            assert res.get("data") == {"repository": {"name": "test-repo"}}
            assert any("GitHub GraphQL returned partial errors" in rec.message for rec in caplog.records)


def test_readme_logging_behavior(caplog):
    import logging
    from unittest.mock import MagicMock, patch
    
    client = GitHubGraphQLClient(token="dummy")
    
    # 1. Test readme not found (repository or blobs are null)
    mock_response_null_repo = MagicMock()
    mock_response_null_repo.status_code = 200
    mock_response_null_repo.json.return_value = {
        "data": {"repository": None}
    }
    
    with patch.object(client.session, "post", return_value=mock_response_null_repo):
        with caplog.at_level(logging.INFO):
            readme = client.get_readme("owner", "name")
            assert readme == ""
            assert any("README not found" in rec.message for rec in caplog.records)

    caplog.clear()

    # 2. Test readme fetch failed (exception occurs)
    with patch.object(client.session, "post", side_effect=Exception("Connection reset")):
        with caplog.at_level(logging.WARNING):
            readme = client.get_readme("owner", "name")
            assert readme == ""
            assert any("README fetch failed" in rec.message for rec in caplog.records)
