"""Phase 1 tests for the vector-platform contract."""

import pytest

from config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_VECTOR_NAME,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    USER_PROFILES_COLLECTION_NAME,
)
from embedding.qdrant_store import QdrantRepositoryStore
from embedding.repository_embedding import RepositoryEmbeddingConfig, build_vector_payload
from embedding.vector_contract import (
    REPOSITORY_COLLECTION_CONTRACT,
    REPOSITORY_PAYLOAD_FIELD_TYPES,
    REPOSITORY_PAYLOAD_REQUIRED_FIELDS,
    USER_PROFILE_COLLECTION_CONTRACT,
    repository_payload_defaults,
    repository_point_id,
    user_point_id,
)


def test_repository_collection_contract_matches_central_config():
    assert REPOSITORY_COLLECTION_CONTRACT.collection_name == QDRANT_COLLECTION_NAME
    assert REPOSITORY_COLLECTION_CONTRACT.vector_name == QDRANT_VECTOR_NAME
    assert REPOSITORY_COLLECTION_CONTRACT.vector_size == REPOSITORY_EMBEDDING_DIM == 384
    assert REPOSITORY_COLLECTION_CONTRACT.distance == QDRANT_DISTANCE == "Cosine"
    assert REPOSITORY_COLLECTION_CONTRACT.model_name == REPOSITORY_EMBEDDING_MODEL


def test_user_collection_contract_preserves_existing_unnamed_vector():
    assert USER_PROFILE_COLLECTION_CONTRACT.collection_name == USER_PROFILES_COLLECTION_NAME
    assert USER_PROFILE_COLLECTION_CONTRACT.vector_name is None
    assert USER_PROFILE_COLLECTION_CONTRACT.vector_size == REPOSITORY_EMBEDDING_DIM
    assert USER_PROFILE_COLLECTION_CONTRACT.distance == QDRANT_DISTANCE
    assert USER_PROFILE_COLLECTION_CONTRACT.model_name == REPOSITORY_EMBEDDING_MODEL


def test_point_ids_are_deterministic_and_namespaced():
    assert repository_point_id("repo-123") == repository_point_id(" repo-123 ")
    assert user_point_id("repo-123") == user_point_id(" repo-123 ")
    assert repository_point_id("repo-123") != user_point_id("repo-123")
    assert repository_point_id("repo-123") != repository_point_id("repo-456")


@pytest.mark.parametrize("value", ["", "   "])
def test_point_ids_reject_empty_identifiers(value):
    with pytest.raises(ValueError):
        repository_point_id(value)
    with pytest.raises(ValueError):
        user_point_id(value)


def test_point_ids_reject_non_string_identifiers():
    with pytest.raises(TypeError):
        repository_point_id(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        user_point_id(None)  # type: ignore[arg-type]


def test_store_uses_the_published_repository_point_id_helper():
    assert QdrantRepositoryStore._point_id("repo-123") == repository_point_id("repo-123")


def test_repository_payload_contract_has_identity_and_embedding_fields():
    assert tuple(REPOSITORY_PAYLOAD_FIELD_TYPES) == REPOSITORY_PAYLOAD_REQUIRED_FIELDS
    assert {"repo_id", "full_name"}.issubset(REPOSITORY_PAYLOAD_REQUIRED_FIELDS)
    assert {
        "embedding_dim",
        "embedding_model",
        "embedding_version",
        "source_hash",
    }.issubset(REPOSITORY_PAYLOAD_REQUIRED_FIELDS)


def test_repository_payload_defaults_are_fresh_and_match_current_model():
    first = repository_payload_defaults()
    second = repository_payload_defaults()

    first["languages"].append("Python")  # type: ignore[union-attr]

    assert second["languages"] == []
    assert second["embedding_dim"] == 384
    assert second["embedding_model"] == REPOSITORY_EMBEDDING_MODEL


def test_current_payload_builder_publishes_the_frozen_contract():
    payload = build_vector_payload(
        {
            "full_name": "owner/repository",
            "description": "Example repository",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["machine-learning"],
            "extracted_paragraphs": ["A documented project."],
        },
        repo_id="repo-123",
        final_embedding=[0.0] * 384,
        readme_chunks=1,
        source_hash="source-hash",
        config=RepositoryEmbeddingConfig(),
    )

    assert set(payload) == set(REPOSITORY_PAYLOAD_REQUIRED_FIELDS)
    for field_name, expected_type in REPOSITORY_PAYLOAD_FIELD_TYPES.items():
        assert isinstance(payload[field_name], expected_type), field_name
