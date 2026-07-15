"""Stable identifiers and data contracts for the vector platform.

This module is the shared boundary published by Person 2.  Downstream code
should treat repository and user identifiers as opaque strings and use the
helpers here whenever it needs the corresponding Qdrant point ID.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_VECTOR_NAME,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
    USER_PROFILES_COLLECTION_NAME,
)


@dataclass(frozen=True, slots=True)
class VectorCollectionContract:
    """Immutable description of one Qdrant collection."""

    collection_name: str
    vector_name: str | None
    vector_size: int
    distance: str
    model_name: str


REPOSITORY_COLLECTION_CONTRACT = VectorCollectionContract(
    collection_name=QDRANT_COLLECTION_NAME,
    vector_name=QDRANT_VECTOR_NAME,
    vector_size=REPOSITORY_EMBEDDING_DIM,
    distance=QDRANT_DISTANCE,
    model_name=REPOSITORY_EMBEDDING_MODEL,
)

# The existing user_profiles collection stores one unnamed vector per user.
# Keeping that choice explicit prevents consumers from guessing a vector name.
USER_PROFILE_COLLECTION_CONTRACT = VectorCollectionContract(
    collection_name=USER_PROFILES_COLLECTION_NAME,
    vector_name=None,
    vector_size=REPOSITORY_EMBEDDING_DIM,
    distance=QDRANT_DISTANCE,
    model_name=REPOSITORY_EMBEDDING_MODEL,
)


def _canonical_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    canonical = value.strip()
    if not canonical:
        raise ValueError(f"{field_name} must be a non-empty string")
    return canonical


def repository_point_id(repo_id: str) -> str:
    """Return the deterministic Qdrant point ID for an opaque repository ID."""
    canonical = _canonical_identifier(repo_id, field_name="repo_id")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"github:{canonical}"))


def user_point_id(user_id: str) -> str:
    """Return the deterministic Qdrant point ID for an opaque user ID."""
    canonical = _canonical_identifier(user_id, field_name="user_id")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"user:{canonical}"))


# Every repository point publishes these keys.  Nullable values are still
# present so retrieval and ranking consumers receive one predictable shape.
REPOSITORY_PAYLOAD_FIELD_TYPES: Mapping[str, type | tuple[type, ...]] = MappingProxyType(
    {
        "repo_id": str,
        "full_name": str,
        "html_url": (str, type(None)),
        "description": str,
        "primary_language": str,
        "languages": list,
        "topics": list,
        "star_count": int,
        "fork_count": int,
        "open_issues_count": int,
        "readme_length": int,
        "readme_chunks": int,
        "pushed_days_ago": int,
        "delta_3d": int,
        "delta_7d": int,
        "delta_30d": int,
        "mentionable_users_count": int,
        "created_at": (str, type(None)),
        "updated_at": (str, type(None)),
        "pushed_at": (str, type(None)),
        "discovery_category": (str, type(None)),
        "discovery_band": (str, type(None)),
        "category": str,
        "tags": list,
        "doc_quality": (int, float),
        "code_health": (int, float),
        "activity_score": (int, float),
        "trend_velocity": (int, float),
        "embedding_dim": int,
        "embedding_model": str,
        "embedding_version": str,
        "source_hash": str,
    }
)

REPOSITORY_PAYLOAD_REQUIRED_FIELDS = tuple(REPOSITORY_PAYLOAD_FIELD_TYPES)


def repository_payload_defaults() -> dict[str, object]:
    """Return fresh defaults for optional repository metadata fields."""
    return {
        "html_url": None,
        "description": "",
        "primary_language": "Unknown",
        "languages": [],
        "topics": [],
        "star_count": 0,
        "fork_count": 0,
        "open_issues_count": 0,
        "readme_length": 0,
        "readme_chunks": 0,
        "pushed_days_ago": 999,
        "delta_3d": 0,
        "delta_7d": 0,
        "delta_30d": 0,
        "mentionable_users_count": 0,
        "created_at": None,
        "updated_at": None,
        "pushed_at": None,
        "discovery_category": None,
        "discovery_band": None,
        "category": "Unknown",
        "tags": [],
        "doc_quality": 0.0,
        "code_health": 0.0,
        "activity_score": 0.0,
        "trend_velocity": 0.0,
        "embedding_dim": REPOSITORY_EMBEDDING_DIM,
        "embedding_model": REPOSITORY_EMBEDDING_MODEL,
        "embedding_version": REPOSITORY_EMBEDDING_VERSION,
    }
