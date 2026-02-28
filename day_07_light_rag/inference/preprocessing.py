"""Query preprocessing and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CleanedQuery:
    """Immutable cleaned query object."""
    text: str


def clean_query(query: str) -> CleanedQuery:
    """Normalize, validate, and return a cleaned query."""
    # Collapse whitespace
    normalized = " ".join(query.strip().split())

    if len(normalized) < 3:
        raise ValueError("Query must be at least 3 characters long.")
    if len(normalized) > 2000:
        raise ValueError("Query must be at most 2 000 characters long.")

    return CleanedQuery(text=normalized)
