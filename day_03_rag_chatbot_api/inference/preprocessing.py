from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CleanedQuery:
    text: str


def clean_query(query: str) -> CleanedQuery:
    normalized = " ".join(query.strip().split())
    if len(normalized) < 3:
        raise ValueError("Query is too short.")
    if len(normalized) > 2000:
        raise ValueError("Query is too long.")
    return CleanedQuery(text=normalized)
