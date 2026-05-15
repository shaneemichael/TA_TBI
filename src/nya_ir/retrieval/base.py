"""Shared retrieval interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    doc_id: str
    score: float
    rank: int


class Retriever(Protocol):
    """Minimal searcher interface used by the pipeline."""

    def search(self, query: str, *, top_k: int = 1000) -> Sequence[RetrievalHit]:
        """Return ranked hits for one query."""

