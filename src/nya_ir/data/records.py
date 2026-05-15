"""Typed records used across the experiment pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Passage:
    doc_id: str
    text: str
    title: str | None = None

    def to_json(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    text: str

    def to_json(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Qrel:
    query_id: str
    doc_id: str
    relevance: int
    iteration: str = "0"


@dataclass(frozen=True, slots=True)
class RunEntry:
    query_id: str
    doc_id: str
    rank: int
    score: float
    run_id: str
    iteration: str = "Q0"

