"""Shared experiment identifiers and condition dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StrategyName(str, Enum):
    """Canonical preprocessing strategy identifiers."""

    KEEP = "keep"
    NAIVE_STRIP = "naive_strip"
    SASTRAWI_CLITIC = "sastrawi_clitic"
    SENTINEL = "sentinel"
    RULE_RESOLVED = "rule_resolved"


class RetrieverName(str, Enum):
    """Canonical retriever identifiers."""

    BM25 = "bm25"
    BGE_M3 = "bge_m3"


@dataclass(frozen=True, slots=True)
class ExperimentCondition:
    """One preprocessing x retriever condition."""

    strategy: StrategyName
    retriever: RetrieverName

    @property
    def run_id(self) -> str:
        return f"{self.retriever.value}__{self.strategy.value}"
