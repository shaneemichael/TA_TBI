"""Pyserini BM25 adapter."""

from __future__ import annotations

from pathlib import Path

from nya_ir.exceptions import OptionalDependencyError
from nya_ir.retrieval.base import RetrievalHit


class PyseriniBM25Searcher:
    """Thin wrapper around Pyserini's LuceneSearcher."""

    def __init__(self, index_dir: str | Path) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("pyserini", "bm25") from exc
        self._searcher = LuceneSearcher(str(index_dir))

    def set_bm25(self, *, k1: float = 0.9, b: float = 0.4) -> None:
        self._searcher.set_bm25(k1=k1, b=b)

    def search(self, query: str, *, top_k: int = 1000) -> list[RetrievalHit]:
        hits = self._searcher.search(query, k=top_k)
        return [
            RetrievalHit(doc_id=str(hit.docid), score=float(hit.score), rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]


def build_pyserini_index(
    *,
    collection_dir: str | Path,
    index_dir: str | Path,
    generator: str = "DefaultLuceneDocumentGenerator",
    threads: int = 8,
) -> list[str]:
    """Return the pyserini command needed to build an index.

    The function does not execute Java/Pyserini. Keeping command construction separate
    makes scripts easier to dry-run and log before expensive indexing.
    """

    return [
        "python",
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection_dir),
        "--index",
        str(index_dir),
        "--generator",
        generator,
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

