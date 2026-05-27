"""Pyserini BM25 adapter."""

from __future__ import annotations

from pathlib import Path

from nya_ir.exceptions import OptionalDependencyError
from nya_ir.retrieval.base import RetrievalHit


class PyseriniBM25Searcher:
    """Thin wrapper around Pyserini's LuceneSearcher.

    Conforms to the :class:`nya_ir.retrieval.base.Retriever` protocol.
    BM25 parameters are applied at construction so callers cannot forget to call
    ``set_bm25`` before ``search``. The language is also applied at construction so
    the analyzer used at query time matches the one used at indexing time --- a
    mismatch silently drops the nDCG@10 by several points without any warning.
    """

    def __init__(
        self,
        index_dir: str | Path,
        *,
        k1: float = 0.9,
        b: float = 0.4,
        language: str = "id",
    ) -> None:
        try:
            # pyrefly: ignore[missing-import]  # optional dep; raised cleanly below
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("pyserini", "bm25") from exc
        self._searcher = LuceneSearcher(str(index_dir))
        self._searcher.set_bm25(k1=k1, b=b)
        # MUST match the --language used during build_pyserini_index. With the wrong
        # analyzer Lucene applies English Porter stemming + English stopwords to
        # Indonesian text, costing ~5 nDCG@10 points vs the published MIRACL baseline.
        self._searcher.set_language(language)
        self.k1 = k1
        self.b = b
        self.language = language

    def set_bm25(self, *, k1: float = 0.9, b: float = 0.4) -> None:
        """Re-tune BM25 parameters after construction."""

        self._searcher.set_bm25(k1=k1, b=b)
        self.k1 = k1
        self.b = b

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
    language: str = "id",
) -> list[str]:
    """Return the pyserini command needed to build an index.

    The function does not execute Java/Pyserini. Keeping command construction separate
    makes scripts easier to dry-run and log before expensive indexing.

    ``collection_dir`` is expected to be a directory of ``JsonCollection`` JSONL files
    (one document per line, each with ``id`` and ``contents`` keys); the prepared
    output of :mod:`nya_ir.cli.prepare_miracl` satisfies this contract.

    ``language`` is the Anserini language code (default ``"id"`` for Indonesian).
    Without it, Anserini defaults to ``DefaultEnglishAnalyzer`` with Porter stemming,
    which silently mangles Indonesian tokens and costs roughly 5 nDCG@10 points
    against the published MIRACL baseline. Match this between indexing and search.
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
        "--language",
        language,
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

