"""Dense BGE-m3 + FAISS adapter skeleton.

This module is scaffolded: the BM25 path is the focus of the current progress
report. Both classes lazy-import their optional dependencies and raise
:class:`nya_ir.exceptions.OptionalDependencyError` with the corresponding
``pip install .[dense]`` hint when those packages are missing.

Intended usage once wired:

1. Build phase --- encode the prepared corpus JSONL with
   :class:`SentenceTransformerEncoder`, write vectors and a parallel
   ``doc_ids`` list, then build a FAISS HNSW index from the vectors.
2. Search phase --- load the index with :class:`FaissDenseSearcher`, encode
   each query, and call :meth:`FaissDenseSearcher.search_vector`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from nya_ir.exceptions import OptionalDependencyError
from nya_ir.retrieval.base import RetrievalHit


class SentenceTransformerEncoder:
    """Lazy sentence-transformers encoder wrapper.

    The HuggingFace model is loaded on construction. Embeddings are L2-normalised
    so that inner-product FAISS indexes behave as cosine-similarity searchers.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", *, max_length: int = 512) -> None:
        try:
            # pyrefly: ignore[missing-import]  # optional dep; raised cleanly below
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("sentence-transformers", "dense") from exc
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_length
        self.model_name = model_name

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode ``texts`` into an ``(N, D)`` float32 matrix of L2-normalised vectors."""

        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(vectors, dtype=np.float32)


class FaissDenseSearcher:
    """FAISS searcher for a precomputed dense index.

    ``doc_ids`` must be a sequence whose order corresponds to the row order used
    when the FAISS index was built. The index is loaded eagerly so a missing or
    corrupted index file fails fast at construction time.
    """

    def __init__(self, index_path: str | Path, doc_ids: Sequence[str]) -> None:
        try:
            # pyrefly: ignore[missing-import]  # optional dep; raised cleanly below
            import faiss
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("faiss-cpu", "dense") from exc
        self._faiss = faiss
        self._index = faiss.read_index(str(index_path))
        self._doc_ids = list(doc_ids)

    def search_vector(self, query_vector: np.ndarray, *, top_k: int = 1000) -> list[RetrievalHit]:
        """Search the FAISS index with a single pre-encoded query vector."""

        scores, indices = self._index.search(
            query_vector.reshape(1, -1).astype(np.float32), top_k
        )
        hits: list[RetrievalHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False), start=1):
            if idx < 0:
                continue
            hits.append(RetrievalHit(doc_id=self._doc_ids[int(idx)], score=float(score), rank=rank))
        return hits

