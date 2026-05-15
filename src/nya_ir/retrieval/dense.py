"""Dense BGE-m3 + FAISS adapter skeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nya_ir.exceptions import OptionalDependencyError
from nya_ir.retrieval.base import RetrievalHit


class SentenceTransformerEncoder:
    """Lazy sentence-transformers encoder wrapper."""

    def __init__(self, model_name: str = "BAAI/bge-m3", *, max_length: int = 512) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("sentence-transformers", "dense") from exc
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_length

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class FaissDenseSearcher:
    """FAISS searcher for precomputed dense vectors."""

    def __init__(self, index_path: str | Path, doc_ids: list[str]) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("faiss-cpu", "dense") from exc
        self._faiss = faiss
        self._index = faiss.read_index(str(index_path))
        self._doc_ids = doc_ids

    def search_vector(self, query_vector: np.ndarray, *, top_k: int = 1000) -> list[RetrievalHit]:
        scores, indices = self._index.search(query_vector.reshape(1, -1).astype(np.float32), top_k)
        hits: list[RetrievalHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False), start=1):
            if idx < 0:
                continue
            hits.append(RetrievalHit(doc_id=self._doc_ids[int(idx)], score=float(score), rank=rank))
        return hits

