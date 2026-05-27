"""Dense BGE-m3 + FAISS HNSW adapters.

The production path streams corpus rows in batches, encodes them with BGE-M3,
adds vectors directly to a FAISS HNSW index, and writes a parallel ``doc_ids``
file. This avoids holding the full MIRACL-id embedding matrix in memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nya_ir.data.io import read_jsonl
from nya_ir.exceptions import OptionalDependencyError
from nya_ir.retrieval.base import RetrievalHit
from nya_ir.utils.hashing import stable_hash

DEFAULT_BGE_M3_MODEL = "BAAI/bge-m3"
DEFAULT_BGE_M3_MAX_LENGTH = 8192
DEFAULT_HNSW_M = 32
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 1000

INDEX_FILENAME = "index.faiss"
DOC_IDS_FILENAME = "doc_ids.txt"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True, slots=True)
class DenseIndexBuildResult:
    """Paths and dimensions produced by a dense index build."""

    index_path: Path
    doc_ids_path: Path
    metadata_path: Path
    num_docs: int
    dimension: int


class SentenceTransformerEncoder:
    """Lazy sentence-transformers encoder wrapper.

    The HuggingFace model is loaded on construction. Embeddings are L2-normalised
    so that inner-product FAISS indexes behave as cosine-similarity searchers.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_M3_MODEL,
        *,
        max_length: int = DEFAULT_BGE_M3_MAX_LENGTH,
    ) -> None:
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
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalised float32 vectors.

    SentenceTransformers already receives ``normalize_embeddings=True`` above,
    but this keeps FAISS ingestion correct when tests or future callers inject a
    lightweight encoder implementation.
    """

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _import_faiss():
    try:
        # pyrefly: ignore[missing-import]  # optional dep; raised cleanly below
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise OptionalDependencyError("faiss-cpu", "dense") from exc
    return faiss


def iter_corpus_jsonl(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield ``(doc_id, contents)`` rows from one JSONL file or JSONL directory."""

    input_path = Path(path)
    paths = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
    for jsonl_path in paths:
        for row in read_jsonl(jsonl_path):
            yield str(row["id"]), str(row["contents"])


def _batched(rows: Iterable[tuple[str, str]], batch_size: int) -> Iterator[list[tuple[str, str]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    batch: list[tuple[str, str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_doc_ids(path: str | Path, doc_ids: Iterable[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for doc_id in doc_ids:
            handle.write(str(doc_id) + "\n")


def read_doc_ids(path: str | Path) -> list[str]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def write_dense_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_dense_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_faiss_hnsw_index(
    collection_path: str | Path,
    index_dir: str | Path,
    *,
    encoder: SentenceTransformerEncoder | None = None,
    model_name: str = DEFAULT_BGE_M3_MODEL,
    max_length: int = DEFAULT_BGE_M3_MAX_LENGTH,
    batch_size: int = 32,
    hnsw_m: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_EF_CONSTRUCTION,
    ef_search: int = DEFAULT_EF_SEARCH,
    threads: int | None = None,
    show_progress_bar: bool = True,
) -> DenseIndexBuildResult:
    """Encode ``collection_path`` and write a FAISS HNSW index under ``index_dir``."""

    input_path = Path(collection_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Collection path not found: {input_path}")

    faiss = _import_faiss()
    if threads is not None and hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(threads)

    active_encoder = encoder or SentenceTransformerEncoder(
        model_name=model_name,
        max_length=max_length,
    )
    output_dir = Path(index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / INDEX_FILENAME
    doc_ids_path = output_dir / DOC_IDS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    index = None
    dimension = 0
    num_docs = 0

    with doc_ids_path.open("w", encoding="utf-8") as doc_id_handle:
        for batch_index, batch in enumerate(
            _batched(iter_corpus_jsonl(input_path), batch_size),
            start=1,
        ):
            doc_ids = [doc_id for doc_id, _contents in batch]
            texts = [contents for _doc_id, contents in batch]
            vectors = active_encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            vectors = np.ascontiguousarray(l2_normalize(vectors), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                raise ValueError(
                    "Encoder returned an invalid shape: "
                    f"expected ({len(batch)}, dim), got {vectors.shape}"
                )
            if index is None:
                dimension = int(vectors.shape[1])
                index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = ef_construction
                index.hnsw.efSearch = ef_search
            elif vectors.shape[1] != dimension:
                raise ValueError(
                    f"Encoder dimension changed from {dimension} to {vectors.shape[1]}"
                )
            index.add(vectors)
            for doc_id in doc_ids:
                doc_id_handle.write(doc_id + "\n")
            num_docs += len(batch)
            if show_progress_bar and batch_index % 100 == 0:
                print(f"  dense indexing: {num_docs:,} docs encoded", flush=True)

    if index is None or num_docs == 0:
        raise ValueError(f"No corpus rows found in {input_path}")

    faiss.write_index(index, str(index_path))
    config_payload = {
        "retriever": "bge_m3",
        "model_name": model_name,
        "max_length": max_length,
        "pooling": "cls",
        "normalize_embeddings": True,
        "faiss_index": "IndexHNSWFlat",
        "metric": "inner_product",
        "hnsw_m": hnsw_m,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "batch_size": batch_size,
        "threads": threads,
    }
    metadata = {
        **config_payload,
        "collection_path": str(input_path),
        "index_path": str(index_path),
        "doc_ids_path": str(doc_ids_path),
        "num_docs": num_docs,
        "dimension": dimension,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": stable_hash(config_payload),
    }
    write_dense_metadata(metadata_path, metadata)
    return DenseIndexBuildResult(
        index_path=index_path,
        doc_ids_path=doc_ids_path,
        metadata_path=metadata_path,
        num_docs=num_docs,
        dimension=dimension,
    )


class FaissDenseSearcher:
    """FAISS searcher for a precomputed dense index.

    ``doc_ids`` must be a sequence whose order corresponds to the row order used
    when the FAISS index was built. The index is loaded eagerly so a missing or
    corrupted index file fails fast at construction time.
    """

    def __init__(
        self,
        index_path: str | Path,
        doc_ids: Sequence[str],
        *,
        ef_search: int | None = None,
    ) -> None:
        faiss = _import_faiss()
        self._faiss = faiss
        self._index = faiss.read_index(str(index_path))
        self._doc_ids = list(doc_ids)
        if len(self._doc_ids) < self._index.ntotal:
            raise ValueError(
                f"doc_ids has {len(self._doc_ids)} rows but FAISS index has "
                f"{self._index.ntotal} vectors"
            )
        if ef_search is not None and hasattr(self._index, "hnsw"):
            self._index.hnsw.efSearch = ef_search

    def search_vector(self, query_vector: np.ndarray, *, top_k: int = 1000) -> list[RetrievalHit]:
        """Search the FAISS index with a single pre-encoded query vector.

        Ranks are derived from the FAISS slot position (1-indexed), NOT from the
        post-filter loop counter. If FAISS returns sentinel ``-1`` indices for
        invalid slots (which happens when ``top_k`` > index size, or after
        deletions), the loop skips them but the surviving hits keep their
        original rank — preserving the contract that ``rank`` reflects FAISS's
        own ordering. The previous reindex-on-skip behaviour silently shifted
        ranks, breaking rank-continuity expectations of downstream trec_eval.
        """

        return self.search_vectors(query_vector, top_k=top_k)[0]

    def search_vectors(
        self,
        query_vectors: np.ndarray,
        *,
        top_k: int = 1000,
    ) -> list[list[RetrievalHit]]:
        """Search the FAISS index with one or more pre-encoded query vectors."""

        matrix = np.ascontiguousarray(l2_normalize(query_vectors), dtype=np.float32)
        scores, indices = self._index.search(matrix, top_k)
        all_hits: list[list[RetrievalHit]] = []
        for query_scores, query_indices in zip(scores, indices, strict=True):
            all_hits.append(self._hits_from_faiss_row(query_scores, query_indices))
        return all_hits

    def _hits_from_faiss_row(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for slot_position, (score, idx) in enumerate(
            zip(scores, indices, strict=False), start=1
        ):
            if idx < 0:
                continue
            hits.append(
                RetrievalHit(
                    doc_id=self._doc_ids[int(idx)],
                    score=float(score),
                    rank=slot_position,
                )
            )
        return hits


class DenseTextSearcher:
    """Text-in, hits-out dense retriever used by the generic retrieval CLI."""

    def __init__(
        self,
        encoder: SentenceTransformerEncoder,
        searcher: FaissDenseSearcher,
        *,
        batch_size: int = 32,
    ) -> None:
        self.encoder = encoder
        self.searcher = searcher
        self.batch_size = batch_size

    def search(self, query: str, *, top_k: int = 1000) -> list[RetrievalHit]:
        vector = self.encoder.encode([query], batch_size=self.batch_size, show_progress_bar=False)[0]
        return self.searcher.search_vector(vector, top_k=top_k)

    def search_many(self, queries: Sequence[str], *, top_k: int = 1000) -> list[list[RetrievalHit]]:
        vectors = self.encoder.encode(queries, batch_size=self.batch_size, show_progress_bar=False)
        return self.searcher.search_vectors(vectors, top_k=top_k)


def load_dense_text_searcher(
    index_dir: str | Path,
    *,
    model_name: str = DEFAULT_BGE_M3_MODEL,
    max_length: int = DEFAULT_BGE_M3_MAX_LENGTH,
    batch_size: int = 32,
    ef_search: int = DEFAULT_EF_SEARCH,
) -> DenseTextSearcher:
    """Load a BGE-M3 text searcher from an index directory."""

    base = Path(index_dir)
    index_path = base / INDEX_FILENAME
    doc_ids_path = base / DOC_IDS_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(f"Dense FAISS index not found: {index_path}")
    if not doc_ids_path.exists():
        raise FileNotFoundError(f"Dense doc_ids file not found: {doc_ids_path}")
    encoder = SentenceTransformerEncoder(model_name=model_name, max_length=max_length)
    return DenseTextSearcher(
        encoder,
        FaissDenseSearcher(index_path, read_doc_ids(doc_ids_path), ef_search=ef_search),
        batch_size=batch_size,
    )
