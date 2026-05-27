"""Dense BGE-m3 + FAISS HNSW adapters.

The production path streams corpus rows in batches, encodes them with BGE-M3
via FlagEmbedding, adds vectors directly to a FAISS HNSW index, and writes a
parallel ``doc_ids`` file. This avoids holding the full MIRACL-id embedding
matrix in memory.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
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


@contextmanager
def _suppress_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def count_corpus_rows(path: str | Path) -> int:
    """Count non-empty JSONL rows in one file or a directory of JSONLs."""

    input_path = Path(path)
    paths = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
    count = 0
    for jsonl_path in paths:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    return count


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _NoOpProgress:
    def update(self, _amount: int) -> None:
        return

    def close(self) -> None:
        return


class _ConsoleProgress:
    def __init__(self, total_docs: int) -> None:
        self.total_docs = max(total_docs, 1)
        self.done_docs = 0
        self._started_at = time.monotonic()
        self._last_emit = 0.0

    def update(self, amount: int) -> None:
        self.done_docs += amount
        now = time.monotonic()
        if now - self._last_emit < 1.0 and self.done_docs < self.total_docs:
            return
        elapsed = max(now - self._started_at, 1e-6)
        rate = self.done_docs / elapsed
        remaining = max(self.total_docs - self.done_docs, 0)
        eta = remaining / rate if rate > 0 else None
        print(
            "\r"
            f"  dense indexing: {self.done_docs:,}/{self.total_docs:,} docs "
            f"| {rate:,.1f} docs/s | ETA {_format_eta(eta)}",
            end="",
            flush=True,
        )
        self._last_emit = now

    def close(self) -> None:
        self.update(0)
        print("", flush=True)


def _make_index_progress(total_docs: int, enabled: bool):
    if not enabled:
        return _NoOpProgress()
    try:
        # pyrefly: ignore[missing-import]  # optional dep in minimal installs
        from tqdm import tqdm
    except ImportError:
        return _ConsoleProgress(total_docs)
    return tqdm(total=total_docs, desc="dense indexing", unit="doc", dynamic_ncols=True)


def parse_devices(raw_devices: str | Sequence[str] | None) -> list[str] | None:
    """Parse a dense-device specification into a clean list of device strings."""

    if raw_devices is None:
        return None
    if isinstance(raw_devices, str):
        devices = [part.strip() for part in raw_devices.split(",")]
    else:
        devices = [str(part).strip() for part in raw_devices]
    parsed = [device for device in devices if device]
    return parsed or None


def default_single_device() -> list[str] | None:
    """Return one explicit device for query-time encoding when possible."""

    try:
        # pyrefly: ignore[missing-import]  # optional dep; raised elsewhere if absent
        import torch
    except ImportError:  # pragma: no cover - depends on optional package
        return None
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        return ["cuda:0"]
    return ["cpu"]


class FlagEmbeddingEncoder:
    """Lazy FlagEmbedding dense encoder wrapper.

    The model is loaded on construction. Embeddings are L2-normalised so that
    inner-product FAISS indexes behave as cosine-similarity searchers.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_M3_MODEL,
        *,
        max_length: int = DEFAULT_BGE_M3_MAX_LENGTH,
        devices: str | Sequence[str] | None = None,
    ) -> None:
        parsed_devices = parse_devices(devices)
        try:
            # pyrefly: ignore[missing-import]  # optional dep; raised cleanly below
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("FlagEmbedding", "dense") from exc
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=any(device.startswith("cuda") for device in parsed_devices or []),
            devices=parsed_devices,
            query_max_length=max_length,
            passage_max_length=max_length,
        )
        self.model_name = model_name
        self.max_length = max_length
        self.devices = parsed_devices

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode ``texts`` into an ``(N, D)`` float32 matrix of L2-normalised vectors."""

        with _suppress_output(enabled=not show_progress_bar):
            output = self.model.encode(
                list(texts),
                batch_size=batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        vectors = output["dense_vecs"]
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalised float32 vectors.

    FlagEmbedding already returns normalized dense vectors, but this keeps FAISS
    ingestion correct when tests or future callers inject a lightweight encoder
    implementation.
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
    encoder: FlagEmbeddingEncoder | None = None,
    model_name: str = DEFAULT_BGE_M3_MODEL,
    max_length: int = DEFAULT_BGE_M3_MAX_LENGTH,
    devices: str | Sequence[str] | None = None,
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

    active_encoder = encoder or FlagEmbeddingEncoder(
        model_name=model_name,
        max_length=max_length,
        devices=devices,
    )
    output_dir = Path(index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / INDEX_FILENAME
    doc_ids_path = output_dir / DOC_IDS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    index = None
    dimension = 0
    num_docs = 0
    total_docs = count_corpus_rows(input_path) if show_progress_bar else 0
    progress = _make_index_progress(total_docs, show_progress_bar)

    try:
        with doc_ids_path.open("w", encoding="utf-8") as doc_id_handle:
            for batch in _batched(iter_corpus_jsonl(input_path), batch_size):
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
                batch_size_done = len(batch)
                num_docs += batch_size_done
                progress.update(batch_size_done)
    finally:
        progress.close()

    if index is None or num_docs == 0:
        raise ValueError(f"No corpus rows found in {input_path}")

    faiss.write_index(index, str(index_path))
    config_payload = {
        "retriever": "bge_m3",
        "model_name": model_name,
        "max_length": max_length,
        "devices": active_encoder.devices,
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
        encoder: FlagEmbeddingEncoder,
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
    encoder = FlagEmbeddingEncoder(
        model_name=model_name,
        max_length=max_length,
        devices=default_single_device(),
    )
    return DenseTextSearcher(
        encoder,
        FaissDenseSearcher(index_path, read_doc_ids(doc_ids_path), ef_search=ef_search),
        batch_size=batch_size,
    )
