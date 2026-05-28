"""Build or dry-run retrieval indexes."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from nya_ir.experiment import RetrieverName
from nya_ir.retrieval.bm25 import build_pyserini_index
from nya_ir.retrieval.dense import (
    DEFAULT_BGE_M3_MAX_LENGTH,
    DEFAULT_BGE_M3_MODEL,
    DEFAULT_EF_CONSTRUCTION,
    DEFAULT_EF_SEARCH,
    DEFAULT_HNSW_M,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=[retriever.value for retriever in RetrieverName], required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", default=DEFAULT_BGE_M3_MODEL)
    parser.add_argument("--max-length", type=int, default=DEFAULT_BGE_M3_MAX_LENGTH)
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated dense encoding devices for FlagEmbedding, e.g. "
            "'cuda:0', 'cuda:0,cuda:1', or 'cpu'. Omit to use library defaults."
        ),
    )
    parser.add_argument("--hnsw-m", type=int, default=DEFAULT_HNSW_M)
    parser.add_argument("--ef-construction", type=int, default=DEFAULT_EF_CONSTRUCTION)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show dense indexing progress with a single docs/second + ETA indicator.",
    )
    parser.add_argument(
        "--language",
        default="id",
        help=(
            "Anserini language code for the analyzer (default: id). "
            "Must match the language used at search time."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the indexing command (default: dry-run, print only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    retriever = RetrieverName(args.retriever)
    if retriever is RetrieverName.BM25:
        if not args.collection_dir.exists():
            raise SystemExit(f"Collection path not found: {args.collection_dir}")
        command = build_pyserini_index(
            collection_dir=args.collection_dir,
            index_dir=args.index_dir,
            threads=args.threads,
            language=args.language,
        )
        print(" ".join(command))
        if args.execute:
            args.index_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.run(command, check=False).returncode
        return 0

    if retriever is RetrieverName.BGE_M3:
        if not args.collection_dir.exists():
            raise SystemExit(f"Collection path not found: {args.collection_dir}")
        plan = (
            "BGE-m3 FAISS HNSW index: "
            f"collection={args.collection_dir} index_dir={args.index_dir} "
            f"model={args.model_name} max_length={args.max_length} "
            f"devices={args.devices or 'auto'} "
            f"batch_size={args.batch_size} hnsw_m={args.hnsw_m} "
            f"ef_construction={args.ef_construction} ef_search={args.ef_search}"
        )
        print(plan)
        if not args.execute:
            return 0

        from nya_ir.retrieval.dense import build_faiss_hnsw_index

        result = build_faiss_hnsw_index(
            args.collection_dir,
            args.index_dir,
            model_name=args.model_name,
            max_length=args.max_length,
            devices=args.devices,
            batch_size=args.batch_size,
            hnsw_m=args.hnsw_m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            threads=args.threads,
            show_progress_bar=args.show_progress,
        )
        print(
            f"Wrote dense index with {result.num_docs} docs, "
            f"dim={result.dimension} to {result.index_path}"
        )
        return 0

    raise SystemExit(f"Unsupported retriever: {retriever.value}")


if __name__ == "__main__":
    raise SystemExit(main())
