"""Build or dry-run retrieval indexes."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from nya_ir.experiment import RetrieverName
from nya_ir.retrieval.bm25 import build_pyserini_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=[retriever.value for retriever in RetrieverName], required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
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
        )
        print(" ".join(command))
        if args.execute:
            args.index_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.run(command, check=False).returncode
        return 0

    # Non-BM25 path: explicit not-implemented exit so callers see a non-zero status
    # rather than a silent success that produces no index.
    raise SystemExit(
        "Dense indexing (BGE-m3 + FAISS HNSW) is scaffolded only. "
        "See nya_ir.retrieval.dense for the implementation stage."
    )


if __name__ == "__main__":
    raise SystemExit(main())

