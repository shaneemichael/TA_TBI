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
    parser.add_argument("--execute", action="store_true", help="Actually run the indexing command.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    retriever = RetrieverName(args.retriever)
    if retriever is RetrieverName.BM25:
        command = build_pyserini_index(
            collection_dir=args.collection_dir,
            index_dir=args.index_dir,
            threads=args.threads,
        )
        print(" ".join(command))
        if args.execute:
            return subprocess.run(command, check=False).returncode
        return 0

    print(
        "Dense indexing requires encoding passages with BGE-m3 and building a FAISS HNSW index. "
        "Use the dense adapter in nya_ir.retrieval.dense for the implementation stage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

