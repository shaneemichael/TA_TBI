"""Run retrieval for prepared queries and an existing index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nya_ir.data.records import RunEntry
from nya_ir.data.io import write_trec_run
from nya_ir.experiment import RetrieverName
from nya_ir.retrieval.bm25 import PyseriniBM25Searcher


def iter_query_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                yield str(row["id"]), str(row["contents"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=[retriever.value for retriever in RetrieverName], required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hits", type=int, default=1000)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    retriever = RetrieverName(args.retriever)
    if retriever is not RetrieverName.BM25:
        raise SystemExit("Dense retrieval runner is scaffolded but not implemented yet.")

    searcher = PyseriniBM25Searcher(args.index_dir)
    searcher.set_bm25(k1=args.k1, b=args.b)
    entries: list[RunEntry] = []
    for query_id, query_text in iter_query_jsonl(args.queries):
        for hit in searcher.search(query_text, top_k=args.hits):
            entries.append(
                RunEntry(
                    query_id=query_id,
                    doc_id=hit.doc_id,
                    rank=hit.rank,
                    score=hit.score,
                    run_id=args.run_id,
                )
            )
    write_trec_run(args.output, entries)
    print(f"Wrote {len(entries)} run entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

