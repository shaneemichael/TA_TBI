"""Run retrieval for prepared queries and an existing index."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from nya_ir.data.io import write_trec_run
from nya_ir.data.records import RunEntry
from nya_ir.experiment import RetrieverName
from nya_ir.retrieval.base import Retriever


def iter_query_jsonl(path: Path) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                yield str(row["id"]), str(row["contents"])


def run_searches(
    searcher: Retriever,
    queries: Iterable[tuple[str, str]],
    *,
    run_id: str,
    hits: int,
) -> list[RunEntry]:
    """Apply ``searcher`` to ``queries`` and collect TREC run entries.

    Factored out of :func:`main` so tests can inject a stub searcher that does not
    require a real Lucene index or Pyserini install.
    """

    entries: list[RunEntry] = []
    for query_id, query_text in queries:
        for hit in searcher.search(query_text, top_k=hits):
            entries.append(
                RunEntry(
                    query_id=query_id,
                    doc_id=hit.doc_id,
                    rank=hit.rank,
                    score=hit.score,
                    run_id=run_id,
                )
            )
    return entries


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
    parser.add_argument(
        "--language",
        default="id",
        help=(
            "Anserini language code for the analyzer (default: id). "
            "Must match the language used when the index was built."
        ),
    )
    return parser


def _build_searcher(args: argparse.Namespace) -> Retriever:
    retriever = RetrieverName(args.retriever)
    if retriever is RetrieverName.BM25:
        from nya_ir.retrieval.bm25 import PyseriniBM25Searcher

        return PyseriniBM25Searcher(
            args.index_dir, k1=args.k1, b=args.b, language=args.language
        )
    raise SystemExit(
        f"Retriever {retriever.value!r} is scaffolded but not yet wired into this CLI."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.queries.exists():
        raise SystemExit(f"Queries file not found: {args.queries}")

    searcher = _build_searcher(args)
    entries = run_searches(
        searcher,
        iter_query_jsonl(args.queries),
        run_id=args.run_id,
        hits=args.hits,
    )
    write_trec_run(args.output, entries)
    print(f"Wrote {len(entries)} run entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

