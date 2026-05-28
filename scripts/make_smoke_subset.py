"""Build a small consistent dense-retrieval smoke subset from preprocessed data.

Given a processed dataset root with the five strategy subdirectories and a qrels
file, this script selects the first N queries from the Keep condition, includes
all relevant docs for those queries, adds a configurable number of filler docs,
and writes a mini dataset that preserves the original per-strategy texts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nya_ir.data.io import read_jsonl, read_qrels, write_jsonl  # noqa: E402

STRATEGIES: tuple[str, ...] = (
    "keep",
    "naive_strip",
    "sastrawi_clitic",
    "sentinel",
    "rule_resolved",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--queries",
        type=int,
        default=20,
        help="Number of dev queries to keep from the head of keep/queries_dev.jsonl.",
    )
    parser.add_argument(
        "--extra-docs",
        type=int,
        default=1000,
        help="Number of non-relevant filler docs to add beyond the relevant set.",
    )
    return parser


def _read_head_queries(path: Path, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_jsonl(path):
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _load_queries_by_id(path: Path, wanted_ids: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_jsonl(path):
        query_id = str(row["id"])
        if query_id in wanted_ids:
            rows.append({"id": query_id, "contents": str(row["contents"])})
    return rows


def _select_doc_ids(corpus_path: Path, relevant_doc_ids: set[str], extra_docs: int) -> set[str]:
    selected = set(relevant_doc_ids)
    filler = 0
    for row in read_jsonl(corpus_path):
        doc_id = str(row["id"])
        if doc_id in selected:
            continue
        selected.add(doc_id)
        filler += 1
        if filler >= extra_docs:
            break
    return selected


def _write_subset_corpus(input_path: Path, output_path: Path, wanted_doc_ids: set[str]) -> int:
    kept_rows: list[dict[str, object]] = []
    for row in read_jsonl(input_path):
        doc_id = str(row["id"])
        if doc_id in wanted_doc_ids:
            kept_rows.append({"id": doc_id, "contents": str(row["contents"])})
    write_jsonl(output_path, kept_rows)
    return len(kept_rows)


def _write_subset_qrels(
    input_path: Path,
    output_path: Path,
    wanted_query_ids: set[str],
    wanted_doc_ids: set[str],
) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            query_id, iteration, doc_id, relevance = line.split()
            if query_id in wanted_query_ids and doc_id in wanted_doc_ids:
                dst.write(f"{query_id} {iteration} {doc_id} {relevance}\n")
                count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    keep_queries_path = args.processed_dir / "keep" / "queries_dev.jsonl"
    keep_corpus_path = args.processed_dir / "keep" / "corpus_train.jsonl"
    if not keep_queries_path.exists():
        raise SystemExit(f"Missing keep queries file: {keep_queries_path}")
    if not keep_corpus_path.exists():
        raise SystemExit(f"Missing keep corpus file: {keep_corpus_path}")
    if not args.qrels.exists():
        raise SystemExit(f"Missing qrels file: {args.qrels}")

    keep_head_rows = _read_head_queries(keep_queries_path, args.queries)
    if not keep_head_rows:
        raise SystemExit("No queries found in keep/queries_dev.jsonl")
    query_ids = [str(row["id"]) for row in keep_head_rows]
    query_id_set = set(query_ids)

    qrels = read_qrels(args.qrels)
    relevant_doc_ids = {
        doc_id
        for query_id in query_ids
        for doc_id, relevance in qrels.get(query_id, {}).items()
        if relevance > 0
    }
    if not relevant_doc_ids:
        raise SystemExit("Selected queries have zero relevant docs in the supplied qrels file")

    subset_doc_ids = _select_doc_ids(keep_corpus_path, relevant_doc_ids, args.extra_docs)

    for strategy in STRATEGIES:
        strategy_dir = args.processed_dir / strategy
        queries_in = strategy_dir / "queries_dev.jsonl"
        corpus_in = strategy_dir / "corpus_train.jsonl"
        if not queries_in.exists() or not corpus_in.exists():
            raise SystemExit(f"Missing strategy files under {strategy_dir}")

        queries_out = args.output_dir / strategy / "queries_dev.jsonl"
        corpus_out = args.output_dir / strategy / "corpus_train.jsonl"
        strategy_queries = _load_queries_by_id(queries_in, query_id_set)
        write_jsonl(queries_out, strategy_queries)
        doc_count = _write_subset_corpus(corpus_in, corpus_out, subset_doc_ids)
        print(f"{strategy:>16}: {len(strategy_queries)} queries, {doc_count} docs")

    qrels_out = args.output_dir / "qrels" / "qrels_dev.txt"
    qrel_count = _write_subset_qrels(args.qrels, qrels_out, query_id_set, subset_doc_ids)
    print(f"{'qrels':>16}: {qrel_count} judgments")
    print(f"subset root: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
