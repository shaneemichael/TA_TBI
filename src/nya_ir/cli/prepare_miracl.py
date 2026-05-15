"""Prepare MIRACL-id data for one preprocessing strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

from nya_ir.data.io import write_jsonl
from nya_ir.data.miracl import load_miracl_corpus, load_miracl_queries
from nya_ir.experiment import StrategyName
from nya_ir.preprocessing import SuffixNyaRemover, apply_strategy


def load_root_dict(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=[strategy.value for strategy in StrategyName], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--language", default="id")
    parser.add_argument("--query-split", default="dev")
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--root-dict", type=Path)
    parser.add_argument("--limit", type=int, help="Optional smoke-test row limit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned outputs only.")
    parser.add_argument(
        "--use-suffix-remover",
        action="store_true",
        help="Use the dependency-light suffix remover instead of PySastrawi for smoke tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    strategy = StrategyName(args.strategy)
    root_dict = load_root_dict(args.root_dict)
    remover = SuffixNyaRemover() if args.use_suffix_remover else None

    query_path = args.output_dir / strategy.value / f"queries_{args.query_split}.jsonl"
    corpus_path = args.output_dir / strategy.value / f"corpus_{args.corpus_split}.jsonl"
    if args.dry_run:
        print(f"queries -> {query_path}")
        print(f"corpus  -> {corpus_path}")
        return 0

    queries = []
    for index, query in enumerate(load_miracl_queries(split=args.query_split, language=args.language)):
        if args.limit is not None and index >= args.limit:
            break
        queries.append(
            {
                "id": query.query_id,
                "contents": apply_strategy(
                    query.text,
                    strategy,
                    root_dict=root_dict,
                    remover=remover,
                ),
            }
        )

    passages = []
    for index, passage in enumerate(load_miracl_corpus(split=args.corpus_split, language=args.language)):
        if args.limit is not None and index >= args.limit:
            break
        contents = passage.text if passage.title is None else f"{passage.title}\n{passage.text}"
        passages.append(
            {
                "id": passage.doc_id,
                "contents": apply_strategy(
                    contents,
                    strategy,
                    root_dict=root_dict,
                    remover=remover,
                ),
            }
        )

    write_jsonl(query_path, queries)
    write_jsonl(corpus_path, passages)
    print(f"Wrote {len(queries)} queries to {query_path}")
    print(f"Wrote {len(passages)} passages to {corpus_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
