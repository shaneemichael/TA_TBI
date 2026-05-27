"""Prepare MIRACL-id data for one preprocessing strategy.

Streams rows through ``write_jsonl`` instead of accumulating the entire
processed corpus in memory first. On the full MIRACL-id corpus this means
(a) the ``.tmp`` output file grows visibly during the run instead of
appearing all at once at the end, (b) peak memory is bounded by the
``write_jsonl`` buffer rather than ~1.5 GB of materialised dicts, and
(c) tqdm progress bars give per-strategy ETAs so a 30-90 min Sastrawi
pass on Strategy 3 / Strategy 5 stops looking stuck.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from nya_ir.data.io import write_jsonl
from nya_ir.data.miracl import (
    load_corpus_jsonl,
    load_miracl_corpus,
    load_miracl_queries,
    load_query_jsonl,
)
from nya_ir.experiment import StrategyName
from nya_ir.preprocessing import SuffixNyaRemover, apply_strategy


def load_root_dict(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _make_progress(label: str, total: int | None):
    """Return a tqdm wrapper if available, else a passthrough that logs every 50k rows.

    tqdm comes in transitively via ``datasets`` and the FlagEmbedding stack so
    in practice the production path always uses it. The fallback exists so the
    CLI keeps working in a minimal sandbox install that pulled only the hard
    deps from ``pyproject.toml``.
    """

    try:
        from tqdm import tqdm  # type: ignore[import]
    except ImportError:
        def _passthrough(iterable):
            for i, item in enumerate(iterable, start=1):
                if i % 50_000 == 0:
                    suffix = f"/{total}" if total else ""
                    print(f"  {label}: {i}{suffix} rows processed", file=sys.stderr, flush=True)
                yield item

        return _passthrough

    def _wrap(iterable):
        return tqdm(iterable, desc=label, unit="row", total=total, dynamic_ncols=True)

    return _wrap


def _try_len(source) -> int | None:
    """``len(source)`` if cheap (HF Dataset, list), else ``None``.

    Used only to feed tqdm's ETA. Counting JSONL lines up front would double-read
    the file, so we accept "no ETA" in that case.
    """

    try:
        return len(source)
    except TypeError:
        return None


def _process_queries(
    source: Iterable,
    *,
    strategy: StrategyName,
    root_dict: set[str],
    remover,
    limit: int | None,
) -> Iterator[dict[str, object]]:
    progress = _make_progress("queries", _try_len(source))
    for index, query in enumerate(progress(source)):
        if limit is not None and index >= limit:
            break
        yield {
            "id": query.query_id,
            "contents": apply_strategy(
                query.text,
                strategy,
                root_dict=root_dict,
                remover=remover,
            ),
        }


def _process_corpus(
    source: Iterable,
    *,
    strategy: StrategyName,
    root_dict: set[str],
    remover,
    limit: int | None,
) -> Iterator[dict[str, object]]:
    progress = _make_progress("corpus", _try_len(source))
    for index, passage in enumerate(progress(source)):
        if limit is not None and index >= limit:
            break
        contents = (
            passage.text if passage.title is None else f"{passage.title}\n{passage.text}"
        )
        yield {
            "id": passage.doc_id,
            "contents": apply_strategy(
                contents,
                strategy,
                root_dict=root_dict,
                remover=remover,
            ),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in StrategyName],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--language", default="id")
    parser.add_argument("--query-split", default="dev")
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument(
        "--queries-jsonl",
        type=Path,
        help="Local query JSONL for offline smoke tests.",
    )
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        help="Local corpus JSONL for offline smoke tests.",
    )
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.queries_jsonl is None) != (args.corpus_jsonl is None):
        parser.error("--queries-jsonl and --corpus-jsonl must be provided together")

    strategy = StrategyName(args.strategy)
    root_dict = load_root_dict(args.root_dict)
    remover = SuffixNyaRemover() if args.use_suffix_remover else None

    query_path = args.output_dir / strategy.value / f"queries_{args.query_split}.jsonl"
    corpus_path = args.output_dir / strategy.value / f"corpus_{args.corpus_split}.jsonl"
    if args.dry_run:
        print(f"queries -> {query_path}")
        print(f"corpus  -> {corpus_path}")
        return 0

    query_source = (
        load_query_jsonl(args.queries_jsonl)
        if args.queries_jsonl is not None
        else load_miracl_queries(split=args.query_split, language=args.language)
    )
    write_jsonl(
        query_path,
        _process_queries(
            query_source,
            strategy=strategy,
            root_dict=root_dict,
            remover=remover,
            limit=args.limit,
        ),
    )
    print(f"Wrote queries to {query_path}")

    corpus_source = (
        load_corpus_jsonl(args.corpus_jsonl)
        if args.corpus_jsonl is not None
        else load_miracl_corpus(split=args.corpus_split, language=args.language)
    )
    write_jsonl(
        corpus_path,
        _process_corpus(
            corpus_source,
            strategy=strategy,
            root_dict=root_dict,
            remover=remover,
            limit=args.limit,
        ),
    )
    print(f"Wrote corpus to {corpus_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
