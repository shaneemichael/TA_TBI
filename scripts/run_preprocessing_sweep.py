"""Run prepare_miracl across all 5 preprocessing strategies.

Owned by Person A on Day 1. Runs prepare_miracl with consistent flags for
each strategy, with pre-flight and post-flight checks so a silent failure
in one strategy can't poison the experiment matrix.

Behaviour:
  * Idempotent. If ``data/processed/{strategy}/corpus_train.jsonl`` and
    ``queries_dev.jsonl`` both exist with non-empty content, the strategy is
    skipped. Pass ``--force`` to re-run anyway.
  * Fails fast. Any non-zero exit from prepare_miracl halts the sweep — we
    do NOT want strategy 4 silently missing while strategies 1-3 succeeded.
  * Sanity-checks at the end:
      - corpus line counts match across strategies (must — same row count
        in / out for every strategy).
      - rule_resolved corpus differs from keep corpus on at least one row
        (proves the resolver actually fired somewhere). A *suspicious*
        zero-difference is logged as a warning, not a hard failure, because
        on a tiny smoke corpus rule_resolved may genuinely produce no
        substitutions.

Usage:
    python scripts/run_preprocessing_sweep.py \\
        --root-dict artifacts/root_dict.txt \\
        --output-dir data/processed

For offline smoke runs against a fixture, pass --queries-jsonl and
--corpus-jsonl (forwarded straight to prepare_miracl).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STRATEGIES: tuple[str, ...] = (
    "keep",
    "naive_strip",
    "sastrawi_clitic",
    "sentinel",
    "rule_resolved",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dict", type=Path, default=Path("artifacts/root_dict.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--language", default="id")
    parser.add_argument("--query-split", default="dev")
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument(
        "--queries-jsonl",
        type=Path,
        help="Forwarded to prepare_miracl for offline runs (must pair with --corpus-jsonl).",
    )
    parser.add_argument("--corpus-jsonl", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--use-suffix-remover",
        action="store_true",
        help="Forward --use-suffix-remover to prepare_miracl (lightweight; for smoke tests).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run strategies even when their outputs already exist.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to invoke prepare_miracl with.",
    )
    return parser


def _strategy_outputs(output_dir: Path, strategy: str, args: argparse.Namespace) -> tuple[Path, Path]:
    return (
        output_dir / strategy / f"queries_{args.query_split}.jsonl",
        output_dir / strategy / f"corpus_{args.corpus_split}.jsonl",
    )


def _already_done(queries: Path, corpus: Path) -> bool:
    return (
        queries.exists()
        and corpus.exists()
        and queries.stat().st_size > 0
        and corpus.stat().st_size > 0
    )


def _run_strategy(strategy: str, args: argparse.Namespace) -> None:
    cmd: list[str] = [
        args.python,
        "-m",
        "nya_ir.cli.prepare_miracl",
        "--strategy",
        strategy,
        "--output-dir",
        str(args.output_dir),
        "--language",
        args.language,
        "--query-split",
        args.query_split,
        "--corpus-split",
        args.corpus_split,
    ]
    if args.root_dict is not None:
        cmd += ["--root-dict", str(args.root_dict)]
    if args.queries_jsonl is not None:
        cmd += ["--queries-jsonl", str(args.queries_jsonl)]
    if args.corpus_jsonl is not None:
        cmd += ["--corpus-jsonl", str(args.corpus_jsonl)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.use_suffix_remover:
        cmd.append("--use-suffix-remover")

    print(f"\n=== {strategy} ===")
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _first_diff_row(left: Path, right: Path) -> int | None:
    """Return the (0-indexed) line number of the first row where two JSONL files differ, or None."""
    with left.open("r", encoding="utf-8") as left_handle, right.open(
        "r", encoding="utf-8"
    ) as right_handle:
        for i, (a, b) in enumerate(zip(left_handle, right_handle, strict=False)):
            if a != b:
                return i
    return None


def _post_flight_checks(args: argparse.Namespace) -> int:
    print("\n=== post-flight checks ===")
    counts: dict[str, tuple[int, int]] = {}
    for strategy in STRATEGIES:
        queries, corpus = _strategy_outputs(args.output_dir, strategy, args)
        counts[strategy] = (_count_jsonl_rows(queries), _count_jsonl_rows(corpus))
        print(f"  {strategy:>16}: {counts[strategy][0]} queries, {counts[strategy][1]} passages")

    # All strategies must have the same row count — they preprocess the same input rows.
    query_counts = {c[0] for c in counts.values()}
    corpus_counts = {c[1] for c in counts.values()}
    if len(query_counts) != 1 or len(corpus_counts) != 1:
        print(
            "  ! ROW COUNT MISMATCH across strategies — preprocessing dropped rows somewhere",
            file=sys.stderr,
        )
        return 1

    # Rule-resolved should differ from keep somewhere unless the corpus is genuinely
    # antecedent-free (only a real risk on tiny smoke fixtures).
    keep_corpus = args.output_dir / "keep" / f"corpus_{args.corpus_split}.jsonl"
    rr_corpus = args.output_dir / "rule_resolved" / f"corpus_{args.corpus_split}.jsonl"
    diff_row = _first_diff_row(keep_corpus, rr_corpus)
    if diff_row is None:
        print(
            "  ! WARNING: rule_resolved corpus is byte-identical to keep corpus. "
            "Expected on tiny smoke fixtures; investigate on full MIRACL-id."
        )
    else:
        print(f"  ✓ rule_resolved differs from keep starting at corpus row {diff_row}")

    print("\nAll strategies produced consistent outputs.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if (args.queries_jsonl is None) != (args.corpus_jsonl is None):
        print("error: --queries-jsonl and --corpus-jsonl must be provided together", file=sys.stderr)
        return 2

    if args.root_dict is not None and not args.root_dict.exists():
        print(
            f"error: --root-dict {args.root_dict} not found. Run scripts/export_root_dict.sh "
            f"or set --root-dict to a valid path.",
            file=sys.stderr,
        )
        return 2

    print(f"Output dir: {args.output_dir.resolve()}")
    print(f"Strategies: {', '.join(STRATEGIES)}")

    skipped: list[str] = []
    for strategy in STRATEGIES:
        queries, corpus = _strategy_outputs(args.output_dir, strategy, args)
        if not args.force and _already_done(queries, corpus):
            print(f"  · {strategy}: outputs already present — skipping (pass --force to re-run)")
            skipped.append(strategy)
            continue
        _run_strategy(strategy, args)

    if skipped:
        print(f"\nSkipped {len(skipped)}/{len(STRATEGIES)}: {', '.join(skipped)}")

    return _post_flight_checks(args)


if __name__ == "__main__":
    raise SystemExit(main())
