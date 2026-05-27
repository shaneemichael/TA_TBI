"""Drive the BM25 sweep across all 5 preprocessing conditions.

Owned by Person B. For each strategy in order, this script:
  1. Stages a per-strategy indexing directory containing only the corpus
     JSONL (Pyserini's ``JsonCollection`` expects a directory, not a file;
     and we never want queries_dev.jsonl indexed alongside the corpus).
  2. Builds the BM25 Lucene index via Pyserini.
  3. Runs top-``--hits`` retrieval on the prepared dev queries.
  4. Evaluates the run against the qrels and writes a per-query CSV.

After the Keep baseline finishes (the very first strategy in the sweep),
the script gates on the published MIRACL Indonesian BM25 nDCG@10 number.
A miss of more than ±2 nDCG@10 points halts the sweep with a non-zero
exit code, so the team can re-check the data prep / qrels / index config
before burning compute on the remaining four conditions.

The sweep is idempotent: a condition is skipped if its per-query metric
CSV already exists. Pass ``--force-condition <name>`` to re-run a single
condition (e.g., after fixing it), or ``--force`` to rebuild everything.

Usage (after Person A's run_preprocessing_sweep.py and extract_qrels):

    python scripts/run_bm25_pipeline.py \\
        --processed-dir data/processed \\
        --qrels qrels/qrels_dev.txt \\
        --index-root artifacts/indexes/bm25 \\
        --run-dir results/runs/bm25 \\
        --metric-dir results/metrics/bm25 \\
        --threads 8

Skip the sanity gate (NOT recommended; only useful for debugging local
indices) with ``--skip-sanity-gate``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# Sweep order: Keep first so the sanity gate can halt early.
STRATEGIES: tuple[str, ...] = (
    "keep",
    "naive_strip",
    "sastrawi_clitic",
    "sentinel",
    "rule_resolved",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=Path("artifacts/indexes/bm25"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/runs/bm25"))
    parser.add_argument("--metric-dir", type=Path, default=Path("results/metrics/bm25"))
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("artifacts/staging/bm25"),
        help="Where per-strategy single-file 'collection' dirs are staged for Pyserini.",
    )
    parser.add_argument("--query-split", default="dev")
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hits", type=int, default=1000)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument(
        "--language",
        default="id",
        help=(
            "Anserini language code threaded into both build_index and run_retrieval "
            "(default: id). Forgetting this drops nDCG@10 by ~5 points because "
            "Pyserini falls back to English Porter stemming on Indonesian text."
        ),
    )

    parser.add_argument(
        "--reference-ndcg10",
        type=float,
        help="Override the published MIRACL BM25 nDCG@10 reference (default: 0.449).",
    )
    parser.add_argument(
        "--sanity-tolerance",
        type=float,
        help="Override the ±tolerance on the sanity gate, in nDCG@10 absolute points (default: 0.02).",
    )
    parser.add_argument(
        "--skip-sanity-gate",
        action="store_true",
        help="Run all 5 conditions even if Keep misses the published reference.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every condition even if its per-query CSV already exists.",
    )
    parser.add_argument(
        "--force-condition",
        action="append",
        default=[],
        choices=list(STRATEGIES),
        help="Force re-run of one specific condition (repeatable).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for subprocess invocations (defaults to current).",
    )
    return parser


def _strategy_paths(args: argparse.Namespace, strategy: str) -> dict[str, Path]:
    """All on-disk paths for one strategy. Keeps the layout in one place."""

    return {
        "corpus_jsonl": args.processed_dir / strategy / f"corpus_{args.corpus_split}.jsonl",
        "queries_jsonl": args.processed_dir / strategy / f"queries_{args.query_split}.jsonl",
        "staging_dir": args.staging_dir / strategy,
        "index_dir": args.index_root / strategy,
        "run_file": args.run_dir / f"bm25__{strategy}.txt",
        "metric_csv": args.metric_dir / f"bm25__{strategy}.csv",
    }


def _stage_corpus(corpus_jsonl: Path, staging_dir: Path) -> None:
    """Create ``staging_dir`` containing exactly the corpus JSONL.

    Pyserini's JsonCollection scans the directory, so the staging dir must
    not contain queries_dev.jsonl (which would get indexed as a passage).
    Hardlinks are tried first to avoid duplicating ~300MB of corpus on disk;
    we fall back to a copy if hardlink fails (cross-filesystem, no support).
    """

    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / corpus_jsonl.name
    if target.exists():
        return
    try:
        os.link(corpus_jsonl, target)
    except OSError:
        shutil.copy2(corpus_jsonl, target)


def _run(command: Sequence[str]) -> None:
    """Echo the command then run it; raise on non-zero exit."""

    printable = " ".join(command)
    print(f"  $ {printable}")
    subprocess.run(list(command), check=True)


def _run_condition(strategy: str, args: argparse.Namespace) -> Path:
    """Build → search → evaluate one strategy. Returns the per-query CSV path."""

    paths = _strategy_paths(args, strategy)
    print(f"\n=== {strategy} ===")

    for required in ("corpus_jsonl", "queries_jsonl"):
        if not paths[required].exists():
            raise SystemExit(
                f"Missing {required} for {strategy!r}: {paths[required]} — "
                f"run scripts/run_preprocessing_sweep.py first."
            )

    _stage_corpus(paths["corpus_jsonl"], paths["staging_dir"])

    paths["index_dir"].mkdir(parents=True, exist_ok=True)
    paths["run_file"].parent.mkdir(parents=True, exist_ok=True)
    paths["metric_csv"].parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            args.python, "-m", "nya_ir.cli.build_index",
            "--retriever", "bm25",
            "--collection-dir", str(paths["staging_dir"]),
            "--index-dir", str(paths["index_dir"]),
            "--threads", str(args.threads),
            "--language", args.language,
            "--execute",
        ]
    )

    run_id = f"bm25__{strategy}"
    _run(
        [
            args.python, "-m", "nya_ir.cli.run_retrieval",
            "--retriever", "bm25",
            "--index-dir", str(paths["index_dir"]),
            "--queries", str(paths["queries_jsonl"]),
            "--output", str(paths["run_file"]),
            "--run-id", run_id,
            "--hits", str(args.hits),
            "--k1", str(args.k1),
            "--b", str(args.b),
            "--language", args.language,
        ]
    )

    _run(
        [
            args.python, "-m", "nya_ir.cli.evaluate_runs",
            "--qrels", str(args.qrels),
            "--run", str(paths["run_file"]),
            "--condition", run_id,
            "--output", str(paths["metric_csv"]),
        ]
    )

    return paths["metric_csv"]


def _check_keep_sanity(args: argparse.Namespace, keep_csv: Path) -> None:
    """Run the Person-B sanity gate on the Keep baseline CSV.

    The gate is symmetric (a surprisingly high observed nDCG@10 also halts).
    On failure, we exit non-zero before touching the remaining four conditions,
    so debugging starts from a clean failure rather than five wasted indices.
    """

    from nya_ir.analysis.sanity import (
        DEFAULT_TOLERANCE,
        PUBLISHED_MIRACL_BM25_NDCG10,
        check_keep_baseline_from_csv,
    )

    reference = args.reference_ndcg10 if args.reference_ndcg10 is not None else PUBLISHED_MIRACL_BM25_NDCG10
    tolerance = args.sanity_tolerance if args.sanity_tolerance is not None else DEFAULT_TOLERANCE

    result = check_keep_baseline_from_csv(
        keep_csv,
        reference=reference,
        tolerance=tolerance,
    )
    print("\n=== sanity gate ===")
    print(f"  {result.explain()}")
    if not result.passed:
        raise SystemExit(
            "Keep BM25 baseline missed the published MIRACL reference by more than the "
            "configured tolerance. Halting the sweep — re-check the data prep, qrels "
            "alignment, and BM25 params before continuing. Override with "
            "--reference-ndcg10 / --sanity-tolerance if you have a defensible reason, "
            "or use --skip-sanity-gate to bypass entirely."
        )


def _should_skip(metric_csv: Path, strategy: str, args: argparse.Namespace) -> bool:
    if args.force:
        return False
    if strategy in set(args.force_condition):
        return False
    return metric_csv.exists() and metric_csv.stat().st_size > 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.qrels.exists():
        raise SystemExit(f"qrels file not found: {args.qrels}")
    if not args.processed_dir.exists():
        raise SystemExit(f"processed-dir not found: {args.processed_dir}")

    print(f"Processed dir: {args.processed_dir}")
    print(f"qrels:         {args.qrels}")
    print(f"Sweep order:   {', '.join(STRATEGIES)}")

    for strategy in STRATEGIES:
        paths = _strategy_paths(args, strategy)
        if _should_skip(paths["metric_csv"], strategy, args):
            print(f"\n=== {strategy} === (skipped — {paths['metric_csv']} exists)")
        else:
            _run_condition(strategy, args)

        if strategy == "keep" and not args.skip_sanity_gate:
            _check_keep_sanity(args, paths["metric_csv"])

    print("\nDone. Per-query metric CSVs:")
    for strategy in STRATEGIES:
        print(f"  {strategy:>16}: {_strategy_paths(args, strategy)['metric_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
