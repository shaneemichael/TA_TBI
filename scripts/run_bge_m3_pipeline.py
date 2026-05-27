"""Drive the BGE-M3 dense sweep across all 5 preprocessing conditions.

For each strategy in order, this script:
  1. Encodes the prepared corpus JSONL with BGE-M3.
  2. Builds a FAISS HNSW index.
  3. Runs top-k dense retrieval for the prepared dev queries.
  4. Evaluates the run against qrels and writes per-query metrics.

Keep runs first. Its mean nDCG@10 is sanity-gated against the published
BGE-M3 Dense MIRACL-id dev number (0.561 by default). If the gate fails, the
remaining four conditions are not run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

STRATEGIES: tuple[str, ...] = (
    "keep",
    "naive_strip",
    "sastrawi_clitic",
    "sentinel",
    "rule_resolved",
)
METRIC_COLUMNS: tuple[str, ...] = (
    "ndcg@1",
    "ndcg@10",
    "recall@10",
    "mrr@100",
    "recall@100",
)
PUBLISHED_BGE_M3_DENSE_NDCG10 = 0.561
DEFAULT_SANITY_TOLERANCE = 0.02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=Path("artifacts/indexes/bge_m3"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/runs/bge_m3"))
    parser.add_argument("--metric-dir", type=Path, default=Path("results/metrics/bge_m3"))
    parser.add_argument("--report-dir", type=Path, default=Path("results/reports/bge_m3"))
    parser.add_argument("--archive-dir", type=Path, default=Path("results/archive"))
    parser.add_argument("--query-split", default="dev")
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--hits", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated dense encoding devices for indexing, e.g. "
            "'cuda:0,cuda:1'. Omit to use library defaults."
        ),
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--reference-ndcg10",
        type=float,
        default=PUBLISHED_BGE_M3_DENSE_NDCG10,
        help="Published BGE-M3 Dense MIRACL-id dev nDCG@10 reference.",
    )
    parser.add_argument(
        "--sanity-tolerance",
        type=float,
        default=DEFAULT_SANITY_TOLERANCE,
        help="Absolute nDCG@10 tolerance around the published reference.",
    )
    parser.add_argument(
        "--skip-sanity-gate",
        action="store_true",
        help="Run all 5 conditions even if Keep misses the published reference.",
    )
    parser.add_argument("--skip-archive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force-condition",
        action="append",
        default=[],
        choices=list(STRATEGIES),
        help="Force re-run of one specific strategy; repeatable.",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser


def _strategy_paths(args: argparse.Namespace, strategy: str) -> dict[str, Path]:
    return {
        "corpus_jsonl": args.processed_dir / strategy / f"corpus_{args.corpus_split}.jsonl",
        "queries_jsonl": args.processed_dir / strategy / f"queries_{args.query_split}.jsonl",
        "index_dir": args.index_root / strategy,
        "run_file": args.run_dir / f"bge_m3__{strategy}.txt",
        "metric_csv": args.metric_dir / f"bge_m3__{strategy}.csv",
        "summary_csv": args.report_dir / f"bge_m3__{strategy}_summary.csv",
        "index_metadata": args.index_root / strategy / "metadata.json",
    }


def _run(command: Sequence[str]) -> None:
    print(f"  $ {' '.join(command)}")
    subprocess.run(list(command), check=True)


def _run_condition(strategy: str, args: argparse.Namespace) -> Path:
    paths = _strategy_paths(args, strategy)
    print(f"\n=== {strategy} ===")
    for required in ("corpus_jsonl", "queries_jsonl"):
        if not paths[required].exists():
            raise SystemExit(
                f"Missing {required} for {strategy!r}: {paths[required]} -- "
                "run scripts/run_preprocessing_sweep.py first."
            )

    paths["index_dir"].mkdir(parents=True, exist_ok=True)
    paths["run_file"].parent.mkdir(parents=True, exist_ok=True)
    paths["metric_csv"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_csv"].parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            args.python,
            "-m",
            "nya_ir.cli.build_index",
            "--retriever",
            "bge_m3",
            "--collection-dir",
            str(paths["corpus_jsonl"]),
            "--index-dir",
            str(paths["index_dir"]),
            "--threads",
            str(args.threads),
            "--batch-size",
            str(args.batch_size),
            "--model-name",
            args.model_name,
            "--max-length",
            str(args.max_length),
            *([] if not args.devices else ["--devices", args.devices]),
            "--hnsw-m",
            str(args.hnsw_m),
            "--ef-construction",
            str(args.ef_construction),
            "--ef-search",
            str(args.ef_search),
            "--show-progress",
            "--execute",
        ]
    )

    run_id = f"bge_m3__{strategy}"
    _run(
        [
            args.python,
            "-m",
            "nya_ir.cli.run_retrieval",
            "--retriever",
            "bge_m3",
            "--index-dir",
            str(paths["index_dir"]),
            "--queries",
            str(paths["queries_jsonl"]),
            "--output",
            str(paths["run_file"]),
            "--run-id",
            run_id,
            "--hits",
            str(args.hits),
            "--batch-size",
            str(args.batch_size),
            "--model-name",
            args.model_name,
            "--max-length",
            str(args.max_length),
            "--ef-search",
            str(args.ef_search),
        ]
    )

    _run(
        [
            args.python,
            "-m",
            "nya_ir.cli.evaluate_runs",
            "--qrels",
            str(args.qrels),
            "--run",
            str(paths["run_file"]),
            "--condition",
            run_id,
            "--output",
            str(paths["metric_csv"]),
            "--summary-output",
            str(paths["summary_csv"]),
        ]
    )
    return paths["metric_csv"]


def _should_skip(metric_csv: Path, strategy: str, args: argparse.Namespace) -> bool:
    if args.force:
        return False
    if strategy in set(args.force_condition):
        return False
    return metric_csv.exists() and metric_csv.stat().st_size > 0


def _check_keep_sanity(args: argparse.Namespace, keep_csv: Path) -> None:
    from nya_ir.analysis.sanity import check_keep_baseline, load_keep_baseline_ndcg10

    result = check_keep_baseline(
        load_keep_baseline_ndcg10(keep_csv),
        reference=args.reference_ndcg10,
        tolerance=args.sanity_tolerance,
    )
    verdict = "PASS" if result.passed else "FAIL"
    print("\n=== sanity gate ===")
    print(
        f"  [{verdict}] Keep BGE-M3 nDCG@10 = {result.observed_mean_ndcg10:.4f} "
        f"(N={result.n_queries}); reference = {result.reference_ndcg10:.4f} "
        f"+/- {result.tolerance:.4f}; delta = {result.delta:+.4f}"
    )
    if not result.passed:
        raise SystemExit(
            "Keep BGE-M3 baseline missed the published MIRACL reference by more than the "
            "configured tolerance. Halting the sweep so the data, qrels, model config, "
            "and FAISS search settings can be checked before spending more compute."
        )


def _read_metric_csvs(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _write_all_metric_summary(metric_paths: Sequence[Path], output_path: Path) -> None:
    rows = _read_metric_csvs(metric_paths)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)

    fields = ["condition", "count"]
    for metric in METRIC_COLUMNS:
        fields += [f"{metric}_mean", f"{metric}_std", f"{metric}_missing"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, condition_rows in sorted(grouped.items()):
            out: dict[str, object] = {"condition": condition, "count": len(condition_rows)}
            for metric in METRIC_COLUMNS:
                values = []
                for row in condition_rows:
                    raw = row.get(metric, "")
                    if raw == "":
                        continue
                    values.append(float(raw))
                out[f"{metric}_mean"] = mean(values) if values else 0.0
                out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
                out[f"{metric}_missing"] = len(condition_rows) - len(values)
            writer.writerow(out)
    print(f"Wrote all-metric condition summary to {output_path}")


def _build_result_tables(metric_paths: Sequence[Path], args: argparse.Namespace) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    _write_all_metric_summary(metric_paths, args.report_dir / "condition_summary_all_metrics.csv")
    _run(
        [
            args.python,
            "-m",
            "nya_ir.cli.analyze_results",
            "--metrics",
            *[str(path) for path in metric_paths],
            "--metric",
            "ndcg@10",
            "--output",
            str(args.report_dir / "condition_summary_ndcg10.csv"),
            "--pairwise-output",
            str(args.report_dir / "pairwise_ndcg10.csv"),
            "--bootstrap-resamples",
            str(args.bootstrap_resamples),
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_files(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = args.archive_dir / f"bge_m3_run_{timestamp}.tar.gz"
    args.archive_dir.mkdir(parents=True, exist_ok=True)

    workspace = Path.cwd().resolve()

    def _archive_name(path: Path) -> Path:
        resolved = path.resolve()
        try:
            return resolved.relative_to(workspace)
        except ValueError:
            return Path(resolved.name)

    files: list[tuple[Path, Path]] = []
    for strategy in STRATEGIES:
        paths = _strategy_paths(args, strategy)
        for key in ("run_file", "metric_csv", "summary_csv", "index_metadata"):
            if paths[key].exists():
                files.append((paths[key], _archive_name(paths[key])))
    for config in sorted(Path("configs").glob("*.yaml")):
        files.append((config, _archive_name(config)))
    script_path = Path(__file__).resolve()
    files.append((script_path, _archive_name(script_path)))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command_args": {key: str(value) for key, value in vars(args).items()},
        "files": [
            {
                "path": str(relative_path),
                "bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
            }
            for source_path, relative_path in files
        ],
    }

    with tarfile.open(archive_path, "w:gz") as archive:
        for source_path, relative_path in files:
            archive.add(source_path, arcname=str(relative_path))
        payload = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(payload)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        archive.addfile(info, io.BytesIO(payload))

    print(f"Wrote reproducibility archive to {archive_path}")
    return archive_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.qrels.exists():
        raise SystemExit(f"qrels file not found: {args.qrels}")
    if not args.processed_dir.exists():
        raise SystemExit(f"processed-dir not found: {args.processed_dir}")

    print(f"Processed dir: {args.processed_dir}")
    print(f"qrels:         {args.qrels}")
    print(f"model:         {args.model_name} (max_length={args.max_length})")
    print(f"Sweep order:   {', '.join(STRATEGIES)}")

    metric_paths: list[Path] = []
    for strategy in STRATEGIES:
        paths = _strategy_paths(args, strategy)
        if _should_skip(paths["metric_csv"], strategy, args):
            print(f"\n=== {strategy} === (skipped -- {paths['metric_csv']} exists)")
        else:
            _run_condition(strategy, args)
        metric_paths.append(paths["metric_csv"])

        if strategy == "keep" and not args.skip_sanity_gate:
            _check_keep_sanity(args, paths["metric_csv"])

    print("\nDone. Per-query metric CSVs:")
    for strategy in STRATEGIES:
        print(f"  {strategy:>16}: {_strategy_paths(args, strategy)['metric_csv']}")

    _build_result_tables(metric_paths, args)
    if not args.skip_archive:
        _archive_files(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
