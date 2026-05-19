"""Evaluate a TREC run file against qrels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from nya_ir.data.io import read_qrels, read_trec_run
from nya_ir.evaluation.tables import condition_summary_rows, per_query_metric_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional condition-level summary CSV output.",
    )
    return parser


def write_csv(path: Path, rows: list[dict[str, object]], fallback_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else fallback_fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = per_query_metric_rows(
        read_qrels(args.qrels),
        read_trec_run(args.run),
        condition=args.condition,
    )
    write_csv(args.output, rows, ["query_id", "condition"])
    print(f"Wrote {len(rows)} per-query metric rows to {args.output}")
    if args.summary_output:
        summaries = condition_summary_rows(rows)
        write_csv(args.summary_output, summaries, ["condition", "count"])
        print(f"Wrote {len(summaries)} condition summary rows to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

