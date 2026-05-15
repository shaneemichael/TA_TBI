"""Run lightweight statistical summaries over per-query metric CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, nargs="+")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to analyze result CSV files.") from exc
    frame = pd.concat((pd.read_csv(path) for path in args.metrics), ignore_index=True)
    summary = (
        frame.groupby("condition", as_index=False)[args.metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print(f"Wrote summary to {args.output}")
    else:
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
