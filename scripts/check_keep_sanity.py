"""Notebook-friendly Keep-baseline sanity check wrapper.

Run this via ``python scripts/check_keep_sanity.py`` from shells or Kaggle
notebook ``!python`` cells to avoid IPython's noisy ``SystemExit`` warning that
appears when the same logic is raised directly inside a notebook Python cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nya_ir.analysis.sanity import check_keep_baseline_from_csv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Per-query metrics CSV to check.")
    parser.add_argument("--reference", type=float, required=True, help="Published nDCG@10 reference.")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Absolute tolerance window.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_keep_baseline_from_csv(
        args.csv,
        reference=args.reference,
        tolerance=args.tolerance,
    )
    print(result.explain())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
