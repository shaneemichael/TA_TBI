"""Render the BM25 forest plot (mean delta-nDCG@10 vs Keep, with bootstrap CIs).

Reads one per-query metric CSV per condition (the format produced by
``nya_ir.cli.evaluate_runs``) and renders one PNG.

Example:
    python -m nya_ir.cli.plot_forest \\
        --baseline results/metrics/bm25__keep.csv \\
        --treatment naive_strip=results/metrics/bm25__naive_strip.csv \\
        --treatment sastrawi_clitic=results/metrics/bm25__sastrawi_clitic.csv \\
        --treatment sentinel=results/metrics/bm25__sentinel.csv \\
        --treatment rule_resolved=results/metrics/bm25__rule_resolved.csv \\
        --output results/reports/bm25_forest.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nya_ir.analysis.figures import (
    compute_effect_rows,
    read_per_query_ndcg10,
    render_forest_plot,
)


def _parse_treatment(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--treatment expects 'name=path', got {spec!r}"
        )
    name, raw_path = spec.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError(
            f"--treatment expects non-empty name and path, got {spec!r}"
        )
    return name, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Per-query metric CSV for the Keep baseline (must include ndcg@10).",
    )
    parser.add_argument(
        "--treatment",
        action="append",
        type=_parse_treatment,
        required=True,
        help="Repeatable: 'condition_name=path/to/metrics.csv'. Use one per non-baseline condition.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="BM25: delta-nDCG@10 vs Keep baseline")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.baseline.exists():
        raise SystemExit(f"Baseline CSV not found: {args.baseline}")
    baseline = read_per_query_ndcg10(args.baseline)
    if not baseline:
        raise SystemExit(
            f"Baseline CSV {args.baseline} produced zero usable ndcg@10 rows."
        )

    treatments: dict[str, dict[str, float]] = {}
    for name, path in args.treatment:
        if not path.exists():
            raise SystemExit(f"Treatment CSV not found: {path} (condition={name})")
        vec = read_per_query_ndcg10(path)
        if not vec:
            raise SystemExit(
                f"Treatment CSV {path} produced zero usable ndcg@10 rows."
            )
        treatments[name] = vec

    rows = compute_effect_rows(
        baseline,
        treatments,
        n_resamples=args.n_resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    for row in rows:
        print(
            f"  {row.condition:>16}: ΔnDCG@10 = {row.mean_delta:+.4f} "
            f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]  δ={row.cliffs_delta:+.3f}  "
            f"N={row.n_pairs}"
        )

    out = render_forest_plot(rows, args.output, title=args.title)
    print(f"Wrote forest plot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
