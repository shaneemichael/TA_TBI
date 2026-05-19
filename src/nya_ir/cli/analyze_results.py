"""Run lightweight statistical summaries over per-query metric CSV files."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from nya_ir.analysis.stats import bootstrap_mean_delta_ci, cliffs_delta, wilcoxon_signed_rank

METRIC_COLUMNS = ("ndcg@1", "ndcg@10", "recall@10", "mrr@100", "recall@100")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, nargs="+")
    parser.add_argument("--metric", default="ndcg@10", choices=METRIC_COLUMNS)
    parser.add_argument("--output", type=Path, help="Condition-level summary CSV output.")
    parser.add_argument(
        "--pairwise-output",
        type=Path,
        help="Optional pairwise Wilcoxon/Cliff/bootstrap comparison CSV output.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples for pairwise confidence intervals.",
    )
    return parser


def _load_metrics(paths: list[Path]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to analyze result CSV files.") from exc

    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    required = {"query_id", "condition", *METRIC_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Metric CSV is missing required columns: {', '.join(sorted(missing))}")
    return frame


def _summary(frame, metric: str):
    return (
        frame.groupby("condition")[metric]
        .agg(mean="mean", std="std", count="count", missing=lambda values: values.isna().sum())
        .reset_index()
        .sort_values("mean", ascending=False)
    )


def _pairwise(frame, metric: str, *, bootstrap_resamples: int):
    import pandas as pd

    rows: list[dict[str, object]] = []
    for left, right in combinations(sorted(frame["condition"].unique()), 2):
        pivot = frame[frame["condition"].isin([left, right])].pivot_table(
            index="query_id",
            columns="condition",
            values=metric,
            aggfunc="first",
        )
        paired = pivot[[left, right]].dropna()
        if paired.empty:
            rows.append(
                {
                    "condition_a": left,
                    "condition_b": right,
                    "metric": metric,
                    "paired_queries": 0,
                    "mean_delta_b_minus_a": None,
                    "wilcoxon_statistic": None,
                    "wilcoxon_p_value": None,
                    "cliffs_delta": None,
                    "bootstrap_ci_low": None,
                    "bootstrap_ci_high": None,
                }
            )
            continue

        baseline = paired[left].to_numpy(dtype=float)
        treatment = paired[right].to_numpy(dtype=float)
        delta = treatment - baseline
        if (delta == 0).all():
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon_signed_rank(baseline, treatment)
        ci_low, ci_high = bootstrap_mean_delta_ci(
            baseline,
            treatment,
            n_resamples=bootstrap_resamples,
        )
        rows.append(
            {
                "condition_a": left,
                "condition_b": right,
                "metric": metric,
                "paired_queries": len(paired),
                "mean_delta_b_minus_a": float(delta.mean()),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
                "cliffs_delta": cliffs_delta(treatment, baseline),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = _load_metrics(args.metrics)
    summary = _summary(frame, args.metric)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print(f"Wrote summary to {args.output}")
    else:
        print(summary.to_string(index=False))
    if args.pairwise_output:
        pairwise = _pairwise(frame, args.metric, bootstrap_resamples=args.bootstrap_resamples)
        args.pairwise_output.parent.mkdir(parents=True, exist_ok=True)
        pairwise.to_csv(args.pairwise_output, index=False)
        print(f"Wrote pairwise comparisons to {args.pairwise_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
