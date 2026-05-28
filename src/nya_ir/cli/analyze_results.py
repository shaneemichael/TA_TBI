"""Run lightweight statistical summaries over per-query metric CSV files."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from nya_ir.analysis.stats import (
    Alternative,
    bootstrap_mean_delta_ci,
    cliffs_delta,
    wilcoxon_signed_rank,
)

METRIC_COLUMNS = ("ndcg@1", "ndcg@10", "recall@10", "mrr@100", "recall@100")

# Pre-registered directional hypotheses. Keys are sorted (left, right) condition tuples
# matching the iteration order of `combinations(sorted(unique_conditions), 2)`; values
# are SciPy's `alternative` argument. H2 predicts Naive strip < Keep (one-tailed less)
# on both retrievers. Add new directional pairs here as new hypotheses are pre-registered.
DEFAULT_DIRECTIONAL_PAIRS: dict[tuple[str, str], Alternative] = {
    ("bm25__keep", "bm25__naive_strip"): "less",
    ("bge_m3__keep", "bge_m3__naive_strip"): "less",
    # H4: Rule-resolved is predicted to outperform the best non-resolving
    # strategy per retriever once results are available. With the current
    # 10-condition handoff, those comparison targets are Sastrawi for BM25 and
    # Sentinel for BGE-m3. Tuple order follows sorted condition order.
    ("bm25__rule_resolved", "bm25__sastrawi_clitic"): "greater",
    ("bge_m3__rule_resolved", "bge_m3__sentinel"): "greater",
}


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
        "--friedman-output",
        type=Path,
        help="Optional H1 Friedman omnibus CSV output, one row per retriever family.",
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        help="Sensitivity annotation CSV with query_id and tertile columns for H5.",
    )
    parser.add_argument(
        "--stratified-output",
        type=Path,
        help="Optional H5 stratified pairwise CSV output. Requires --sensitivity.",
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


def _retriever_family(condition: str) -> str:
    """Extract retriever name from a condition id like 'bm25__keep' → 'bm25'.

    Falls back to ``"_global_"`` if the condition does not follow the
    ``{retriever}__{strategy}`` convention from ``ExperimentCondition.run_id``.
    Used to stratify Bonferroni correction per retriever family — the
    methodology blueprint pre-commits to correction within each retriever, not
    across the full conditions matrix.
    """

    parts = condition.split("__", 1)
    return parts[0] if len(parts) == 2 else "_global_"


def _pairwise(
    frame,
    metric: str,
    *,
    bootstrap_resamples: int,
    directional_pairs: dict[tuple[str, str], Alternative] | None = None,
    apply_bonferroni: bool = True,
):
    """Pairwise comparisons across unique conditions in ``frame``.

    Comparisons are **stratified by retriever family** (parsed from the
    ``{retriever}__{strategy}`` convention); cross-retriever pairs are skipped
    because the methodology pre-registers H3 as a separate architecture-interaction
    analysis, not as Bonferroni-corrected pairwise.

    Bonferroni correction is applied within each retriever family: each pair's
    ``bonferroni_p_value`` is ``min(1.0, raw_p_value * pairs_in_family)``. Set
    ``apply_bonferroni=False`` to skip (useful for diagnostic runs).

    ``directional_pairs`` maps ``(sorted-left, sorted-right)`` condition tuples to the
    Wilcoxon ``alternative``; unmapped pairs default to ``"two-sided"``. Pre-registered
    directional hypotheses (e.g., H2: Naive < Keep) must appear here to get the one-tailed
    p-values the methodology blueprint commits to.
    """

    import pandas as pd

    directional_pairs = directional_pairs or {}
    rows: list[dict[str, object]] = []

    # Group conditions by retriever family; only emit within-family pairs.
    unique_conditions = sorted(frame["condition"].unique())
    families: dict[str, list[str]] = {}
    for condition in unique_conditions:
        families.setdefault(_retriever_family(condition), []).append(condition)

    for family_name, family_conditions in families.items():
        family_pairs = list(combinations(family_conditions, 2))
        num_family_pairs = len(family_pairs)
        for left, right in family_pairs:
            alternative: Alternative = directional_pairs.get((left, right), "two-sided")
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
                        "retriever_family": family_name,
                        "condition_a": left,
                        "condition_b": right,
                        "metric": metric,
                        "alternative": alternative,
                        "paired_queries": 0,
                        "mean_delta_b_minus_a": None,
                        "wilcoxon_statistic": None,
                        "wilcoxon_p_value": None,
                        "wilcoxon_bonferroni_p_value": None,
                        "bonferroni_family_size": num_family_pairs,
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
                try:
                    statistic, p_value = wilcoxon_signed_rank(
                        baseline, treatment, alternative=alternative
                    )
                except ValueError:
                    statistic, p_value = None, None
            bonferroni_p = None if p_value is None else (
                min(1.0, p_value * num_family_pairs) if apply_bonferroni else p_value
            )
            ci_low, ci_high = bootstrap_mean_delta_ci(
                baseline,
                treatment,
                n_resamples=bootstrap_resamples,
            )
            rows.append(
                {
                    "retriever_family": family_name,
                    "condition_a": left,
                    "condition_b": right,
                    "metric": metric,
                    "alternative": alternative,
                    "paired_queries": len(paired),
                    "mean_delta_b_minus_a": float(delta.mean()),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                    "wilcoxon_bonferroni_p_value": bonferroni_p,
                    "bonferroni_family_size": num_family_pairs,
                    "cliffs_delta": cliffs_delta(treatment, baseline),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def _friedman(frame, metric: str):
    import pandas as pd

    rows: list[dict[str, object]] = []
    for family_name, family_frame in frame.groupby(frame["condition"].map(_retriever_family)):
        pivot = family_frame.pivot_table(
            index="query_id",
            columns="condition",
            values=metric,
            aggfunc="first",
        ).dropna()
        conditions = sorted(pivot.columns)
        if len(conditions) < 3 or len(pivot) == 0:
            rows.append(
                {
                    "retriever_family": family_name,
                    "metric": metric,
                    "conditions": "|".join(conditions),
                    "paired_queries": len(pivot),
                    "friedman_statistic": None,
                    "friedman_p_value": None,
                }
            )
            continue
        from nya_ir.analysis.stats import friedman_test

        statistic, p_value = friedman_test(
            *(pivot[condition].to_numpy(dtype=float) for condition in conditions)
        )
        rows.append(
            {
                "retriever_family": family_name,
                "metric": metric,
                "conditions": "|".join(conditions),
                "paired_queries": len(pivot),
                "friedman_statistic": statistic,
                "friedman_p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def _load_sensitivity(path: Path):
    import pandas as pd

    sensitivity = pd.read_csv(path)
    required = {"query_id", "tertile"}
    missing = required - set(sensitivity.columns)
    if missing:
        raise SystemExit(
            f"Sensitivity CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    sensitivity = sensitivity.copy()
    sensitivity["query_id"] = sensitivity["query_id"].astype(str)
    return sensitivity[["query_id", "tertile"]]


def _stratified_pairwise(
    frame,
    sensitivity,
    metric: str,
    *,
    bootstrap_resamples: int,
    directional_pairs: dict[tuple[str, str], Alternative] | None = None,
):
    import pandas as pd

    merged = frame.copy()
    merged["query_id"] = merged["query_id"].astype(str)
    merged = merged.merge(sensitivity, on="query_id", how="inner")
    if merged.empty:
        raise SystemExit("No metric rows matched sensitivity annotations by query_id.")

    outputs = []
    for tertile in ("low", "mid", "high"):
        subset = merged[merged["tertile"] == tertile]
        if subset.empty:
            continue
        pairwise = _pairwise(
            subset,
            metric,
            bootstrap_resamples=bootstrap_resamples,
            directional_pairs=directional_pairs,
            apply_bonferroni=True,
        )
        if not pairwise.empty:
            pairwise.insert(0, "tertile", tertile)
            outputs.append(pairwise)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


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
        pairwise = _pairwise(
            frame,
            args.metric,
            bootstrap_resamples=args.bootstrap_resamples,
            directional_pairs=DEFAULT_DIRECTIONAL_PAIRS,
            apply_bonferroni=True,
        )
        args.pairwise_output.parent.mkdir(parents=True, exist_ok=True)
        pairwise.to_csv(args.pairwise_output, index=False)
        print(f"Wrote pairwise comparisons to {args.pairwise_output}")
    if args.friedman_output:
        friedman = _friedman(frame, args.metric)
        args.friedman_output.parent.mkdir(parents=True, exist_ok=True)
        friedman.to_csv(args.friedman_output, index=False)
        print(f"Wrote Friedman omnibus tests to {args.friedman_output}")
    if args.stratified_output:
        if args.sensitivity is None:
            raise SystemExit("--stratified-output requires --sensitivity")
        sensitivity = _load_sensitivity(args.sensitivity)
        stratified = _stratified_pairwise(
            frame,
            sensitivity,
            args.metric,
            bootstrap_resamples=args.bootstrap_resamples,
            directional_pairs=DEFAULT_DIRECTIONAL_PAIRS,
        )
        args.stratified_output.parent.mkdir(parents=True, exist_ok=True)
        stratified.to_csv(args.stratified_output, index=False)
        print(f"Wrote H5 stratified pairwise comparisons to {args.stratified_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
