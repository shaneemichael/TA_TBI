"""Forest plot of paired effect sizes for the BM25 sweep.

Person B's final deliverable. Each non-Keep strategy gets one row showing its
paired mean ΔnDCG@10 versus the Keep baseline, with bootstrap-95% CI whiskers.
Cliff's δ is rendered as the marker label so the same plot communicates both
magnitude (the dot position) and direction-with-overlap (the δ symbol).

The pure-computation entry points (``compute_effect_rows``,
``read_per_query_ndcg10``) take in-memory data so they are trivially testable
without matplotlib. The plotting entry point (``render_forest_plot``) renders
to a file; tests cover it by asserting the file is non-empty on a tiny
synthetic input.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nya_ir.analysis.stats import bootstrap_mean_delta_ci, cliffs_delta

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True, slots=True)
class EffectRow:
    """One row of the forest plot: a treatment condition vs the Keep baseline."""

    condition: str
    mean_delta: float
    ci_low: float
    ci_high: float
    cliffs_delta: float
    n_pairs: int


def read_per_query_ndcg10(csv_path: str | Path) -> dict[str, float]:
    """Return ``{query_id: ndcg@10}`` from an evaluate_runs per-query CSV.

    Rows with missing or non-numeric ``ndcg@10`` are silently dropped; rows
    with missing ``query_id`` are skipped. We do NOT raise on a missing
    column here — callers (``compute_effect_rows``) need to know which file
    is broken, and we hand back an empty dict so they can surface that.
    """

    path = Path(csv_path)
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "ndcg@10" not in reader.fieldnames:
            return out
        for row in reader:
            qid = row.get("query_id")
            raw = row.get("ndcg@10", "")
            if not qid or not raw:
                continue
            try:
                out[str(qid)] = float(raw)
            except (TypeError, ValueError):
                continue
    return out


def _paired_arrays(
    baseline: dict[str, float], treatment: dict[str, float]
) -> tuple[list[float], list[float]]:
    """Inner-join two ``query_id -> metric`` maps on ``query_id``.

    Forest plot effect sizes must be paired by query, so a query missing from
    either condition is dropped from the pair (not imputed). The intersection
    size is reported as ``n_pairs`` on the resulting EffectRow.
    """

    shared = sorted(set(baseline) & set(treatment))
    return ([baseline[qid] for qid in shared], [treatment[qid] for qid in shared])


def compute_effect_rows(
    baseline: dict[str, float],
    treatments: dict[str, dict[str, float]],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> list[EffectRow]:
    """Compute one :class:`EffectRow` per treatment condition.

    ``baseline`` is the Keep per-query nDCG@10. ``treatments`` maps condition
    name → per-query nDCG@10 for each non-Keep strategy. Each row reports
    mean ΔnDCG@10 (treatment − baseline) with a paired bootstrap CI and
    the unpaired Cliff's δ of the two metric distributions.
    """

    rows: list[EffectRow] = []
    for condition, treatment in treatments.items():
        baseline_arr, treatment_arr = _paired_arrays(baseline, treatment)
        if not baseline_arr:
            raise ValueError(
                f"No paired queries between baseline and condition {condition!r}; "
                "check that both CSVs were produced from the same dev split."
            )
        deltas = [t - b for b, t in zip(baseline_arr, treatment_arr, strict=True)]
        mean_delta = sum(deltas) / len(deltas)
        ci_low, ci_high = bootstrap_mean_delta_ci(
            baseline_arr,
            treatment_arr,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        delta = cliffs_delta(treatment_arr, baseline_arr)
        rows.append(
            EffectRow(
                condition=condition,
                mean_delta=mean_delta,
                ci_low=ci_low,
                ci_high=ci_high,
                cliffs_delta=delta,
                n_pairs=len(baseline_arr),
            )
        )
    return rows


def render_forest_plot(
    rows: list[EffectRow],
    output_path: str | Path,
    *,
    title: str = "BM25: ΔnDCG@10 vs Keep baseline",
    xlabel: str = "Mean ΔnDCG@10 (paired, 95% bootstrap CI)",
    dpi: int = 200,
) -> Path:
    """Render a forest plot to ``output_path`` and return the resolved path.

    Uses ``Agg`` backend explicitly so headless machines (Pyserini boxes
    without a display) don't fail at ``import pyplot``.
    """

    if not rows:
        raise ValueError("render_forest_plot requires at least one EffectRow")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(rows) + 1.5))
    _populate_axes(ax, rows, title=title, xlabel=xlabel)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _populate_axes(
    ax: "Axes",
    rows: list[EffectRow],
    *,
    title: str,
    xlabel: str,
) -> None:
    """Draw the forest plot onto ``ax``. Pure-matplotlib; no I/O.

    Kept separate from :func:`render_forest_plot` so callers wanting to embed
    the plot in a larger figure (e.g., a multi-panel results figure) can do
    so without monkey-patching ``savefig``.
    """

    y_positions = list(range(len(rows), 0, -1))
    means = [row.mean_delta for row in rows]
    lows = [row.mean_delta - row.ci_low for row in rows]
    highs = [row.ci_high - row.mean_delta for row in rows]

    ax.errorbar(
        means,
        y_positions,
        xerr=[lows, highs],
        fmt="o",
        capsize=4,
        linewidth=1.2,
        markersize=6,
    )
    ax.axvline(0.0, linestyle="--", linewidth=0.8, color="grey")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{row.condition}\n(δ={row.cliffs_delta:+.2f}, N={row.n_pairs})" for row in rows]
    )
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
