"""Sanity-gate the Keep BM25 baseline against published MIRACL numbers.

Person B's brief: run the Keep baseline first and halt the team if it misses
the published MIRACL BM25 result by more than ±2 nDCG@10 points. That gate
lives here so it can be invoked from both the orchestrator script and an
ad-hoc CLI run.

Reference number: MIRACL paper (Zhang et al., TACL 2023, "MIRACL: A Multilingual
Retrieval Dataset Covering 18 Diverse Languages"), Table 4. Indonesian dev BM25
with default Anserini parameters: nDCG@10 = 0.449. The published number is
configurable so callers can pin a different reference if the upstream benchmark
moves; the default exists so the gate has a single source of truth.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

# Published MIRACL Indonesian dev BM25 nDCG@10. Source: Zhang et al., TACL 2023.
PUBLISHED_MIRACL_BM25_NDCG10: float = 0.449

# ±tolerance on the published number, in nDCG@10 absolute points. The brief
# explicitly says "±2 nDCG@10 points" so the gate accepts [0.429, 0.469].
DEFAULT_TOLERANCE: float = 0.02


@dataclass(frozen=True, slots=True)
class SanityResult:
    """Outcome of a baseline sanity check.

    ``passed`` is the only field the orchestrator branches on; the rest exist
    so the report can show the exact numbers without recomputing.
    """

    observed_mean_ndcg10: float
    reference_ndcg10: float
    tolerance: float
    delta: float  # observed - reference (signed)
    passed: bool
    n_queries: int

    def explain(self) -> str:
        """Human-readable one-liner for logs and runbook output."""

        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] Keep BM25 nDCG@10 = {self.observed_mean_ndcg10:.4f} "
            f"(N={self.n_queries}); reference = {self.reference_ndcg10:.4f} "
            f"± {self.tolerance:.4f}; delta = {self.delta:+.4f}"
        )


def _read_ndcg10_column(csv_path: Path) -> list[float]:
    """Pull the ``ndcg@10`` column out of an evaluate_runs per-query CSV.

    Skips rows with empty or non-numeric values; those would otherwise poison
    the mean. We keep this private to discourage callers from reaching past
    :func:`load_keep_baseline_ndcg10`.
    """

    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "ndcg@10" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} does not have an 'ndcg@10' column; got fields={reader.fieldnames!r}"
            )
        for row in reader:
            raw = row.get("ndcg@10", "")
            if raw is None or raw == "":
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
    return values


def load_keep_baseline_ndcg10(csv_path: str | Path) -> list[float]:
    """Load the per-query nDCG@10 vector from the Keep BM25 evaluation CSV."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Keep baseline metrics CSV not found: {path}")
    values = _read_ndcg10_column(path)
    if not values:
        raise ValueError(f"{path} has zero usable ndcg@10 rows")
    return values


def check_keep_baseline(
    per_query_ndcg10: list[float] | tuple[float, ...],
    *,
    reference: float = PUBLISHED_MIRACL_BM25_NDCG10,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SanityResult:
    """Compare observed mean nDCG@10 against the published BM25 reference.

    The check is symmetric: failing low *or* failing high triggers a halt.
    A surprisingly *high* baseline is just as suspicious as a low one
    (e.g., qrels file scrambled so a constant retriever scores well).
    """

    if not per_query_ndcg10:
        raise ValueError("per_query_ndcg10 must be non-empty")
    observed = mean(per_query_ndcg10)
    delta = observed - reference
    return SanityResult(
        observed_mean_ndcg10=observed,
        reference_ndcg10=reference,
        tolerance=tolerance,
        delta=delta,
        passed=abs(delta) <= tolerance,
        n_queries=len(per_query_ndcg10),
    )


def check_keep_baseline_from_csv(
    csv_path: str | Path,
    *,
    reference: float = PUBLISHED_MIRACL_BM25_NDCG10,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SanityResult:
    """Convenience wrapper: load + check in one call."""

    return check_keep_baseline(
        load_keep_baseline_ndcg10(csv_path),
        reference=reference,
        tolerance=tolerance,
    )
