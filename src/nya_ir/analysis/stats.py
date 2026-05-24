"""Statistical utilities for paired retrieval metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

# Minimum sample size for Wilcoxon to be meaningfully powered. Smaller N typically
# means the caller passed aggregated condition means rather than per-query metric vectors;
# raising here surfaces that mistake fast instead of producing a misleading p-value.
_WILCOXON_MIN_N = 6

Alternative = Literal["two-sided", "less", "greater"]


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Cliff's delta for two paired or unpaired samples."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("cliffs_delta requires non-empty samples")
    greater = sum(float(a > b) for a in x_arr for b in y_arr)
    less = sum(float(a < b) for a in x_arr for b in y_arr)
    return (greater - less) / (x_arr.size * y_arr.size)


def bootstrap_mean_delta_ci(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for mean paired delta ``treatment - baseline``."""

    baseline_arr = np.asarray(baseline, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=float)
    if baseline_arr.shape != treatment_arr.shape:
        raise ValueError("baseline and treatment must have the same shape")
    deltas = treatment_arr - baseline_arr
    rng = np.random.default_rng(seed)
    sampled = rng.choice(deltas, size=(n_resamples, deltas.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(sampled, alpha)), float(np.quantile(sampled, 1.0 - alpha)))


def friedman_test(*samples: Sequence[float]) -> tuple[float, float]:
    """Run Friedman's test lazily via SciPy."""

    try:
        from scipy.stats import friedmanchisquare
    except ImportError as exc:  # pragma: no cover - scipy is core dependency
        raise RuntimeError("scipy is required for friedman_test") from exc
    statistic, p_value = friedmanchisquare(*samples)
    return float(statistic), float(p_value)


def wilcoxon_signed_rank(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    alternative: Alternative = "two-sided",
) -> tuple[float, float]:
    """Run paired Wilcoxon signed-rank lazily via SciPy.

    ``alternative`` accepts SciPy's standard ``"two-sided" | "less" | "greater"``.
    The H2 pre-registration calls for one-tailed ``"less"`` (Naive < Keep); callers
    must pass it explicitly — the function default stays ``"two-sided"`` so generic
    pairwise comparisons remain conservative.

    Per-query guard: raises ``ValueError`` if either input is shorter than
    ``_WILCOXON_MIN_N=6`` (a strong heuristic that the caller has aggregated by
    condition rather than passing per-query metric vectors).
    """

    if len(baseline) < _WILCOXON_MIN_N or len(treatment) < _WILCOXON_MIN_N:
        raise ValueError(
            f"wilcoxon_signed_rank requires per-query metric vectors with N >= "
            f"{_WILCOXON_MIN_N}; got len(baseline)={len(baseline)}, "
            f"len(treatment)={len(treatment)}. Did you accidentally pass aggregated "
            f"condition means instead of per-query metrics?"
        )

    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:  # pragma: no cover - scipy is core dependency
        raise RuntimeError("scipy is required for wilcoxon_signed_rank") from exc
    statistic, p_value = wilcoxon(baseline, treatment, alternative=alternative)
    return float(statistic), float(p_value)

