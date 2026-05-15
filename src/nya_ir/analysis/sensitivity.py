"""``-nya`` sensitivity scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass

NYA_SUFFIX_RE = re.compile(r"\b\w+nya\b", flags=re.IGNORECASE | re.UNICODE)


@dataclass(frozen=True, slots=True)
class NyaSensitivityWeights:
    alpha_total: int = 1
    beta_anaphoric: int = 2
    gamma_entity_referenced: int = 3


def count_nya_suffixes(text: str) -> int:
    return len(NYA_SUFFIX_RE.findall(text))


def compute_sensitivity_score(
    gold_passages_text: str,
    *,
    anaphoric_count: int = 0,
    entity_referenced_via_nya: bool = False,
    weights: NyaSensitivityWeights = NyaSensitivityWeights(),
) -> int:
    """Compute the blueprint sensitivity score for one query."""

    return (
        weights.alpha_total * count_nya_suffixes(gold_passages_text)
        + weights.beta_anaphoric * anaphoric_count
        + weights.gamma_entity_referenced * int(entity_referenced_via_nya)
    )


def tertile_label(value: float, low_cut: float, high_cut: float) -> str:
    if value <= low_cut:
        return "low"
    if value >= high_cut:
        return "high"
    return "mid"

