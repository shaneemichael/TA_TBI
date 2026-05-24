"""``-nya`` sensitivity scoring.

Implements the H5 stratification score from the methodology blueprint §6:

    sensitivity(q) = α · count_nya(gold)
                   + β · count_anaphoric_nya(gold)
                   + γ · 1[queried_entity_referenced_via_nya]

with α=1, β=2, γ=3 frozen prior to main analysis. The β term requires the
anaphoric-detection heuristic — without it, sensitivity collapses to total
*-nya* count (effective β=0), which is NOT what was pre-registered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NYA_SUFFIX_RE = re.compile(r"\b\w+nya\b", flags=re.IGNORECASE | re.UNICODE)

# Same conventions as the resolver's antecedent search: an anaphoric -nya is one
# preceded (within the prior 1–2 sentences) by at least one capitalised noun phrase
# that is itself NOT a -nya form. The window matches the resolver's default.
_SENTENCE_BOUNDARY_RE = re.compile(r"[^.!?]+[.!?]?")
_CAPITALIZED_NP_RE = re.compile(
    r"\b[A-Z][\wÀ-ÿ]*(?:\s+[A-Z][\wÀ-ÿ]*)*"
)
_ANAPHORIC_WINDOW_SENTENCES = 2


@dataclass(frozen=True, slots=True)
class NyaSensitivityWeights:
    alpha_total: int = 1
    beta_anaphoric: int = 2
    gamma_entity_referenced: int = 3


def count_nya_suffixes(text: str) -> int:
    return len(NYA_SUFFIX_RE.findall(text))


def count_anaphoric_nya(text: str, *, window_sentences: int = _ANAPHORIC_WINDOW_SENTENCES) -> int:
    """Count -nya occurrences that look anaphoric by the resolver's heuristic.

    A -nya is counted as anaphoric when the prior ``window_sentences`` sentences
    contain at least one capitalised NP that is itself NOT a -nya form. This is the
    same surface-context rule the Strategy 5 resolver uses to decide whether to
    substitute an antecedent (resolver.py:_select_antecedent), so the stratification
    score aligns with the actual preprocessing the experiment runs.
    """

    if not text:
        return 0

    count = 0
    for match in NYA_SUFFIX_RE.finditer(text):
        left_context = text[: match.start()]
        sentences = [
            m.group(0).strip()
            for m in _SENTENCE_BOUNDARY_RE.finditer(left_context)
            if m.group(0).strip()
        ]
        window = sentences[-window_sentences:] if window_sentences > 0 else sentences
        found_antecedent = False
        for sentence in reversed(window):
            for np_match in _CAPITALIZED_NP_RE.finditer(sentence):
                candidate = np_match.group(0)
                if candidate.lower().endswith("nya"):
                    continue  # skip -nya forms; mirrors the chained-nya sweep in resolver
                found_antecedent = True
                break
            if found_antecedent:
                break
        if found_antecedent:
            count += 1
    return count


def queried_entity_referenced_via_nya(
    text: str, queried_entity: str | None
) -> bool:
    """True when ``queried_entity`` appears as a capitalised NP within the
    anaphoric window of at least one -nya occurrence in ``text``.

    Returns False if ``queried_entity`` is None or empty — gamma-term contributors
    require an explicit entity from the query.
    """

    if not queried_entity or not queried_entity.strip() or not text:
        return False
    needle = queried_entity.strip()
    for match in NYA_SUFFIX_RE.finditer(text):
        left_context = text[: match.start()]
        sentences = [
            m.group(0).strip()
            for m in _SENTENCE_BOUNDARY_RE.finditer(left_context)
            if m.group(0).strip()
        ]
        window = sentences[-_ANAPHORIC_WINDOW_SENTENCES:]
        if any(needle in sentence for sentence in window):
            return True
    return False


def compute_sensitivity_score(
    gold_passages_text: str,
    *,
    anaphoric_count: int | None = None,
    entity_referenced_via_nya: bool = False,
    queried_entity: str | None = None,
    weights: NyaSensitivityWeights = NyaSensitivityWeights(),
) -> int:
    """Compute the blueprint sensitivity score for one query.

    ``anaphoric_count`` defaults to the heuristic count computed from
    ``gold_passages_text``; callers may override (e.g., when using gold-annotated
    anaphor labels). Passing ``anaphoric_count=0`` explicitly disables the β term
    only when the caller intends that, not by accident.

    ``queried_entity`` (string) is used to derive ``entity_referenced_via_nya``
    when the caller does not pre-compute it.
    """

    if anaphoric_count is None:
        anaphoric_count = count_anaphoric_nya(gold_passages_text)

    if queried_entity is not None and not entity_referenced_via_nya:
        entity_referenced_via_nya = queried_entity_referenced_via_nya(
            gold_passages_text, queried_entity
        )

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

