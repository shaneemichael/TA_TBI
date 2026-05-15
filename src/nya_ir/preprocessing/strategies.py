"""Pure preprocessing strategy functions."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Container

from nya_ir.experiment import StrategyName
from nya_ir.preprocessing.resolver import RuleBasedNyaResolver
from nya_ir.preprocessing.sastrawi import PossessivePronounRemover, create_sastrawi_remover

NYA_PATTERN = re.compile(r"(\w+?)nya\b", flags=re.UNICODE | re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize text to NFC while preserving content."""

    return unicodedata.normalize("NFC", text)


def preprocess_keep(text: str) -> str:
    """Strategy 1: identity preprocessing, with Unicode normalization."""

    return normalize_text(text)


def _strip_nya_match(match: re.Match[str]) -> str:
    return match.group(1)


def _sentinel_match(match: re.Match[str]) -> str:
    return f"{match.group(1)} <NYA>"


def preprocess_naive_strip(text: str) -> str:
    """Strategy 2: deliberately naive regex stripping of final ``nya``."""

    return NYA_PATTERN.sub(_strip_nya_match, normalize_text(text))


def preprocess_sentinel(text: str) -> str:
    """Strategy 4: replace final ``nya`` with a spaced sentinel token."""

    return NYA_PATTERN.sub(_sentinel_match, normalize_text(text))


def preprocess_sastrawi_clitic(
    text: str,
    root_dict: Container[str],
    remover: PossessivePronounRemover | None = None,
) -> str:
    """Strategy 3: Sastrawi possessive-pronoun removal with dictionary guard.

    The guard commits a stripped token only when the candidate root is present in
    ``root_dict``. Matching is case-insensitive against the root dictionary.
    """

    active_remover = remover or create_sastrawi_remover()
    roots = {root.casefold() for root in root_dict}
    output_tokens: list[str] = []

    for token in normalize_text(text).split():
        stripped = active_remover.filter(token)
        output_tokens.append(stripped if stripped.casefold() in roots else token)

    return " ".join(output_tokens)


def preprocess_rule_resolved(
    text: str,
    root_dict: Container[str],
    resolver: RuleBasedNyaResolver | None = None,
) -> str:
    """Strategy 5: rule-based anaphora resolution."""

    active_resolver = resolver or RuleBasedNyaResolver(root_dict=root_dict)
    return active_resolver.resolve(normalize_text(text))


def apply_strategy(
    text: str,
    strategy: StrategyName | str,
    *,
    root_dict: Container[str] | None = None,
    remover: PossessivePronounRemover | None = None,
    resolver: RuleBasedNyaResolver | None = None,
) -> str:
    """Apply a named strategy to one text string."""

    strategy_name = StrategyName(strategy)
    if strategy_name is StrategyName.KEEP:
        return preprocess_keep(text)
    if strategy_name is StrategyName.NAIVE_STRIP:
        return preprocess_naive_strip(text)
    if strategy_name is StrategyName.SENTINEL:
        return preprocess_sentinel(text)
    if strategy_name is StrategyName.SASTRAWI_CLITIC:
        if root_dict is None:
            raise ValueError("root_dict is required for sastrawi_clitic")
        return preprocess_sastrawi_clitic(text, root_dict=root_dict, remover=remover)
    if strategy_name is StrategyName.RULE_RESOLVED:
        if root_dict is None:
            raise ValueError("root_dict is required for rule_resolved")
        return preprocess_rule_resolved(text, root_dict=root_dict, resolver=resolver)
    raise ValueError(f"Unsupported strategy: {strategy}")

