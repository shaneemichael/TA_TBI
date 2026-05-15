"""Lazy Sastrawi integration and small test doubles."""

from __future__ import annotations

from typing import Protocol

from nya_ir.exceptions import OptionalDependencyError


class PossessivePronounRemover(Protocol):
    """Minimal interface used from Sastrawi's possessive-pronoun filter."""

    def filter(self, token: str) -> str:
        """Return a token with a possessive pronoun removed when applicable."""


class SuffixNyaRemover:
    """Deterministic remover for unit tests and dependency-light smoke checks.

    This is not a replacement for the Sastrawi condition in final experiments. It exists so
    dictionary-guard behavior can be tested before installing optional linguistic tooling.
    """

    def filter(self, token: str) -> str:
        return token[:-3] if token.lower().endswith("nya") and len(token) > 3 else token


def create_sastrawi_remover() -> PossessivePronounRemover:
    """Create Sastrawi's possessive-pronoun remover lazily."""

    try:
        from Sastrawi.Stemmer.Filter.RemovePossessivePronoun import RemovePossessivePronoun
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise OptionalDependencyError("PySastrawi", "preprocessing") from exc
    return RemovePossessivePronoun()

