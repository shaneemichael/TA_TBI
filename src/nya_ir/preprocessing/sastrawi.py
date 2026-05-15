"""Lazy Sastrawi integration and small test doubles."""

from __future__ import annotations

from functools import lru_cache
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


class _SastrawiCliticAdapter:
    """Adapt Sastrawi's ``RemoveInflectionalPossessivePronoun`` visitor to our Protocol.

    The visitor exposes ``remove(word)``; we expose ``filter(token)`` so it slots into the
    ``PossessivePronounRemover`` interface used by ``preprocess_sastrawi_clitic``. The visitor
    itself is dictionary-blind — it always strips word-final ``-ku/-mu/-nya``. The dictionary
    guard in ``preprocess_sastrawi_clitic`` is what enforces correctness; this adapter only
    isolates Sastrawi's clitic-removal step from the rest of its stemming pipeline (no prefix
    stripping, no other suffixes).
    """

    __slots__ = ("_visitor",)

    def __init__(self, visitor: object) -> None:
        self._visitor = visitor

    def filter(self, token: str) -> str:
        return self._visitor.remove(token)  # type: ignore[attr-defined]


def create_sastrawi_remover() -> PossessivePronounRemover:
    """Create a Sastrawi-backed clitic remover, lazily.

    Uses the ``RemoveInflectionalPossessivePronoun`` visitor from Sastrawi's implementation
    of the Asian (2007) confix-stripping algorithm — isolated to the clitic-removal step.
    The visitor's regex strips word-final ``-ku/-mu/-nya``; the dictionary guard in
    ``preprocess_sastrawi_clitic`` rejects strips that don't produce a known root.
    """

    try:
        from Sastrawi.Stemmer.Context.Visitor.RemoveInflectionalPossessivePronoun import (
            RemoveInflectionalPossessivePronoun,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise OptionalDependencyError("PySastrawi", "preprocessing") from exc
    return _SastrawiCliticAdapter(RemoveInflectionalPossessivePronoun())


@lru_cache(maxsize=1)
def load_sastrawi_root_dictionary() -> frozenset[str]:
    """Load Sastrawi's bundled ~30k-word Indonesian root dictionary (``kata-dasar.txt``).

    This is the production ``root_dict`` for Strategy 3 under Path A: use Sastrawi's own
    dictionary as the gate, mirroring what real practitioners using Sastrawi would
    experience. Loaded via Sastrawi's ``StemmerFactory.get_words()`` API rather than
    reading the file directly, so future packaging changes do not break this loader.

    Cached after the first call (the file is ~230 KB and immutable).

    Known consequence under Path A: words like ``biasanya``, ``karenanya``, ``tahunya`` are
    stripped because their roots (``biasa``, ``karena``, ``tahu``) are in the dictionary.
    Documented as a Strategy 3 limitation in the discussion section.

    Raises ``OptionalDependencyError`` if PySastrawi is not installed.
    """

    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise OptionalDependencyError("PySastrawi", "preprocessing") from exc
    words = StemmerFactory().get_words()
    return frozenset(word.strip() for word in words if word.strip())

