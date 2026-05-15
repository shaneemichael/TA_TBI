"""Rule-based ``-nya`` resolver skeleton.

The implementation is intentionally conservative and dependency-light. It provides the
pipeline shape required by the experiment while keeping POS tagging and antecedent
selection replaceable.
"""

from __future__ import annotations

import re
from collections.abc import Container, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from nya_ir.exceptions import OptionalDependencyError
from nya_ir.preprocessing.sastrawi import PossessivePronounRemover, SuffixNyaRemover

WORD_NYA_RE = re.compile(r"\b(?P<host>[A-Za-z\u00C0-\u00FF]+)nya\b", flags=re.IGNORECASE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
CAPITALIZED_NP_RE = re.compile(r"\b[A-Z][\w\u00C0-\u00FF]*(?:\s+[A-Z][\w\u00C0-\u00FF]*)*")


class NyaFunction(str, Enum):
    """Heuristic function labels for ``-nya`` occurrences."""

    POSSESSIVE = "possessive"
    DEFINITE = "definite"
    ANAPHORIC = "anaphoric"
    NOMINALISER = "nominaliser"


@dataclass(frozen=True, slots=True)
class PosToken:
    text: str
    xpos: str | None = None
    upos: str | None = None


class PosTagger(Protocol):
    """Replaceable POS tagger interface."""

    def tag(self, text: str) -> Sequence[PosToken]:
        """Return POS-tagged tokens for ``text``."""


class StanzaPosTagger:
    """Lazy Stanza Indonesian POS tagger adapter."""

    def __init__(self, lang: str = "id", processors: str = "tokenize,pos") -> None:
        try:
            import stanza
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("stanza", "preprocessing") from exc
        self._pipeline = stanza.Pipeline(lang=lang, processors=processors, use_gpu=False)

    def tag(self, text: str) -> Sequence[PosToken]:  # pragma: no cover - optional dependency
        doc = self._pipeline(text)
        return [
            PosToken(text=word.text, xpos=getattr(word, "xpos", None), upos=getattr(word, "upos", None))
            for sentence in doc.sentences
            for word in sentence.words
        ]


@dataclass(frozen=True, slots=True)
class CliticOccurrence:
    start: int
    end: int
    token: str
    host: str
    root: str
    antecedent: str | None
    function: NyaFunction


class RuleBasedNyaResolver:
    """Best-effort automated resolver for Strategy 5.

    The current skeleton implements deterministic detection, simple functional
    classification, and replaceable POS-tagging hooks. It is suitable for smoke tests
    and small pilots; final experiments can tighten the heuristics behind the same API.
    """

    def __init__(
        self,
        root_dict: Container[str],
        *,
        remover: PossessivePronounRemover | None = None,
        pos_tagger: PosTagger | None = None,
        antecedent_window_sentences: int = 2,
    ) -> None:
        self.root_dict = {root.casefold() for root in root_dict}
        self.remover = remover or SuffixNyaRemover()
        self.pos_tagger = pos_tagger
        self.antecedent_window_sentences = antecedent_window_sentences

    def resolve(self, text: str) -> str:
        """Resolve anaphoric ``-nya`` occurrences by substituting an antecedent."""

        replacements: list[tuple[int, int, str]] = []
        for occurrence in self.detect(text):
            if occurrence.function is NyaFunction.ANAPHORIC and occurrence.antecedent:
                replacements.append(
                    (occurrence.start, occurrence.end, f"{occurrence.root} {occurrence.antecedent}")
                )

        if not replacements:
            return text

        output = text
        for start, end, replacement in reversed(replacements):
            output = f"{output[:start]}{replacement}{output[end:]}"
        return output

    def detect(self, text: str) -> list[CliticOccurrence]:
        """Detect dictionary-guarded ``-nya`` occurrences with heuristic labels."""

        occurrences: list[CliticOccurrence] = []
        for match in WORD_NYA_RE.finditer(text):
            token = match.group(0)
            root = self.remover.filter(token)
            if root.casefold() not in self.root_dict:
                continue
            antecedent = self._select_antecedent(text[: match.start()])
            function = self._classify(root=root, antecedent=antecedent)
            occurrences.append(
                CliticOccurrence(
                    start=match.start(),
                    end=match.end(),
                    token=token,
                    host=match.group("host"),
                    root=root,
                    antecedent=antecedent,
                    function=function,
                )
            )
        return occurrences

    def _classify(self, root: str, antecedent: str | None) -> NyaFunction:
        if self.pos_tagger is not None:
            tagged = self.pos_tagger.tag(root)
            if tagged and tagged[0].upos == "VERB":
                return NyaFunction.NOMINALISER
        if antecedent:
            return NyaFunction.ANAPHORIC
        return NyaFunction.POSSESSIVE

    def _select_antecedent(self, left_context: str) -> str | None:
        sentences = [m.group(0).strip() for m in SENTENCE_RE.finditer(left_context) if m.group(0).strip()]
        for sentence in reversed(sentences[-self.antecedent_window_sentences :]):
            match = CAPITALIZED_NP_RE.search(sentence)
            if match:
                return match.group(0)
        return None
