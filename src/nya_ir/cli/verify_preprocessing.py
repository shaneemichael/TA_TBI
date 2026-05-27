"""Verify preprocessing behavior on known ``-nya`` edge cases.

Produces the Phase 1 deliverable report table: per-strategy outputs across canonical
single-token cases (false positives + known clitics) and multi-sentence anaphoric cases
(Strategy 5 territory). Exits non-zero if any false positive is modified by the
dictionary-guarded clitic strategy.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from nya_ir.preprocessing import (
    RuleBasedNyaResolver,
    SuffixNyaRemover,
    create_sastrawi_remover,
    preprocess_keep,
    preprocess_naive_strip,
    preprocess_rule_resolved,
    preprocess_sastrawi_clitic,
    preprocess_sentinel,
)

# --- Single-token canonical cases (Phase 1 false-positive + known-root coverage) ---
# Hard false positives: words ending in -nya that are NOT -nya constructions
# at all. These are root words (or proper nouns / loanwords) that happen to
# end in the letters "nya". Any sensible dictionary-guarded strategy must
# preserve them.
#
# NOT in this list: borderline adverbial -nya forms like biasanya, karenanya,
# tahunya. Linguistically these ARE -nya constructions (biasa + -nya → "usually");
# Sastrawi correctly strips them under Path A because their roots are in the
# dictionary. We document this as intended Strategy 3 behaviour in
# test_path_a_dictionary_strips_biasanya rather than treating it as a
# false-positive failure here.
FALSE_POSITIVES = ["punya", "tanya", "hanya", "Kenya", "Sonya", "Tanya"]
POSITIVE_EXAMPLES = ["rumahnya", "bukunya", "pidatonya"]

# --- Multi-sentence anaphoric examples (Strategy 5 territory) ---
# Each entry is a short Indonesian passage that triggers (or fails to trigger) Strategy 5's
# rule-based resolver. They cover: simple anaphora, no-antecedent fallback, sentence-initial
# common-noun limitation, multiple -nya in same passage, and false-positive + true-clitic mix.
SENTENCE_EXAMPLES = [
    # 1: simple anaphoric case — proper-noun antecedent in prior sentence
    "Sukarno datang. Pidatonya terkenal.",
    # 2: false positive in same sentence as true clitic
    "Dia punya rumahnya sendiri.",
    # 3: no antecedent in window → Strategy 5 leaves -nya alone
    "Pidatonya menggemparkan dunia.",
    # 4: multiple candidate antecedents → resolver picks first (documented behavior)
    "Sukarno bertemu Hatta. Karyanya terkenal.",
    # 5: KNOWN LIMITATION — sentence-initial common noun wrongly picked as antecedent
    "Cuaca cerah. Pidatonya disampaikan.",
    # 6: KNOWN LIMITATION — chained -nya picks prior -nya form
    "Sukarno datang. Pidatonya menggemparkan. Karyanya juga.",
    # 7: stacked false positives — none should be modified by Strategy 3
    "Tanya bilang biasanya hujan.",
    # 8: passage-level mix of true clitic and false positive
    "Bukunya tentang Kenya dan Sonya.",
    # 9: anaphor across long sentence
    "Soekarno dilahirkan di Surabaya pada 1901. Pidatonya disampaikan di Jakarta.",
    # 10: -nya in middle of sentence after a proper noun
    "Hatta menulis pidato. Pidatonya juga terkenal.",
]

# Path A by default uses Sastrawi's full dictionary; the small DEFAULT_ROOTS is for the
# dependency-light test mode (--root-dict not set, --use-sastrawi-dict not set).
DEFAULT_ROOTS = {"rumah", "buku", "pidato", "karya"}


def load_root_dict(path: Path | None = None, *, use_sastrawi_dict: bool = False) -> set[str]:
    """Load the root dictionary from one of three sources, in priority order.

    1. ``--root-dict path``  → load from file
    2. ``--use-sastrawi-dict`` → load Sastrawi's bundled ~30k-word dict (Path A)
    3. otherwise              → small built-in DEFAULT_ROOTS (test mode)

    Falls back to DEFAULT_ROOTS with a warning if ``--use-sastrawi-dict`` is requested
    but PySastrawi is not installed; this prevents the CLI from crashing in a way that
    obscures the underlying missing-dependency error.
    """

    if path is not None:
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if use_sastrawi_dict:
        from nya_ir.exceptions import OptionalDependencyError
        from nya_ir.preprocessing import load_sastrawi_root_dictionary

        try:
            return set(load_sastrawi_root_dictionary())
        except OptionalDependencyError as exc:
            print(
                f"WARNING: --use-sastrawi-dict requested but {exc}. "
                f"Falling back to DEFAULT_ROOTS ({len(DEFAULT_ROOTS)} entries)."
            )
            return set(DEFAULT_ROOTS)
    return set(DEFAULT_ROOTS)


def print_table(rows: Iterable[dict[str, str]]) -> None:
    columns = ["input", "keep", "naive_strip", "sastrawi_clitic", "sentinel", "rule_resolved"]
    widths = {column: len(column) for column in columns}
    materialized = list(rows)
    for row in materialized:
        for column in columns:
            widths[column] = max(widths[column], len(row[column]))
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in materialized:
        print(" | ".join(row[column].ljust(widths[column]) for column in columns))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dict", type=Path, help="Plain-text Indonesian root dictionary.")
    parser.add_argument(
        "--use-sastrawi",
        action="store_true",
        help="Use PySastrawi remover instead of the dependency-light suffix test double.",
    )
    parser.add_argument(
        "--use-sastrawi-dict",
        action="store_true",
        help="Use Sastrawi's bundled ~30k-word dictionary as root_dict (Path A).",
    )
    return parser


def _make_row(label: str, text: str, root_dict: set[str], remover, resolver) -> dict[str, str]:
    return {
        "input": label,
        "keep": preprocess_keep(text),
        "naive_strip": preprocess_naive_strip(text),
        "sastrawi_clitic": preprocess_sastrawi_clitic(
            text, root_dict=root_dict, remover=remover
        ),
        "sentinel": preprocess_sentinel(text),
        "rule_resolved": preprocess_rule_resolved(
            text, root_dict=root_dict, resolver=resolver
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root_dict = load_root_dict(args.root_dict, use_sastrawi_dict=args.use_sastrawi_dict)
    remover = create_sastrawi_remover() if args.use_sastrawi else SuffixNyaRemover()
    resolver = RuleBasedNyaResolver(root_dict=root_dict, remover=remover)

    print("=== Single-token canonical cases ===")
    token_rows = [
        _make_row(token, token, root_dict, remover, resolver)
        for token in [*FALSE_POSITIVES, *POSITIVE_EXAMPLES]
    ]
    print_table(token_rows)

    print()
    print("=== Multi-sentence anaphoric cases (Strategy 5 territory) ===")
    sentence_rows = [
        _make_row(f"S{i+1}", text, root_dict, remover, resolver)
        for i, text in enumerate(SENTENCE_EXAMPLES)
    ]
    print_table(sentence_rows)
    print()
    for i, text in enumerate(SENTENCE_EXAMPLES):
        print(f"  S{i+1}: {text}")

    # Phase 1 success gate: the dictionary-guarded clitic strategy must preserve
    # all canonical false positives. This is independent of Strategy 5 behavior.
    changed_false_positives = [
        row for row in token_rows[: len(FALSE_POSITIVES)] if row["sastrawi_clitic"] != row["input"]
    ]
    if changed_false_positives:
        print("\nWARNING: dictionary-guarded clitic preprocessing changed false positives.")
        return 1
    print("\nOK: dictionary-guarded clitic preprocessing preserved canonical false positives.")
    print(f"     ({len(token_rows)} tokens + {len(sentence_rows)} sentences = "
          f"{len(token_rows) + len(sentence_rows)} report rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
