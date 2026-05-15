"""Verify preprocessing behavior on known ``-nya`` edge cases."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from nya_ir.preprocessing import (
    SuffixNyaRemover,
    create_sastrawi_remover,
    preprocess_keep,
    preprocess_naive_strip,
    preprocess_sastrawi_clitic,
    preprocess_sentinel,
)

FALSE_POSITIVES = ["punya", "tanya", "hanya", "biasanya", "Kenya", "Sonya", "Tanya"]
POSITIVE_EXAMPLES = ["rumahnya", "bukunya", "pidatonya"]
DEFAULT_ROOTS = {"rumah", "buku", "pidato"}


def load_root_dict(path: Path | None) -> set[str]:
    if path is None:
        return set(DEFAULT_ROOTS)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def print_table(rows: Iterable[dict[str, str]]) -> None:
    columns = ["token", "keep", "naive_strip", "sastrawi_clitic", "sentinel"]
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
        help="Use PySastrawi instead of the dependency-light suffix remover test double.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root_dict = load_root_dict(args.root_dict)
    remover = create_sastrawi_remover() if args.use_sastrawi else SuffixNyaRemover()

    rows: list[dict[str, str]] = []
    for token in [*FALSE_POSITIVES, *POSITIVE_EXAMPLES]:
        rows.append(
            {
                "token": token,
                "keep": preprocess_keep(token),
                "naive_strip": preprocess_naive_strip(token),
                "sastrawi_clitic": preprocess_sastrawi_clitic(
                    token,
                    root_dict=root_dict,
                    remover=remover,
                ),
                "sentinel": preprocess_sentinel(token),
            }
        )
    print_table(rows)

    changed_false_positives = [
        row for row in rows[: len(FALSE_POSITIVES)] if row["sastrawi_clitic"] != row["token"]
    ]
    if changed_false_positives:
        print("\nWARNING: dictionary-guarded clitic preprocessing changed false positives.")
        return 1
    print("\nOK: dictionary-guarded clitic preprocessing preserved canonical false positives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

