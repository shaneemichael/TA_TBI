"""Extract a TREC-format qrels file from a MIRACL query dataset.

MIRACL's dev split embeds gold judgments inline on each query row as
``positive_passages`` (and optionally ``negative_passages``). The downstream
``evaluate_runs`` CLI expects a standard 4-column TREC qrels file
(``<query_id> 0 <doc_id> <relevance>``) on disk, so we materialise one here.

By default we emit positives only with ``relevance=1`` — this matches the
canonical MIRACL evaluation convention where unjudged docs are simply absent.
Pass ``--include-negatives`` to also emit ``negative_passages`` with
``relevance=0``; that is useful only when downstream code wants to distinguish
judged-irrelevant from unjudged (it does not change nDCG/Recall/MRR values).

Two input modes:
  * Default --- pulls ``miracl/miracl`` from HuggingFace
    (requires ``datasets<4.0`` because MIRACL is a script-based dataset).
  * ``--queries-jsonl PATH`` --- reads rows from a local JSONL file with the
    same MIRACL schema. Used by tests and by anyone working offline with a
    snapshot of the data.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path

from nya_ir.data.io import read_jsonl
from nya_ir.exceptions import OptionalDependencyError

# MIRACL has reshuffled docid keys across releases (``docid`` in v1.0 dumps,
# ``doc_id`` in some mirrors, ``id`` in others). Tolerate all three so a future
# schema bump on the Hub doesn't silently produce an empty qrels file.
_DOCID_KEYS: tuple[str, ...] = ("docid", "doc_id", "id")
_QUERYID_KEYS: tuple[str, ...] = ("query_id", "id")


def _first_string(row: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--language", default="id")
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help=(
            "Emit negative_passages with relevance=0 as well. Off by default; "
            "MIRACL's published evaluation uses positives only."
        ),
    )
    parser.add_argument(
        "--queries-jsonl",
        type=Path,
        help=(
            "Local query JSONL (MIRACL schema, with positive_passages "
            "embedded). Skips the HuggingFace fetch — used by tests and by "
            "anyone working from a snapshot."
        ),
    )
    return parser


def _iter_rows(args: argparse.Namespace) -> Iterable[Mapping[str, object]]:
    if args.queries_jsonl is not None:
        yield from read_jsonl(args.queries_jsonl)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise OptionalDependencyError("datasets", "data") from exc
    # trust_remote_code=True is load-bearing: MIRACL is script-based on the
    # Hub. See src/nya_ir/data/miracl.py for the matching note + version pin.
    dataset = load_dataset(
        "miracl/miracl", args.language, split=args.split, trust_remote_code=True
    )
    for row in dataset:
        yield row


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_queries = n_positives = n_negatives = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for row in _iter_rows(args):
            qid = _first_string(row, _QUERYID_KEYS)
            if qid is None:
                continue
            n_queries += 1
            for passage in row.get("positive_passages") or []:
                if not isinstance(passage, Mapping):
                    continue
                did = _first_string(passage, _DOCID_KEYS)
                if did is None:
                    continue
                handle.write(f"{qid} 0 {did} 1\n")
                n_positives += 1
            if not args.include_negatives:
                continue
            for passage in row.get("negative_passages") or []:
                if not isinstance(passage, Mapping):
                    continue
                did = _first_string(passage, _DOCID_KEYS)
                if did is None:
                    continue
                handle.write(f"{qid} 0 {did} 0\n")
                n_negatives += 1

    summary = f"Wrote qrels for {n_queries} queries: {n_positives} positives"
    if args.include_negatives:
        summary += f" + {n_negatives} negatives"
    print(summary)
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
