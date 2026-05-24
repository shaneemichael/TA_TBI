"""Score MIRACL-id dev queries on the H5 ``-nya`` sensitivity rubric.

For each query in the input, this CLI:
  1. Concatenates its ``positive_passages`` (title + text) into a single gold
     text blob.
  2. Heuristically extracts a queried entity (the longest non-stopword
     capitalised noun phrase in the query text).
  3. Calls :func:`nya_ir.analysis.sensitivity.compute_sensitivity_score` with
     the frozen α=1 / β=2 / γ=3 weights from the methodology blueprint.
  4. Writes a CSV row per evaluable query.

After per-query scoring, the script computes tertile cut-points from the
distribution of scores and adds a ``tertile`` column (``low``/``mid``/``high``)
that Person D's H5 stratified pairwise consumes.

Queries with zero ``positive_passages`` are silently skipped — they're
non-evaluable and would only produce noise in the tertile cut-points.

Two input modes, mirroring extract_qrels:
  * Default → pulls ``miracl/miracl`` from HuggingFace (requires datasets<4.0).
  * ``--queries-jsonl PATH`` → reads rows from a local JSONL snapshot.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from nya_ir.analysis.sensitivity import compute_sensitivity_score, tertile_label
from nya_ir.data.io import read_jsonl
from nya_ir.exceptions import OptionalDependencyError

# Capitalised NP recogniser. Mirrors the resolver's antecedent search so the
# γ-term's "queried entity" candidates use the same notion of entity-shape as
# the resolver's antecedent candidates.
_CAPITALIZED_NP_RE = re.compile(r"\b[A-Z][\wÀ-ÿ]*(?:\s+[A-Z][\wÀ-ÿ]*)*\b")

# Indonesian function words that often appear sentence-initially in MIRACL-id
# queries (capitalised by virtue of being first). They are not entities and
# we don't want them dominating the γ-term match. Add new ones cautiously —
# false negatives (missing an entity) are worse than false positives (matching
# on a stopword) because the former silently zeros γ for the whole query.
_QUERY_STOPWORDS: frozenset[str] = frozenset(
    {
        "Apa",
        "Siapa",
        "Kapan",
        "Mengapa",
        "Bagaimana",
        "Berapa",
        "Mana",
        "Di",
        "Dari",
        "Ke",
        "Yang",
        "Pada",
        "Untuk",
        "Dengan",
        "Dalam",
        "Adalah",
        "Akan",
        "Itu",
        "Ini",
    }
)


def extract_queried_entity(query: str) -> str | None:
    """Best-effort extract the queried entity from a MIRACL-id query string.

    Heuristic: scan all capitalised noun phrases; if a span starts with an
    Indonesian question stopword (Apa/Siapa/etc.), strip that leading token;
    return the longest remaining span. Returns ``None`` if no qualifying NP
    is found — in that case the γ-term of the sensitivity score will be 0.

    The heuristic is documented as a known limitation in the paper's H5
    discussion: it favours specificity (longest span) but cannot disambiguate
    "Indonesia" in "What is the capital of Indonesia?" from "Indonesia" in
    "When was Indonesia founded?" — both produce the same entity. That's
    fine because the γ-term is binary (referenced via -nya or not).
    """
    candidates: list[str] = []
    for match in _CAPITALIZED_NP_RE.finditer(query):
        span = match.group(0)
        tokens = span.split()
        if tokens and tokens[0] in _QUERY_STOPWORDS:
            tokens = tokens[1:]
            if not tokens:
                continue
            span = " ".join(tokens)
        candidates.append(span)
    if not candidates:
        return None
    # Longest by character length; on ties, first wins (Python sort is stable
    # and max() is left-biased on ties).
    return max(candidates, key=len)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--language", default="id")
    parser.add_argument(
        "--queries-jsonl",
        type=Path,
        help="Local MIRACL-shape query JSONL (with positive_passages embedded). Used by tests.",
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
    # trust_remote_code=True is load-bearing; see src/nya_ir/data/miracl.py.
    dataset = load_dataset(
        "miracl/miracl", args.language, split=args.split, trust_remote_code=True
    )
    for row in dataset:
        yield row


def _gold_text(passages: Iterable[Mapping[str, object]]) -> str:
    """Concatenate (title + \\n + text) for each positive passage, blank-line-joined."""
    chunks: list[str] = []
    for passage in passages:
        title = str(passage.get("title") or "").strip()
        text = str(passage.get("text") or "").strip()
        if title and text:
            chunks.append(f"{title}\n{text}")
        elif text:
            chunks.append(text)
        elif title:
            chunks.append(title)
    return "\n\n".join(chunks)


def _score_one(row: Mapping[str, object]) -> dict[str, object] | None:
    """Return the per-query score dict, or None if the row is not evaluable."""
    qid = row.get("query_id") or row.get("id")
    if qid is None:
        return None
    positives = row.get("positive_passages") or []
    if not positives:
        return None
    valid_passages = [p for p in positives if isinstance(p, Mapping)]
    if not valid_passages:
        return None

    gold = _gold_text(valid_passages)
    query_text = str(row.get("query") or row.get("text") or "")
    entity = extract_queried_entity(query_text)

    # We call the high-level compute_sensitivity_score so any future tweak to
    # the formula (weight changes, β-term refinement) lives in one place.
    score = compute_sensitivity_score(gold, queried_entity=entity)

    # We also break out the components for downstream auditing; they're
    # recomputed cheaply by re-calling the helpers, which keeps this CLI free
    # of any "private decomposition" coupling with sensitivity.py.
    from nya_ir.analysis.sensitivity import (
        count_anaphoric_nya,
        count_nya_suffixes,
        queried_entity_referenced_via_nya,
    )

    return {
        "query_id": str(qid),
        "nya_total": count_nya_suffixes(gold),
        "nya_anaphoric": count_anaphoric_nya(gold),
        "entity_referenced": int(queried_entity_referenced_via_nya(gold, entity)),
        "sensitivity_score": score,
    }


def _assign_tertiles(rows: list[dict[str, object]]) -> None:
    """Mutate ``rows`` in place to add a ``tertile`` column.

    Cut-points are the 33rd and 67th percentiles of the score distribution.
    With heavy ties (the sensitivity score is integer-valued), the cut-points
    may collapse and one bin can get most of the mass. That's expected and
    we don't try to rebalance — the H5 analysis must work on the actual
    score distribution, not a forced 3/3/3 split.
    """
    if not rows:
        return
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a hard dep already
        raise RuntimeError("numpy required for tertile computation") from exc

    scores = np.asarray([int(row["sensitivity_score"]) for row in rows], dtype=float)
    low_cut = float(np.quantile(scores, 1.0 / 3.0))
    high_cut = float(np.quantile(scores, 2.0 / 3.0))
    for row in rows:
        row["tertile"] = tertile_label(int(row["sensitivity_score"]), low_cut, high_cut)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    rows: list[dict[str, object]] = []
    skipped = 0
    for raw in _iter_rows(args):
        scored = _score_one(raw)
        if scored is None:
            skipped += 1
            continue
        rows.append(scored)

    _assign_tertiles(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "nya_total",
        "nya_anaphoric",
        "entity_referenced",
        "sensitivity_score",
        "tertile",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Scored {len(rows)} queries ({skipped} skipped — no positives or no id).")
    if rows:
        from collections import Counter

        dist = Counter(row["tertile"] for row in rows)
        print(f"Tertile distribution: low={dist.get('low', 0)} "
              f"mid={dist.get('mid', 0)} high={dist.get('high', 0)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
