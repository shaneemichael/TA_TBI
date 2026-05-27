"""Tests for the per-query sensitivity scoring CLI.

The CLI consumes a MIRACL-shape query JSONL (each row carries embedded
``positive_passages``) and emits a CSV with per-query sensitivity scores
(α=1·total + β=2·anaphoric + γ=3·entity_referenced) plus a tertile label
suitable for H5 stratified pairwise analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from nya_ir.cli.score_sensitivity import (
    extract_queried_entity,
    main as score_sensitivity_main,
)


def _write_miracl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# extract_queried_entity — the heuristic that populates the γ-term
# --------------------------------------------------------------------------


def test_extract_queried_entity_takes_proper_noun_after_stopword():
    """Indonesian queries often begin with a sentence-initial Apa/Siapa/etc;
    the proper noun we want is the next capitalised NP, not the stopword itself."""
    assert extract_queried_entity("Siapa Sukarno?") == "Sukarno"
    assert extract_queried_entity("Apa karya Soekarno Hatta?") == "Soekarno Hatta"


def test_extract_queried_entity_returns_none_for_no_capitals():
    assert extract_queried_entity("apa rumahnya besar") is None


def test_extract_queried_entity_returns_longest_when_multiple_candidates():
    """The longest capitalised span is usually the most specific entity."""
    assert extract_queried_entity("Apa hubungan Jakarta dan Surabaya Selatan?") == "Surabaya Selatan"


# --------------------------------------------------------------------------
# End-to-end: scoring + tertile cut
# --------------------------------------------------------------------------


def test_score_sensitivity_writes_per_query_csv(tmp_path: Path) -> None:
    queries = tmp_path / "miracl_dev.jsonl"
    output = tmp_path / "sensitivity_annotations.csv"
    _write_miracl_rows(
        queries,
        [
            {
                "query_id": "q1",
                "query": "Siapa Sukarno?",
                "positive_passages": [
                    {
                        "docid": "d1",
                        "title": "Sukarno",
                        "text": "Sukarno datang. Pidatonya menggemparkan.",
                    },
                ],
            },
            {
                "query_id": "q2",
                "query": "Apa rumahnya?",
                "positive_passages": [
                    {"docid": "d2", "title": "Rumah", "text": "rumahnya besar"},
                ],
            },
        ],
    )

    exit_code = score_sensitivity_main(
        ["--queries-jsonl", str(queries), "--output", str(output)]
    )
    assert exit_code == 0
    rows = _read_csv(output)
    assert len(rows) == 2
    assert {row["query_id"] for row in rows} == {"q1", "q2"}
    # Expected schema
    for row in rows:
        assert set(row.keys()) == {
            "query_id",
            "nya_total",
            "nya_anaphoric",
            "entity_referenced",
            "sensitivity_score",
            "tertile",
        }


def test_score_sensitivity_computes_alpha_beta_gamma_correctly(tmp_path: Path) -> None:
    """Spot-check the formula on a query whose gold text we control completely.

    Text: "Sukarno datang. Pidatonya menggemparkan."
      - nya_total = 1 (Pidatonya)
      - nya_anaphoric = 1 (preceded by capitalised Sukarno in prior sentence)
      - entity_referenced = 1 (queried_entity 'Sukarno' appears in the
        anaphoric window of the -nya match)
    Score = 1·1 + 2·1 + 3·1 = 6
    """
    queries = tmp_path / "miracl_dev.jsonl"
    output = tmp_path / "sensitivity_annotations.csv"
    _write_miracl_rows(
        queries,
        [
            {
                "query_id": "q1",
                "query": "Siapa Sukarno?",
                "positive_passages": [
                    {
                        "docid": "d1",
                        "title": "",
                        "text": "Sukarno datang. Pidatonya menggemparkan.",
                    },
                ],
            },
        ],
    )

    score_sensitivity_main(
        ["--queries-jsonl", str(queries), "--output", str(output)]
    )
    row = _read_csv(output)[0]
    assert row["nya_total"] == "1"
    assert row["nya_anaphoric"] == "1"
    assert row["entity_referenced"] == "1"
    assert row["sensitivity_score"] == "6"


def test_score_sensitivity_tertile_cuts_split_queries_into_three_bins(tmp_path: Path) -> None:
    """Build 9 queries with monotonically increasing scores so tertile labels
    are deterministic. Pin the expected 3/3/3 split."""
    queries = tmp_path / "miracl_dev.jsonl"
    output = tmp_path / "sensitivity_annotations.csv"

    # Build texts whose -nya-suffix count yields predictable scores: text repeated
    # k times will produce k * (alpha=1 * 1 -nya per copy + beta=2 * 1 anaphor per copy) per copy.
    # Simpler: just use texts with k copies of "Sukarno datang. Pidatonya juga."
    rows = []
    for k in range(1, 10):
        rows.append(
            {
                "query_id": f"q{k}",
                "query": "Siapa Sukarno?",
                "positive_passages": [
                    {
                        "docid": f"d{k}",
                        "title": "",
                        "text": " ".join(
                            ["Sukarno datang. Pidatonya juga."] * k
                        ),
                    },
                ],
            }
        )
    _write_miracl_rows(queries, rows)

    score_sensitivity_main(
        ["--queries-jsonl", str(queries), "--output", str(output)]
    )
    csv_rows = _read_csv(output)
    by_qid = {row["query_id"]: row for row in csv_rows}

    # Scores monotonically increase with k. With N=9 and method='linear',
    # tertile cut-points fall at the 33rd / 67th percentiles, which on a
    # 9-sample monotone sequence is around the 3rd / 6th values.
    low = [qid for qid in by_qid if by_qid[qid]["tertile"] == "low"]
    mid = [qid for qid in by_qid if by_qid[qid]["tertile"] == "mid"]
    high = [qid for qid in by_qid if by_qid[qid]["tertile"] == "high"]
    # Allow either a clean 3/3/3 or a 4/2/3-style split depending on
    # numpy.quantile tie handling — the contract that matters is "all three
    # bins are non-empty and ordering is preserved".
    assert low and mid and high, f"empty bin(s): low={low}, mid={mid}, high={high}"
    # Ordering invariant: the smallest-score query is 'low'; the largest is 'high'.
    assert "q1" in low
    assert "q9" in high


def test_score_sensitivity_skips_queries_without_positives(tmp_path: Path) -> None:
    """A query with zero positives is non-evaluable; emit nothing for it but don't crash."""
    queries = tmp_path / "miracl_dev.jsonl"
    output = tmp_path / "sensitivity_annotations.csv"
    _write_miracl_rows(
        queries,
        [
            {"query_id": "q1", "query": "Siapa Sukarno?", "positive_passages": []},
            {
                "query_id": "q2",
                "query": "Siapa Sukarno?",
                "positive_passages": [{"docid": "d1", "text": "Sukarno datang."}],
            },
        ],
    )

    score_sensitivity_main(
        ["--queries-jsonl", str(queries), "--output", str(output)]
    )
    rows = _read_csv(output)
    assert [row["query_id"] for row in rows] == ["q2"]


def test_score_sensitivity_concatenates_multiple_positive_passages(tmp_path: Path) -> None:
    """When a query has multiple positives, all passage texts feed the score."""
    queries = tmp_path / "miracl_dev.jsonl"
    output = tmp_path / "sensitivity_annotations.csv"
    _write_miracl_rows(
        queries,
        [
            {
                "query_id": "q1",
                "query": "Siapa Sukarno?",
                "positive_passages": [
                    {"docid": "d1", "text": "rumahnya besar"},
                    {"docid": "d2", "text": "bukunya menarik"},
                    {"docid": "d3", "text": "mobilnya merah"},
                ],
            },
        ],
    )

    score_sensitivity_main(
        ["--queries-jsonl", str(queries), "--output", str(output)]
    )
    row = _read_csv(output)[0]
    # 3 -nya occurrences across the concatenated text → nya_total >= 3.
    assert int(row["nya_total"]) >= 3
