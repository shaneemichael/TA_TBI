"""Tests for the MIRACL qrels extractor CLI.

The CLI consumes the MIRACL query-row schema (which embeds ``positive_passages``
and ``negative_passages``) and emits a 4-column TREC qrels file. We exercise it
via the offline ``--queries-jsonl`` path so the test never reaches HuggingFace.
"""

from __future__ import annotations

import json
from pathlib import Path

from nya_ir.cli.extract_qrels import main as extract_qrels_main
from nya_ir.data.io import read_qrels


def _write_miracl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_extract_positives_only_by_default(tmp_path: Path) -> None:
    queries = tmp_path / "miracl_dev.jsonl"
    output = tmp_path / "qrels_dev.txt"
    _write_miracl_rows(
        queries,
        [
            {
                "query_id": "q1",
                "query": "rumahnya",
                "positive_passages": [{"docid": "d1"}, {"docid": "d3"}],
                "negative_passages": [{"docid": "d2"}],
            },
            {
                "query_id": "q2",
                "query": "pidatonya",
                "positive_passages": [{"docid": "d4"}],
                "negative_passages": [],
            },
        ],
    )

    exit_code = extract_qrels_main(
        ["--queries-jsonl", str(queries), "--output", str(output)]
    )
    assert exit_code == 0

    # Round-trip through the production qrels reader to prove the file is well-formed.
    qrels = read_qrels(output)
    assert qrels == {"q1": {"d1": 1, "d3": 1}, "q2": {"d4": 1}}


def test_extract_with_negatives_emits_relevance_zero(tmp_path: Path) -> None:
    queries = tmp_path / "miracl_dev.jsonl"
    output = tmp_path / "qrels_dev.txt"
    _write_miracl_rows(
        queries,
        [
            {
                "query_id": "q1",
                "query": "rumahnya",
                "positive_passages": [{"docid": "d1"}],
                "negative_passages": [{"docid": "d2"}],
            },
        ],
    )

    exit_code = extract_qrels_main(
        [
            "--queries-jsonl",
            str(queries),
            "--output",
            str(output),
            "--include-negatives",
        ]
    )
    assert exit_code == 0
    qrels = read_qrels(output)
    # Negatives are kept with relevance=0 so downstream evaluation can see they
    # were judged (and not just unjudged).
    assert qrels == {"q1": {"d1": 1, "d2": 0}}


def test_extract_tolerates_alternative_docid_keys(tmp_path: Path) -> None:
    """MIRACL has gone through schema churn (docid vs doc_id vs id) across releases."""
    queries = tmp_path / "miracl_dev.jsonl"
    output = tmp_path / "qrels_dev.txt"
    _write_miracl_rows(
        queries,
        [
            {
                "query_id": "q1",
                "positive_passages": [
                    {"docid": "d1"},
                    {"doc_id": "d2"},
                    {"id": "d3"},
                ],
            },
        ],
    )

    exit_code = extract_qrels_main(
        ["--queries-jsonl", str(queries), "--output", str(output)]
    )
    assert exit_code == 0
    qrels = read_qrels(output)
    assert qrels == {"q1": {"d1": 1, "d2": 1, "d3": 1}}


def test_extract_skips_rows_without_positives(tmp_path: Path) -> None:
    """A query with zero positives is non-evaluable; emit nothing but don't crash."""
    queries = tmp_path / "miracl_dev.jsonl"
    output = tmp_path / "qrels_dev.txt"
    _write_miracl_rows(
        queries,
        [
            {"query_id": "q1", "positive_passages": []},
            {"query_id": "q2", "positive_passages": [{"docid": "d1"}]},
        ],
    )

    exit_code = extract_qrels_main(
        ["--queries-jsonl", str(queries), "--output", str(output)]
    )
    assert exit_code == 0
    qrels = read_qrels(output)
    # q1 produced no rows so it's absent from the qrels file entirely —
    # evaluation will skip it via the qrels-keys filter rather than treating
    # it as a query with no gold.
    assert qrels == {"q2": {"d1": 1}}
