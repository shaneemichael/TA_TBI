"""Tests for the Person-B Keep-baseline sanity gate.

The gate is symmetric (rejects too-low *and* too-high observed means) and
must surface clearly-typed errors on malformed CSVs so misuse by the
orchestrator script fails fast instead of producing a misleading pass.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from nya_ir.analysis.sanity import (
    DEFAULT_TOLERANCE,
    PUBLISHED_MIRACL_BM25_NDCG10,
    SanityResult,
    check_keep_baseline,
    check_keep_baseline_from_csv,
    load_keep_baseline_ndcg10,
)


def _write_per_query_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["query_id", "condition", "ndcg@1", "ndcg@10", "recall@10", "mrr@100", "recall@100"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def test_check_keep_baseline_passes_at_reference():
    n = 50
    result = check_keep_baseline([PUBLISHED_MIRACL_BM25_NDCG10] * n)
    assert isinstance(result, SanityResult)
    assert result.passed
    assert result.n_queries == n
    assert result.delta == pytest.approx(0.0, abs=1e-9)


def test_check_keep_baseline_passes_within_tolerance():
    result = check_keep_baseline([PUBLISHED_MIRACL_BM25_NDCG10 + DEFAULT_TOLERANCE - 1e-6] * 10)
    assert result.passed


def test_check_keep_baseline_fails_below_tolerance():
    result = check_keep_baseline([PUBLISHED_MIRACL_BM25_NDCG10 - DEFAULT_TOLERANCE - 0.005] * 10)
    assert not result.passed
    assert result.delta < 0


def test_check_keep_baseline_is_symmetric_and_rejects_too_high():
    """A surprisingly *high* observed nDCG@10 should fail too."""
    result = check_keep_baseline([PUBLISHED_MIRACL_BM25_NDCG10 + DEFAULT_TOLERANCE + 0.01] * 10)
    assert not result.passed
    assert result.delta > 0


def test_check_keep_baseline_respects_custom_reference_and_tolerance():
    # Custom reference = 0.5, tolerance = 0.001 — observed 0.5005 passes, 0.51 fails.
    pass_result = check_keep_baseline([0.5005] * 10, reference=0.5, tolerance=0.001)
    fail_result = check_keep_baseline([0.51] * 10, reference=0.5, tolerance=0.001)
    assert pass_result.passed
    assert not fail_result.passed


def test_check_keep_baseline_empty_raises():
    with pytest.raises(ValueError):
        check_keep_baseline([])


def test_load_keep_baseline_ndcg10_filters_blanks_and_non_numerics(tmp_path: Path):
    csv_path = tmp_path / "keep.csv"
    _write_per_query_csv(
        csv_path,
        [
            {"query_id": "q1", "condition": "bm25__keep", "ndcg@10": 0.5},
            {"query_id": "q2", "condition": "bm25__keep", "ndcg@10": ""},
            {"query_id": "q3", "condition": "bm25__keep", "ndcg@10": "n/a"},
            {"query_id": "q4", "condition": "bm25__keep", "ndcg@10": 0.4},
        ],
    )
    values = load_keep_baseline_ndcg10(csv_path)
    assert values == [0.5, 0.4]


def test_load_keep_baseline_ndcg10_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_keep_baseline_ndcg10(tmp_path / "no_such.csv")


def test_load_keep_baseline_ndcg10_missing_column_raises(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "score"])
        writer.writeheader()
        writer.writerow({"query_id": "q1", "score": 0.4})
    with pytest.raises(ValueError, match="ndcg@10"):
        load_keep_baseline_ndcg10(csv_path)


def test_load_keep_baseline_ndcg10_all_blank_rows_raises(tmp_path: Path):
    csv_path = tmp_path / "blank.csv"
    _write_per_query_csv(
        csv_path,
        [{"query_id": "q1", "condition": "bm25__keep", "ndcg@10": ""}],
    )
    with pytest.raises(ValueError, match="zero usable"):
        load_keep_baseline_ndcg10(csv_path)


def test_check_keep_baseline_from_csv_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "keep.csv"
    _write_per_query_csv(
        csv_path,
        [
            {
                "query_id": f"q{i}",
                "condition": "bm25__keep",
                "ndcg@10": PUBLISHED_MIRACL_BM25_NDCG10,
            }
            for i in range(20)
        ],
    )
    result = check_keep_baseline_from_csv(csv_path)
    assert result.passed
    assert result.n_queries == 20


def test_sanity_result_explain_includes_pass_or_fail_verdict():
    passed = check_keep_baseline([PUBLISHED_MIRACL_BM25_NDCG10] * 10)
    failed = check_keep_baseline([0.0] * 10)
    assert "[PASS]" in passed.explain()
    assert "[FAIL]" in failed.explain()
