"""Tests for Person-B's forest plot generator.

Pure-computation tests (effect rows, pairing, CSV parsing) cover the
correctness surface. The render test only asserts a non-empty PNG was
written; visual-regression of matplotlib output is out of scope.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from nya_ir.analysis.figures import (
    EffectRow,
    compute_effect_rows,
    read_per_query_ndcg10,
    render_forest_plot,
)


def _write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_id", "condition", "ndcg@1", "ndcg@10", "recall@10", "mrr@100", "recall@100"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def test_read_per_query_ndcg10_returns_dict_of_floats(tmp_path: Path):
    csv_path = tmp_path / "m.csv"
    _write_metrics_csv(
        csv_path,
        [
            {"query_id": "q1", "condition": "bm25__keep", "ndcg@10": 0.5},
            {"query_id": "q2", "condition": "bm25__keep", "ndcg@10": 0.3},
            {"query_id": "q3", "condition": "bm25__keep", "ndcg@10": ""},
            {"query_id": "", "condition": "bm25__keep", "ndcg@10": 0.9},
        ],
    )
    out = read_per_query_ndcg10(csv_path)
    assert out == {"q1": 0.5, "q2": 0.3}


def test_read_per_query_ndcg10_missing_column_returns_empty(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id"])
        writer.writeheader()
        writer.writerow({"query_id": "q1"})
    assert read_per_query_ndcg10(csv_path) == {}


def test_compute_effect_rows_pairs_by_query_id():
    """Queries missing from either side must drop out, not be imputed."""
    baseline = {"q1": 0.5, "q2": 0.4, "q3": 0.6}
    # treatment is missing q3 entirely
    treatment = {"q1": 0.55, "q2": 0.45}
    rows = compute_effect_rows(baseline, {"treat": treatment}, n_resamples=100)
    assert len(rows) == 1
    row = rows[0]
    assert row.condition == "treat"
    assert row.n_pairs == 2
    # mean delta over the two shared queries
    assert row.mean_delta == pytest.approx(0.05, abs=1e-6)


def test_compute_effect_rows_emits_one_row_per_treatment():
    n = 20
    baseline = {f"q{i}": 0.4 + 0.01 * (i % 3) for i in range(n)}
    treatments = {
        "naive_strip": {qid: v - 0.05 for qid, v in baseline.items()},
        "rule_resolved": {qid: v + 0.03 for qid, v in baseline.items()},
    }
    rows = compute_effect_rows(baseline, treatments, n_resamples=200)
    assert [row.condition for row in rows] == ["naive_strip", "rule_resolved"]
    assert all(isinstance(row, EffectRow) for row in rows)
    assert all(row.n_pairs == n for row in rows)
    # naive_strip is uniformly lower → mean delta strictly negative
    assert rows[0].mean_delta < 0
    # rule_resolved is uniformly higher → mean delta strictly positive
    assert rows[1].mean_delta > 0


def test_compute_effect_rows_raises_on_disjoint_query_ids():
    baseline = {"q1": 0.5}
    treatment = {"q2": 0.4}  # no overlap at all
    with pytest.raises(ValueError, match="No paired queries"):
        compute_effect_rows(baseline, {"treat": treatment}, n_resamples=10)


def test_compute_effect_rows_ci_brackets_mean_delta():
    """A correct bootstrap CI must straddle (or touch) the observed mean delta."""
    n = 30
    baseline = {f"q{i}": 0.4 for i in range(n)}
    treatment = {f"q{i}": 0.42 + 0.0001 * i for i in range(n)}
    rows = compute_effect_rows(baseline, {"treat": treatment}, n_resamples=500)
    row = rows[0]
    assert row.ci_low <= row.mean_delta <= row.ci_high


def test_render_forest_plot_writes_a_non_empty_png(tmp_path: Path):
    rows = [
        EffectRow(condition="naive_strip", mean_delta=-0.03, ci_low=-0.05, ci_high=-0.01,
                  cliffs_delta=-0.18, n_pairs=100),
        EffectRow(condition="rule_resolved", mean_delta=0.02, ci_low=0.005, ci_high=0.035,
                  cliffs_delta=0.12, n_pairs=100),
    ]
    out_path = tmp_path / "forest.png"
    rendered = render_forest_plot(rows, out_path)
    assert rendered == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # PNG headers + content


def test_render_forest_plot_empty_rows_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        render_forest_plot([], tmp_path / "empty.png")
