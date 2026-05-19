import re
from pathlib import Path

import pytest

from nya_ir.data.io import read_jsonl, read_qrels, read_trec_run, write_jsonl, write_trec_run
from nya_ir.data.records import RunEntry


def _error_re(path: Path, line_number: int, message: str) -> str:
    return re.escape(f"{path}:{line_number}: {message}")


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"id": "q1", "text": "rumahnya"}, {"id": "q2", "score": 2}]

    write_jsonl(path, rows)

    assert list(read_jsonl(path)) == rows


def test_read_jsonl_rejects_non_object_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match=_error_re(path, 1, "JSONL row must be an object")):
        list(read_jsonl(path))


def test_read_jsonl_reports_invalid_json_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "q1"}\nnot json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(f"{path}:2: invalid JSON")):
        list(read_jsonl(path))


def test_read_qrels_parses_relevance_by_query(tmp_path):
    path = tmp_path / "qrels.txt"
    path.write_text("q1 0 d1 1\nq1 0 d2 0\nq2 0 d3 2\n", encoding="utf-8")

    assert read_qrels(path) == {"q1": {"d1": 1, "d2": 0}, "q2": {"d3": 2}}


def test_read_qrels_rejects_malformed_rows_with_path_and_line(tmp_path):
    path = tmp_path / "qrels.txt"
    path.write_text("q1 0 d1\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=_error_re(path, 1, "qrels row must have exactly 4 columns"),
    ):
        read_qrels(path)


def test_read_qrels_rejects_bad_relevance_with_path_and_line(tmp_path):
    path = tmp_path / "qrels.txt"
    path.write_text("q1 0 d1 relevant\n", encoding="utf-8")

    with pytest.raises(ValueError, match=_error_re(path, 1, "invalid relevance: relevant")):
        read_qrels(path)


def test_trec_run_roundtrip_uses_fixed_format_and_sorts_by_rank(tmp_path):
    path = tmp_path / "run.txt"
    entries = [
        RunEntry(query_id="q1", doc_id="d2", rank=2, score=0.125, run_id="bm25__keep"),
        RunEntry(query_id="q1", doc_id="d1", rank=1, score=1.0, run_id="bm25__keep"),
    ]

    write_trec_run(path, entries)

    assert path.read_text(encoding="utf-8").splitlines() == [
        "q1 Q0 d2 2 0.125000 bm25__keep",
        "q1 Q0 d1 1 1.000000 bm25__keep",
    ]
    assert [entry.doc_id for entry in read_trec_run(path)["q1"]] == ["d1", "d2"]


def test_read_trec_run_rejects_malformed_rows_with_path_and_line(tmp_path):
    path = tmp_path / "run.txt"
    path.write_text("q1 Q0 d1 1 0.5\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=_error_re(path, 1, "TREC run row must have exactly 6 columns"),
    ):
        read_trec_run(path)


def test_read_trec_run_rejects_bad_rank_and_score_with_path_and_line(tmp_path):
    bad_rank = tmp_path / "bad_rank.run"
    bad_rank.write_text("q1 Q0 d1 first 0.5 run\n", encoding="utf-8")
    with pytest.raises(ValueError, match=_error_re(bad_rank, 1, "invalid rank: first")):
        read_trec_run(bad_rank)

    bad_score = tmp_path / "bad_score.run"
    bad_score.write_text("q1 Q0 d1 1 high run\n", encoding="utf-8")
    with pytest.raises(ValueError, match=_error_re(bad_score, 1, "invalid score: high")):
        read_trec_run(bad_score)
