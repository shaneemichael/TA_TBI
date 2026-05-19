"""Smoke tests for the retrieval CLI flow.

These tests inject a stub searcher that does not require Pyserini or FAISS, so
the queries -> run-file path can be exercised in pure Python.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nya_ir.cli.build_index import build_parser as build_index_parser
from nya_ir.cli.build_index import main as build_index_main
from nya_ir.cli.run_retrieval import iter_query_jsonl, run_searches
from nya_ir.data.io import read_trec_run, write_jsonl
from nya_ir.retrieval.base import RetrievalHit
from nya_ir.retrieval.bm25 import build_pyserini_index


class StubSearcher:
    """Deterministic searcher: returns `hits_per_query` fake docs per query."""

    def __init__(self, hits_per_query: int = 3) -> None:
        self.hits_per_query = hits_per_query
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int = 1000) -> list[RetrievalHit]:
        self.calls.append((query, top_k))
        n = min(self.hits_per_query, top_k)
        return [
            RetrievalHit(doc_id=f"doc{i}", score=1.0 / (i + 1), rank=i + 1)
            for i in range(n)
        ]


def _write_queries(path: Path, rows: list[tuple[str, str]]) -> None:
    write_jsonl(path, [{"id": qid, "contents": text} for qid, text in rows])


def test_iter_query_jsonl_yields_id_text_pairs(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.jsonl"
    _write_queries(queries_path, [("q1", "siapa Sukarno"), ("q2", "pidato terkenal")])
    pairs = list(iter_query_jsonl(queries_path))
    assert pairs == [("q1", "siapa Sukarno"), ("q2", "pidato terkenal")]


def test_run_searches_collects_one_entry_per_hit() -> None:
    searcher = StubSearcher(hits_per_query=3)
    queries = [("q1", "siapa Sukarno"), ("q2", "pidato terkenal")]
    entries = run_searches(searcher, queries, run_id="smoke", hits=10)

    assert len(entries) == 6  # 2 queries x 3 hits
    assert {e.query_id for e in entries} == {"q1", "q2"}
    assert all(e.run_id == "smoke" for e in entries)
    assert all(e.iteration == "Q0" for e in entries)
    # top_k forwarded
    assert searcher.calls == [("siapa Sukarno", 10), ("pidato terkenal", 10)]


def test_run_searches_roundtrip_through_trec_run_file(tmp_path: Path) -> None:
    """Smoke flow: prepared queries -> stub searcher -> TREC run file -> read back."""

    queries_path = tmp_path / "queries.jsonl"
    run_path = tmp_path / "run.txt"
    _write_queries(queries_path, [("q1", "siapa Sukarno"), ("q2", "pidato")])

    entries = run_searches(
        StubSearcher(hits_per_query=2),
        iter_query_jsonl(queries_path),
        run_id="bm25__keep",
        hits=5,
    )

    from nya_ir.data.io import write_trec_run

    write_trec_run(run_path, entries)
    parsed = read_trec_run(run_path)

    assert set(parsed) == {"q1", "q2"}
    for query_id, hits in parsed.items():
        assert [hit.rank for hit in hits] == [1, 2]
        assert all(hit.run_id == "bm25__keep" for hit in hits)
        assert all(hit.iteration == "Q0" for hit in hits)
        assert all(hit.query_id == query_id for hit in hits)

    # File format: six whitespace-separated columns per line, scores formatted to 6dp.
    text_lines = [
        line for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(text_lines) == 4
    for line in text_lines:
        parts = line.split()
        assert len(parts) == 6
        # Score column has at least one decimal point with 6 digits after it.
        assert "." in parts[4]
        assert len(parts[4].split(".")[1]) == 6


def test_build_pyserini_index_command_contains_required_flags(tmp_path: Path) -> None:
    """The BM25 indexer command must hit the JsonCollection generator with --storeRaw."""

    command = build_pyserini_index(
        collection_dir=tmp_path / "corpus",
        index_dir=tmp_path / "index",
        threads=4,
    )
    assert command[:5] == ["python", "-m", "pyserini.index.lucene", "--collection", "JsonCollection"]
    assert "--storeRaw" in command
    assert "--storePositions" in command
    assert "--storeDocvectors" in command
    assert "--threads" in command and command[command.index("--threads") + 1] == "4"


def test_build_index_cli_dry_runs_bm25(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    collection = tmp_path / "corpus"
    collection.mkdir()
    exit_code = build_index_main(
        [
            "--retriever",
            "bm25",
            "--collection-dir",
            str(collection),
            "--index-dir",
            str(tmp_path / "index"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "pyserini.index.lucene" in captured.out
    assert "JsonCollection" in captured.out


def test_build_index_cli_errors_on_missing_collection(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_index_main(
            [
                "--retriever",
                "bm25",
                "--collection-dir",
                str(tmp_path / "missing"),
                "--index-dir",
                str(tmp_path / "index"),
            ]
        )
    assert "not found" in str(excinfo.value)


def test_build_index_cli_dense_is_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_index_main(
            [
                "--retriever",
                "bge_m3",
                "--collection-dir",
                str(tmp_path),
                "--index-dir",
                str(tmp_path / "index"),
            ]
        )
    assert "scaffolded" in str(excinfo.value)


def test_build_index_parser_accepts_both_retriever_choices() -> None:
    parser = build_index_parser()
    # both enum values must parse without error
    parser.parse_args(["--retriever", "bm25", "--collection-dir", ".", "--index-dir", "."])
    parser.parse_args(["--retriever", "bge_m3", "--collection-dir", ".", "--index-dir", "."])
