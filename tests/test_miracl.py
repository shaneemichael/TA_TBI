from pathlib import Path

import pytest

from nya_ir.cli.prepare_miracl import main as prepare_miracl_main
from nya_ir.data.io import read_jsonl
from nya_ir.data.miracl import (
    load_corpus_jsonl,
    load_query_jsonl,
    normalize_corpus_row,
    normalize_query_row,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "miracl_tiny"


def test_normalize_query_row_accepts_miracl_and_local_shapes():
    assert normalize_query_row({"query_id": "q1", "query": "rumahnya besar"}).query_id == "q1"
    assert normalize_query_row({"id": "q2", "text": "biasanya turun"}).text == "biasanya turun"


def test_normalize_query_row_rejects_missing_required_fields():
    with pytest.raises(ValueError, match="missing required field query_id/id"):
        normalize_query_row({"query": "rumahnya besar"})
    with pytest.raises(ValueError, match="missing required field query/text"):
        normalize_query_row({"query_id": "q1"})


def test_normalize_corpus_row_accepts_doc_id_variants_and_optional_title():
    passage = normalize_corpus_row({"doc_id": "d1", "title": "Judul", "text": "isi"})
    assert passage.doc_id == "d1"
    assert passage.title == "Judul"
    assert passage.text == "isi"

    assert normalize_corpus_row({"docid": "d2", "text": "isi"}).title is None
    assert normalize_corpus_row({"id": "d3", "title": "", "text": "isi"}).title is None


def test_normalize_corpus_row_rejects_missing_required_fields():
    with pytest.raises(ValueError, match="missing required field docid/doc_id/id"):
        normalize_corpus_row({"text": "isi"})
    with pytest.raises(ValueError, match="missing required field text"):
        normalize_corpus_row({"id": "d1"})


def test_local_jsonl_loaders_read_tiny_fixtures():
    queries = list(load_query_jsonl(FIXTURE_DIR / "queries.jsonl"))
    corpus = list(load_corpus_jsonl(FIXTURE_DIR / "corpus.jsonl"))

    assert [query.query_id for query in queries] == ["q1", "q2", "q3"]
    assert [passage.doc_id for passage in corpus] == ["d1", "d2", "d3"]
    assert corpus[0].title == "Rumah"
    assert corpus[2].title is None


def test_prepare_miracl_local_jsonl_smoke_writes_tiny_artifacts(tmp_path):
    exit_code = prepare_miracl_main(
        [
            "--strategy",
            "keep",
            "--queries-jsonl",
            str(FIXTURE_DIR / "queries.jsonl"),
            "--corpus-jsonl",
            str(FIXTURE_DIR / "corpus.jsonl"),
            "--limit",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    query_path = tmp_path / "keep" / "queries_dev.jsonl"
    corpus_path = tmp_path / "keep" / "corpus_train.jsonl"

    assert exit_code == 0
    assert list(read_jsonl(query_path)) == [
        {"id": "q1", "contents": "rumahnya besar"},
        {"id": "q2", "contents": "kenapa biasanya turun"},
    ]
    assert list(read_jsonl(corpus_path)) == [
        {"id": "d1", "contents": "Rumah\nrumahnya di kota"},
        {"id": "d2", "contents": "biasanya hujan turun"},
    ]


def test_prepare_miracl_requires_local_jsonl_paths_as_a_pair():
    with pytest.raises(SystemExit):
        prepare_miracl_main(
            [
                "--strategy",
                "keep",
                "--queries-jsonl",
                str(FIXTURE_DIR / "queries.jsonl"),
            ]
        )
