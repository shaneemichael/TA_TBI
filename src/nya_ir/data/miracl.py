"""MIRACL dataset loading helpers.

No dataset is downloaded at import time. Hugging Face ``datasets`` is imported only
inside the loading functions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from nya_ir.data.io import read_jsonl
from nya_ir.data.records import Passage, Query
from nya_ir.exceptions import OptionalDependencyError


def _required_string(
    row: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    source: str,
) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    expected = "/".join(keys)
    raise ValueError(f"{source}: missing required field {expected}")


def _optional_string(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value)
    return text or None


def normalize_query_row(row: Mapping[str, object], *, source: str = "query row") -> Query:
    return Query(
        query_id=_required_string(row, ("query_id", "id"), source=source),
        text=_required_string(row, ("query", "text"), source=source),
    )


def normalize_corpus_row(row: Mapping[str, object], *, source: str = "corpus row") -> Passage:
    return Passage(
        doc_id=_required_string(row, ("docid", "doc_id", "id"), source=source),
        title=_optional_string(row, "title"),
        text=_required_string(row, ("text",), source=source),
    )


def _load_dataset(*args: object, **kwargs: object):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise OptionalDependencyError("datasets", "data") from exc
    return load_dataset(*args, **kwargs)


def load_miracl_queries(split: str = "dev", language: str = "id") -> Iterable[Query]:
    dataset = _load_dataset("miracl/miracl", language, split=split, trust_remote_code=True)
    for row in dataset:
        yield normalize_query_row(row, source="MIRACL query row")


def load_miracl_corpus(split: str = "train", language: str = "id") -> Iterable[Passage]:
    dataset = _load_dataset("miracl/miracl-corpus", language, split=split, trust_remote_code=True)
    for row in dataset:
        yield normalize_corpus_row(row, source="MIRACL corpus row")


def load_query_jsonl(path: str | Path) -> Iterable[Query]:
    input_path = Path(path)
    for row in read_jsonl(input_path):
        yield normalize_query_row(row, source=str(input_path))


def load_corpus_jsonl(path: str | Path) -> Iterable[Passage]:
    input_path = Path(path)
    for row in read_jsonl(input_path):
        yield normalize_corpus_row(row, source=str(input_path))
