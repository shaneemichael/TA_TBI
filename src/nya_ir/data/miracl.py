"""MIRACL dataset loading helpers.

No dataset is downloaded at import time. Hugging Face ``datasets`` is imported only
inside the loading functions.
"""

from __future__ import annotations

from collections.abc import Iterable

from nya_ir.data.records import Passage, Query
from nya_ir.exceptions import OptionalDependencyError


def _load_dataset(*args: object, **kwargs: object):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise OptionalDependencyError("datasets", "data") from exc
    return load_dataset(*args, **kwargs)


def load_miracl_queries(split: str = "dev", language: str = "id") -> Iterable[Query]:
    dataset = _load_dataset("miracl/miracl", language, split=split)
    for row in dataset:
        yield Query(query_id=str(row.get("query_id") or row.get("id")), text=str(row["query"]))


def load_miracl_corpus(split: str = "train", language: str = "id") -> Iterable[Passage]:
    dataset = _load_dataset("miracl/miracl-corpus", language, split=split)
    for row in dataset:
        yield Passage(
            doc_id=str(row.get("docid") or row.get("id")),
            title=row.get("title"),
            text=str(row["text"]),
        )

