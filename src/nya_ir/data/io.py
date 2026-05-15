"""File I/O for JSONL, qrels, and TREC run files."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from nya_ir.data.records import Qrel, RunEntry


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query_id, _iteration, doc_id, relevance = line.split()[:4]
            qrels[query_id][doc_id] = int(relevance)
    return dict(qrels)


def read_trec_run(path: str | Path) -> dict[str, list[RunEntry]]:
    runs: dict[str, list[RunEntry]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query_id, iteration, doc_id, rank, score, run_id = line.split()[:6]
            runs[query_id].append(
                RunEntry(
                    query_id=query_id,
                    iteration=iteration,
                    doc_id=doc_id,
                    rank=int(rank),
                    score=float(score),
                    run_id=run_id,
                )
            )
    return {query_id: sorted(entries, key=lambda entry: entry.rank) for query_id, entries in runs.items()}


def write_trec_run(path: str | Path, entries: Iterable[RunEntry]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                f"{entry.query_id} {entry.iteration} {entry.doc_id} "
                f"{entry.rank} {entry.score:.6f} {entry.run_id}\n"
            )


def iter_qrels(path: str | Path) -> Iterable[Qrel]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query_id, iteration, doc_id, relevance = line.split()[:4]
            yield Qrel(query_id=query_id, iteration=iteration, doc_id=doc_id, relevance=int(relevance))

