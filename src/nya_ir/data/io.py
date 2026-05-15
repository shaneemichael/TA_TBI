"""File I/O for JSONL, qrels, and TREC run files."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from nya_ir.data.records import Qrel, RunEntry


def _parse_error(path: Path, line_number: int, message: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {message}")


def read_jsonl(path: str | Path) -> Iterator[dict[str, object]]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _parse_error(input_path, line_number, f"invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise _parse_error(input_path, line_number, "JSONL row must be an object")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    input_path = Path(path)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 4:
                raise _parse_error(input_path, line_number, "qrels row must have exactly 4 columns")
            query_id, _iteration, doc_id, relevance = fields
            try:
                qrels[query_id][doc_id] = int(relevance)
            except ValueError as exc:
                raise _parse_error(
                    input_path,
                    line_number,
                    f"invalid relevance: {relevance}",
                ) from exc
    return dict(qrels)


def read_trec_run(path: str | Path) -> dict[str, list[RunEntry]]:
    input_path = Path(path)
    runs: dict[str, list[RunEntry]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise _parse_error(
                    input_path,
                    line_number,
                    "TREC run row must have exactly 6 columns",
                )
            query_id, iteration, doc_id, rank, score, run_id = fields
            try:
                rank_value = int(rank)
            except ValueError as exc:
                raise _parse_error(input_path, line_number, f"invalid rank: {rank}") from exc
            try:
                score_value = float(score)
            except ValueError as exc:
                raise _parse_error(input_path, line_number, f"invalid score: {score}") from exc
            runs[query_id].append(
                RunEntry(
                    query_id=query_id,
                    iteration=iteration,
                    doc_id=doc_id,
                    rank=rank_value,
                    score=score_value,
                    run_id=run_id,
                )
            )
    return {
        query_id: sorted(entries, key=lambda entry: entry.rank)
        for query_id, entries in runs.items()
    }


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
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 4:
                raise _parse_error(input_path, line_number, "qrels row must have exactly 4 columns")
            query_id, iteration, doc_id, relevance = fields
            try:
                relevance_value = int(relevance)
            except ValueError as exc:
                raise _parse_error(
                    input_path,
                    line_number,
                    f"invalid relevance: {relevance}",
                ) from exc
            yield Qrel(
                query_id=query_id,
                iteration=iteration,
                doc_id=doc_id,
                relevance=relevance_value,
            )
