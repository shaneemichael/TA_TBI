"""Helpers for per-query metric tables."""

from __future__ import annotations

from collections.abc import Mapping

from nya_ir.data.records import RunEntry
from nya_ir.evaluation.metrics import compute_metrics


def per_query_metric_rows(
    qrels: Mapping[str, Mapping[str, int]],
    runs: Mapping[str, list[RunEntry]],
    *,
    condition: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query_id, query_qrels in qrels.items():
        ranked_doc_ids = [entry.doc_id for entry in runs.get(query_id, [])]
        metrics = compute_metrics(query_qrels, ranked_doc_ids)
        rows.append({"query_id": query_id, "condition": condition, **metrics})
    return rows

