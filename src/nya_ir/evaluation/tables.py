"""Helpers for per-query metric tables."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import mean, stdev

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


def condition_summary_rows(rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-query metric rows into condition-level summary rows."""

    metric_names = ["ndcg@1", "ndcg@10", "recall@10", "mrr@100", "recall@100"]
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)

    summaries: list[dict[str, object]] = []
    for condition, condition_rows in sorted(grouped.items()):
        summary: dict[str, object] = {"condition": condition, "count": len(condition_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in condition_rows if row.get(metric) is not None]
            summary[f"{metric}_mean"] = mean(values) if values else 0.0
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_missing"] = len(condition_rows) - len(values)
        summaries.append(summary)
    return summaries

