"""Metric computation for ranked retrieval output."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))


def ndcg_at_k(qrels: Mapping[str, int], ranked_doc_ids: Sequence[str], k: int) -> float:
    actual = [int(qrels.get(doc_id, 0)) for doc_id in ranked_doc_ids[:k]]
    ideal = sorted((int(rel) for rel in qrels.values()), reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(actual, k) / ideal_dcg


def recall_at_k(qrels: Mapping[str, int], ranked_doc_ids: Sequence[str], k: int) -> float:
    relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def reciprocal_rank(qrels: Mapping[str, int], ranked_doc_ids: Sequence[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    qrels: Mapping[str, int],
    ranked_doc_ids: Sequence[str],
    *,
    cutoffs: tuple[int, ...] = (1, 10, 100),
) -> dict[str, float]:
    """Compute the metrics pre-registered in the blueprint."""

    metrics: dict[str, float] = {}
    if 1 in cutoffs:
        metrics["ndcg@1"] = ndcg_at_k(qrels, ranked_doc_ids, 1)
    if 10 in cutoffs:
        metrics["ndcg@10"] = ndcg_at_k(qrels, ranked_doc_ids, 10)
        metrics["recall@10"] = recall_at_k(qrels, ranked_doc_ids, 10)
    if 100 in cutoffs:
        metrics["mrr@100"] = reciprocal_rank(qrels, ranked_doc_ids, 100)
        metrics["recall@100"] = recall_at_k(qrels, ranked_doc_ids, 100)
    return metrics

