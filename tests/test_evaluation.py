from nya_ir.evaluation.metrics import compute_metrics, ndcg_at_k, recall_at_k, reciprocal_rank


def test_basic_metric_computation():
    qrels = {"d1": 1, "d3": 1}
    ranked = ["d2", "d1", "d3"]

    assert reciprocal_rank(qrels, ranked, 100) == 0.5
    assert recall_at_k(qrels, ranked, 2) == 0.5
    assert ndcg_at_k(qrels, ranked, 3) > 0

    metrics = compute_metrics(qrels, ranked)
    assert set(metrics) == {"ndcg@1", "ndcg@10", "recall@10", "mrr@100", "recall@100"}

