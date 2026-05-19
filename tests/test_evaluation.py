from pathlib import Path

import pandas as pd

from nya_ir.cli.analyze_results import main as analyze_results_main
from nya_ir.cli.evaluate_runs import main as evaluate_runs_main
from nya_ir.evaluation.metrics import compute_metrics, ndcg_at_k, recall_at_k, reciprocal_rank


def test_basic_metric_computation():
    qrels = {"d1": 1, "d3": 1}
    ranked = ["d2", "d1", "d3"]

    assert reciprocal_rank(qrels, ranked, 100) == 0.5
    assert recall_at_k(qrels, ranked, 2) == 0.5
    assert ndcg_at_k(qrels, ranked, 3) > 0

    metrics = compute_metrics(qrels, ranked)
    assert set(metrics) == {"ndcg@1", "ndcg@10", "recall@10", "mrr@100", "recall@100"}


def test_evaluate_runs_writes_per_query_metrics_and_summary(tmp_path: Path):
    qrels = tmp_path / "qrels.txt"
    run = tmp_path / "run.txt"
    metrics = tmp_path / "metrics" / "bm25__keep.csv"
    summary = tmp_path / "reports" / "bm25__keep_summary.csv"

    qrels.write_text(
        "\n".join(
            [
                "q1 0 d1 1",
                "q1 0 d3 1",
                "q2 0 d4 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run.write_text(
        "\n".join(
            [
                "q1 Q0 d2 1 2.0 bm25__keep",
                "q1 Q0 d1 2 1.5 bm25__keep",
                "q1 Q0 d3 3 1.0 bm25__keep",
                "q2 Q0 d4 1 3.0 bm25__keep",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = evaluate_runs_main(
        [
            "--qrels",
            str(qrels),
            "--run",
            str(run),
            "--condition",
            "bm25__keep",
            "--output",
            str(metrics),
            "--summary-output",
            str(summary),
        ]
    )

    assert exit_code == 0
    metric_rows = pd.read_csv(metrics)
    assert list(metric_rows["query_id"]) == ["q1", "q2"]
    assert set(metric_rows.columns) == {
        "query_id",
        "condition",
        "ndcg@1",
        "ndcg@10",
        "recall@10",
        "mrr@100",
        "recall@100",
    }
    summary_rows = pd.read_csv(summary)
    assert summary_rows.loc[0, "condition"] == "bm25__keep"
    assert summary_rows.loc[0, "count"] == 2


def test_analyze_results_writes_summary_and_pairwise_tables(tmp_path: Path):
    keep = tmp_path / "bm25__keep.csv"
    naive = tmp_path / "bm25__naive_strip.csv"
    summary = tmp_path / "summary.csv"
    pairwise = tmp_path / "pairwise.csv"

    header = "query_id,condition,ndcg@1,ndcg@10,recall@10,mrr@100,recall@100\n"
    keep.write_text(
        header
        + "q1,bm25__keep,1.0,0.8,1.0,1.0,1.0\n"
        + "q2,bm25__keep,0.0,0.5,1.0,0.5,1.0\n",
        encoding="utf-8",
    )
    naive.write_text(
        header
        + "q1,bm25__naive_strip,0.0,0.3,0.5,0.5,1.0\n"
        + "q2,bm25__naive_strip,0.0,0.2,0.5,0.0,0.5\n",
        encoding="utf-8",
    )

    exit_code = analyze_results_main(
        [
            "--metrics",
            str(keep),
            str(naive),
            "--metric",
            "ndcg@10",
            "--output",
            str(summary),
            "--pairwise-output",
            str(pairwise),
            "--bootstrap-resamples",
            "100",
        ]
    )

    assert exit_code == 0
    summary_rows = pd.read_csv(summary)
    assert list(summary_rows["condition"]) == ["bm25__keep", "bm25__naive_strip"]
    pairwise_rows = pd.read_csv(pairwise)
    assert pairwise_rows.loc[0, "condition_a"] == "bm25__keep"
    assert pairwise_rows.loc[0, "condition_b"] == "bm25__naive_strip"
    assert pairwise_rows.loc[0, "paired_queries"] == 2

