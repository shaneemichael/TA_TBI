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
    """Smoke test for the analyze_results CLI.

    Uses N=8 queries because the Wilcoxon per-query guard requires N >= 6 to fire,
    which is the same defensive constraint that protects production analyses from
    accidentally being run on aggregated condition means.
    """
    keep = tmp_path / "bm25__keep.csv"
    naive = tmp_path / "bm25__naive_strip.csv"
    summary = tmp_path / "summary.csv"
    pairwise = tmp_path / "pairwise.csv"

    header = "query_id,condition,ndcg@1,ndcg@10,recall@10,mrr@100,recall@100\n"
    
    # 8 queries; Keep dominates Naive on every row so the directional one-tailed
    # ("Naive < Keep") test for the H2 pair has a clear signal.
    keep_rows = "".join(
        f"q{i},bm25__keep,1.0,{0.75 + 0.01 * i:.3f},1.0,1.0,1.0\n" for i in range(1, 9)
    )
    naive_rows = "".join(
        f"q{i},bm25__naive_strip,0.0,{0.15 + 0.01 * i:.3f},0.5,0.5,0.75\n"
        for i in range(1, 9)
    )
    keep.write_text(header + keep_rows, encoding="utf-8")
    naive.write_text(header + naive_rows, encoding="utf-8")

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
    # Order-invariant: keep should outrank naive, but don't assume specific column order
    assert set(summary_rows["condition"]) == {"bm25__keep", "bm25__naive_strip"}
    pairwise_rows = pd.read_csv(pairwise)
    assert len(pairwise_rows) == 1
    row = pairwise_rows.iloc[0]
    assert {row["condition_a"], row["condition_b"]} == {"bm25__keep", "bm25__naive_strip"}
    assert row["paired_queries"] == 8

    # The H2 directional one-tailed alternative should be picked up automatically
    # via DEFAULT_DIRECTIONAL_PAIRS in analyze_results.
    assert row["alternative"] == "less"
    # And Bonferroni column should exist for the retriever family (size 1 pair here).
    assert row["retriever_family"] == "bm25"
    assert row["bonferroni_family_size"] == 1


def test_analyze_results_writes_friedman_and_stratified_outputs(tmp_path: Path):
    paths: list[Path] = []
    header = "query_id,condition,ndcg@1,ndcg@10,recall@10,mrr@100,recall@100\n"
    conditions = {
        "bm25__keep": 0.50,
        "bm25__sastrawi_clitic": 0.55,
        "bm25__rule_resolved": 0.60,
    }
    for condition, base in conditions.items():
        path = tmp_path / f"{condition}.csv"
        rows = "".join(
            f"q{i},{condition},0.0,{base + 0.001 * i:.3f},1.0,0.5,1.0\n"
            for i in range(1, 9)
        )
        path.write_text(header + rows, encoding="utf-8")
        paths.append(path)

    sensitivity = tmp_path / "sensitivity.csv"
    sensitivity.write_text(
        "query_id,nya_total,nya_anaphoric,entity_referenced,sensitivity_score,tertile\n"
        "q1,0,0,0,0,low\n"
        "q2,0,0,0,0,low\n"
        "q3,1,1,0,3,mid\n"
        "q4,1,1,0,3,mid\n"
        "q5,3,3,1,12,high\n"
        "q6,3,3,1,12,high\n"
        "q7,3,3,1,12,high\n"
        "q8,3,3,1,12,high\n",
        encoding="utf-8",
    )
    friedman = tmp_path / "friedman.csv"
    stratified = tmp_path / "stratified.csv"

    exit_code = analyze_results_main(
        [
            "--metrics",
            *(str(path) for path in paths),
            "--metric",
            "ndcg@10",
            "--friedman-output",
            str(friedman),
            "--sensitivity",
            str(sensitivity),
            "--stratified-output",
            str(stratified),
            "--bootstrap-resamples",
            "100",
        ]
    )

    assert exit_code == 0
    friedman_rows = pd.read_csv(friedman)
    assert list(friedman_rows["retriever_family"]) == ["bm25"]
    assert friedman_rows.loc[0, "paired_queries"] == 8

    stratified_rows = pd.read_csv(stratified)
    assert set(stratified_rows["tertile"]) == {"low", "mid", "high"}
    assert set(stratified_rows["retriever_family"]) == {"bm25"}

