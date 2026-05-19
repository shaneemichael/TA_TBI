# Person 4 — Evaluation + Analysis Flow

## Scope

Person 4 owns the post-retrieval stage:

1. evaluate TREC run files against qrels,
2. write per-query metric CSVs,
3. write condition-level summaries,
4. run lightweight pairwise statistical comparisons across conditions.

The stage starts after Person 3 produces run files such as:

```text
results/runs/bm25__keep.txt
results/runs/bm25__naive_strip.txt
```

## Implemented flow

### 1. Evaluate one run file

```bash
python scripts/evaluate_runs.py \
  --qrels data/raw/qrels.miracl-v1.0-id-dev.tsv \
  --run results/runs/bm25__keep.txt \
  --condition bm25__keep \
  --output results/metrics/bm25__keep.csv \
  --summary-output results/reports/bm25__keep_summary.csv
```

Outputs:

- per-query CSV with `query_id`, `condition`, `ndcg@1`, `ndcg@10`, `recall@10`, `mrr@100`, `recall@100`
- optional condition summary CSV with mean/std/count/missing per metric

### 2. Analyze multiple metric files

```bash
python scripts/analyze_results.py \
  --metrics results/metrics/*.csv \
  --metric ndcg@10 \
  --output results/reports/condition_summary_ndcg10.csv \
  --pairwise-output results/reports/pairwise_ndcg10.csv
```

Outputs:

- condition-level summary sorted by mean metric value
- optional pairwise table with paired query count, mean delta, Wilcoxon p-value,
  Cliff's delta, and bootstrap confidence interval

## Definition of Done

- `evaluate_runs.py --help` exposes per-query output and optional summary output.
- `analyze_results.py --help` exposes condition summary and pairwise comparison outputs.
- Per-query metrics include all pre-registered metrics:
  - `ndcg@1`
  - `ndcg@10`
  - `mrr@100`
  - `recall@10`
  - `recall@100`
- Evaluation handles normal TREC-6 run files and qrels.
- Analysis validates required metric columns before computing statistics.
- Smoke tests cover:
  - qrels + run file -> per-query metrics
  - per-query metrics -> condition summary
  - multiple condition CSVs -> pairwise statistical table
- Local verification status: `44 passed`.

## Current blockers outside Person 4

- Full MIRACL qrels/run artifacts are not present in the repository yet.
- Person 3 still needs to produce real run files for all conditions before final analysis can be executed.
- Dense retrieval remains scaffolded, so BGE-m3 condition analysis depends on later retrieval work.
