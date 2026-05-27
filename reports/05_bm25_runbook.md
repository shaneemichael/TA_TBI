# Person B — BM25 Track Runbook

Run this on the same machine as Person A (the prepared JSONLs at
`data/processed/<strategy>/{queries_dev,corpus_train}.jsonl` must already
exist locally).

## 0. One-time setup on the BM25 machine

```bash
# Install BM25 extras (pulls Pyserini, which needs a JDK).
pip install -e ".[bm25]"

# Verify JDK is reachable. Pyserini wraps Anserini (Lucene); without a JDK,
# building any index will fail with a JVM-not-found error.
java -version

# Confirm the prepared data is present.
ls data/processed/keep/corpus_train.jsonl    # ~1.4M lines for full MIRACL-id
ls data/processed/keep/queries_dev.jsonl     # ~960 lines for full MIRACL-id
ls artifacts/root_dict.txt                   # Sastrawi Path A export (~29 931 lines)
```

## 1. Extract the qrels file once

The `evaluate_runs` CLI consumes a 4-column TREC qrels file. Generate it once
from the same MIRACL dev split Person A pulled.

```bash
python scripts/extract_qrels.py \
  --output qrels/qrels_dev.txt \
  --language id \
  --split dev
```

Add `--queries-jsonl <path>` if you are working from a local MIRACL snapshot
instead of fetching from HuggingFace.

## 2. Run the full BM25 sweep

```bash
python scripts/run_bm25_pipeline.py \
  --processed-dir data/processed \
  --qrels qrels/qrels_dev.txt \
  --index-root artifacts/indexes/bm25 \
  --run-dir results/runs/bm25 \
  --metric-dir results/metrics/bm25 \
  --threads 8
```

What the orchestrator does, in order:

1. **`keep`** — stages a single-file collection dir under
   `artifacts/staging/bm25/keep/` (hardlink, falls back to copy), then
   `build_index → run_retrieval → evaluate_runs`, writing
   `results/metrics/bm25/bm25__keep.csv`.
2. **Sanity gate.** Reads that CSV, compares the mean nDCG@10 against the
   published MIRACL Indonesian BM25 number (`0.449 ± 0.02` by default;
   override with `--reference-ndcg10` and `--sanity-tolerance`). If the
   observed mean is outside the tolerance window (in either direction),
   the script exits non-zero and the remaining four conditions are NOT
   run. Fix the data / qrels / params and re-run.
3. **`naive_strip`, `sastrawi_clitic`, `sentinel`, `rule_resolved`** —
   same build → search → evaluate sequence per condition. The sweep is
   idempotent: any condition whose per-query CSV already exists is
   skipped on re-run.

**Outputs**

| Path                                          | What's there                                                                 |
|-----------------------------------------------|------------------------------------------------------------------------------|
| `artifacts/indexes/bm25/<strategy>/`          | Lucene index for each condition (large; gitignored)                          |
| `results/runs/bm25/bm25__<strategy>.txt`      | TREC-6 run file with top-`hits` per query (default `--hits 1000`)            |
| `results/metrics/bm25/bm25__<strategy>.csv`   | Per-query CSV with `ndcg@1, ndcg@10, recall@10, mrr@100, recall@100`         |

## 3. Re-run a single failed condition

```bash
# Example: rerun rule_resolved after fixing a resolver bug.
python scripts/run_bm25_pipeline.py \
  --processed-dir data/processed \
  --qrels qrels/qrels_dev.txt \
  --force-condition rule_resolved
```

The Keep CSV stays cached, so the sanity gate still runs (and re-passes)
without rebuilding the Keep index.

## 4. Build the forest plot

After all 5 per-query CSVs exist:

```bash
python scripts/plot_forest.py \
  --baseline results/metrics/bm25/bm25__keep.csv \
  --treatment naive_strip=results/metrics/bm25/bm25__naive_strip.csv \
  --treatment sastrawi_clitic=results/metrics/bm25/bm25__sastrawi_clitic.csv \
  --treatment sentinel=results/metrics/bm25/bm25__sentinel.csv \
  --treatment rule_resolved=results/metrics/bm25/bm25__rule_resolved.csv \
  --output results/reports/bm25_forest.png \
  --title "BM25: ΔnDCG@10 vs Keep baseline (MIRACL-id dev)"
```

The plot shows, for each non-Keep condition: paired mean ΔnDCG@10 with a
95% bootstrap CI, Cliff's δ on the marker label, and N (paired query count).
The stdout also prints the numbers, suitable for pasting into the paper:

```
naive_strip:     ΔnDCG@10 = -0.0312  [-0.0398, -0.0227]  δ=-0.158  N=960
sastrawi_clitic: ΔnDCG@10 = +0.0014  [-0.0042, +0.0070]  δ=+0.007  N=960
sentinel:        ΔnDCG@10 = -0.0089  [-0.0145, -0.0033]  δ=-0.045  N=960
rule_resolved:   ΔnDCG@10 = +0.0027  [-0.0030, +0.0084]  δ=+0.013  N=960
```

(These are illustrative numbers; the real values depend on your run.)

## 5. Hand off to Person D

Person D consumes:

- `results/metrics/bm25/bm25__<strategy>.csv` × 5 — for Friedman + Wilcoxon.
- `artifacts/sensitivity_annotations.csv` — Person A's tertile assignment, for H5 stratified pairwise.
- `results/reports/bm25_forest.png` — for the paper's main figure.

## Sanity-gate failure modes (what to check first)

| Failure symptom                                            | First thing to check                                                                                       |
|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `[FAIL]` with `delta < 0` (observed nDCG@10 too low)       | qrels file misaligned (wrong split, stale doc ids); corpus rows truncated; tokenisation breaking Indonesian. |
| `[FAIL]` with `delta > 0` (observed nDCG@10 too high)      | qrels file scrambled (same-id leakage); evaluating on the training split by accident; wrong run-id mapping. |
| Sanity passes but downstream conditions look identical     | Re-check `_first_diff_row` from `run_preprocessing_sweep` post-flight; strategies may have collapsed.        |

If you have a defensible reason to expect a different baseline (different
tokeniser, different Pyserini version), override the reference number on
the command line:

```bash
python scripts/run_bm25_pipeline.py ... \
  --reference-ndcg10 0.460 \
  --sanity-tolerance 0.025
```

Do NOT use `--skip-sanity-gate` for the actual run; it exists only for
debugging local-index experiments.
