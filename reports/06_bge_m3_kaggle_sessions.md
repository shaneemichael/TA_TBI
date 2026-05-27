# BGE-M3 Kaggle Session Commands

This runbook matches the current dense stack in this repo:

- encoder: `FlagEmbedding` / `BGEM3FlagModel`
- indexer: FAISS HNSW
- multi-device indexing: `--devices`
- query-time retrieval: no `--devices` flag; the code picks one default device automatically

Assumptions:

- repo path: `/kaggle/working/TA_TBI`
- preprocessed dataset path: `/kaggle/working/nya-ir-miracl-id-preprocessed`
- qrels path: `/kaggle/working/nya-ir-miracl-id-preprocessed/qrels/qrels_dev.txt`
- output paths stay inside the repo under `artifacts/` and `results/`

Important Kaggle note:

- Kaggle usually exposes a single GPU, so use `DEVICES=cuda:0`
- if your runtime really has multiple GPUs, replace with `DEVICES=cuda:0,cuda:1`
- set `--max-length 8192` explicitly in every session; do not rely on defaults
- if you see `[Errno 2] No such file or directory: '$REPO'`, you passed a
  literal string; use `/kaggle/working/TA_TBI` directly or run the command in
  a shell cell where env vars are expanded

## Common Setup For Every Session

Run this at the start of each fresh Kaggle session:

```bash
git clone <your-repo-url> /kaggle/working/TA_TBI
cd /kaggle/working/TA_TBI

pip install -e ".[dense,eval,dev,data]"

export REPO=/kaggle/working/TA_TBI
export DATA_ROOT=/kaggle/working/nya-ir-miracl-id-preprocessed
export QRELS=/kaggle/working/nya-ir-miracl-id-preprocessed/qrels/qrels_dev.txt
export DEVICES=cuda:0

ls "$DATA_ROOT"
ls "$DATA_ROOT/keep"
ls "$DATA_ROOT/qrels"
```

## Session 1: `keep`

Build the dense index:

```bash
cd /kaggle/working/TA_TBI

python scripts/build_index.py \
  --retriever bge_m3 \
  --collection-dir "$DATA_ROOT/keep/corpus_train.jsonl" \
  --index-dir artifacts/indexes/bge_m3/keep \
  --threads 4 \
  --batch-size 4 \
  --max-length 8192 \
  --devices "$DEVICES" \
  --hnsw-m 32 \
  --ef-construction 200 \
  --ef-search 1000 \
  --show-progress \
  --execute
```

Run retrieval:

```bash
cd /kaggle/working/TA_TBI

python scripts/run_retrieval.py \
  --retriever bge_m3 \
  --index-dir artifacts/indexes/bge_m3/keep \
  --queries "$DATA_ROOT/keep/queries_dev.jsonl" \
  --output results/runs/bge_m3/bge_m3__keep.txt \
  --run-id bge_m3__keep \
  --hits 1000 \
  --batch-size 4 \
  --max-length 8192 \
  --ef-search 1000
```

Evaluate:

```bash
cd /kaggle/working/TA_TBI

python scripts/evaluate_runs.py \
  --qrels "$QRELS" \
  --run results/runs/bge_m3/bge_m3__keep.txt \
  --condition bge_m3__keep \
  --output results/metrics/bge_m3/bge_m3__keep.csv \
  --summary-output results/reports/bge_m3/bge_m3__keep_summary.csv
```

Run the Keep sanity gate:

```bash
cd /kaggle/working/TA_TBI

python - <<'PY'
from nya_ir.analysis.sanity import check_keep_baseline_from_csv

result = check_keep_baseline_from_csv(
    "results/metrics/bge_m3/bge_m3__keep.csv",
    reference=0.561,
    tolerance=0.02,
)
print(result.explain())
if not result.passed:
    raise SystemExit("Keep failed sanity gate")
PY
```

Zip the outputs for handoff to the next session:

```bash
cd /kaggle/working/TA_TBI

mkdir -p handoff
zip -r handoff/keep_outputs.zip \
  artifacts/indexes/bge_m3/keep \
  results/runs/bge_m3/bge_m3__keep.txt \
  results/metrics/bge_m3/bge_m3__keep.csv \
  results/reports/bge_m3/bge_m3__keep_summary.csv
```

## Sessions 2 To 5: One Strategy Per Session

Use the same command pattern for each strategy:

- Session 2: `naive_strip`
- Session 3: `sastrawi_clitic`
- Session 4: `sentinel`
- Session 5: `rule_resolved`

Set the strategy first:

```bash
export STRATEGY=naive_strip
```

Build the dense index:

```bash
cd /kaggle/working/TA_TBI

python scripts/build_index.py \
  --retriever bge_m3 \
  --collection-dir "$DATA_ROOT/$STRATEGY/corpus_train.jsonl" \
  --index-dir "artifacts/indexes/bge_m3/$STRATEGY" \
  --threads 4 \
  --batch-size 4 \
  --max-length 8192 \
  --devices "$DEVICES" \
  --hnsw-m 32 \
  --ef-construction 200 \
  --ef-search 1000 \
  --show-progress \
  --execute
```

Run retrieval:

```bash
cd /kaggle/working/TA_TBI

python scripts/run_retrieval.py \
  --retriever bge_m3 \
  --index-dir "artifacts/indexes/bge_m3/$STRATEGY" \
  --queries "$DATA_ROOT/$STRATEGY/queries_dev.jsonl" \
  --output "results/runs/bge_m3/bge_m3__$STRATEGY.txt" \
  --run-id "bge_m3__$STRATEGY" \
  --hits 1000 \
  --batch-size 4 \
  --max-length 8192 \
  --ef-search 1000
```

Evaluate:

```bash
cd /kaggle/working/TA_TBI

python scripts/evaluate_runs.py \
  --qrels "$QRELS" \
  --run "results/runs/bge_m3/bge_m3__$STRATEGY.txt" \
  --condition "bge_m3__$STRATEGY" \
  --output "results/metrics/bge_m3/bge_m3__$STRATEGY.csv" \
  --summary-output "results/reports/bge_m3/bge_m3__${STRATEGY}_summary.csv"
```

Zip the outputs:

```bash
cd /kaggle/working/TA_TBI

mkdir -p handoff
zip -r "handoff/${STRATEGY}_outputs.zip" \
  "artifacts/indexes/bge_m3/$STRATEGY" \
  "results/runs/bge_m3/bge_m3__$STRATEGY.txt" \
  "results/metrics/bge_m3/bge_m3__$STRATEGY.csv" \
  "results/reports/bge_m3/bge_m3__${STRATEGY}_summary.csv"
```

Repeat with:

```bash
export STRATEGY=sastrawi_clitic
export STRATEGY=sentinel
export STRATEGY=rule_resolved
```

## Session 6: Finalization

Start a fresh Kaggle session, run the common setup, then restore the zipped
outputs from Sessions 1 to 5 into `/kaggle/working/TA_TBI`.

Unzip all handoff archives:

```bash
cd /kaggle/working/TA_TBI

unzip handoff/keep_outputs.zip -d /kaggle/working/TA_TBI
unzip handoff/naive_strip_outputs.zip -d /kaggle/working/TA_TBI
unzip handoff/sastrawi_clitic_outputs.zip -d /kaggle/working/TA_TBI
unzip handoff/sentinel_outputs.zip -d /kaggle/working/TA_TBI
unzip handoff/rule_resolved_outputs.zip -d /kaggle/working/TA_TBI
```

Run the orchestrator to:

- confirm Keep still passes the sanity gate
- skip the already-built per-strategy metrics
- build the final condition tables
- write the reproducibility archive

```bash
cd /kaggle/working/TA_TBI

python scripts/run_bge_m3_pipeline.py \
  --processed-dir "$DATA_ROOT" \
  --qrels "$QRELS" \
  --index-root artifacts/indexes/bge_m3 \
  --run-dir results/runs/bge_m3 \
  --metric-dir results/metrics/bge_m3 \
  --report-dir results/reports/bge_m3 \
  --archive-dir results/archive \
  --threads 4 \
  --batch-size 4 \
  --max-length 8192 \
  --devices "$DEVICES"
```

## Expected Final Outputs

After Session 6, you should have:

- `results/metrics/bge_m3/bge_m3__keep.csv`
- `results/metrics/bge_m3/bge_m3__naive_strip.csv`
- `results/metrics/bge_m3/bge_m3__sastrawi_clitic.csv`
- `results/metrics/bge_m3/bge_m3__sentinel.csv`
- `results/metrics/bge_m3/bge_m3__rule_resolved.csv`
- `results/reports/bge_m3/condition_summary_all_metrics.csv`
- `results/reports/bge_m3/condition_summary_ndcg10.csv`
- `results/reports/bge_m3/pairwise_ndcg10.csv`
- `results/archive/bge_m3_run_*.tar.gz`

## If Kaggle Runs Out Of Memory

Lower the indexing and retrieval batch size:

```bash
--batch-size 4
--batch-size 2
--batch-size 1
```

Keep `--max-length 8192` unchanged if you want the Keep condition to stay
comparable to the published BGE-M3 MIRACL-id Dense number.
