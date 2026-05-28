# Stage 2.5 Integrity Check

## Purpose

This audit sits between experiment completion and merge/paper finalisation. It
checks whether the analysis inputs are complete, internally consistent, and
interpreted conservatively in the paper draft.

## Artifact Completeness

| Artifact | Status | Notes |
|---|---:|---|
| BM25 metric CSVs | PASS | 5 files under `results/metrics/bm25/` |
| BGE-m3 metric CSVs | PASS | 5 files under `results/metrics/bge_m3/` |
| Sensitivity annotations | PASS | `artifacts/sensitivity_annotations.csv` |
| Person D nDCG@10 summary | PASS | `results/reports/person_d_condition_summary_ndcg10.csv` |
| Person D pairwise tests | PASS | `results/reports/person_d_pairwise_ndcg10.csv` |
| Person D Friedman tests | PASS | `results/reports/person_d_friedman_ndcg10.csv` |
| Person D H5 stratified tests | PASS | `results/reports/person_d_h5_stratified_pairwise_ndcg10.csv` |
| Results section | PASS | `paper/sections/05_results.tex` |
| Discussion/limitations section | PASS | `paper/sections/06_discussion.tex` |

## Data Consistency Checks

All checks below passed on the 10 per-query metric CSVs.

| Check | Result |
|---|---:|
| Number of metric CSVs | 10 |
| Rows per condition | 960 |
| Unique query IDs per condition | 960 |
| Query ID set identical across all conditions | PASS |
| Required metric columns present | PASS |
| Missing metric values | 0 |
| Metric values outside `[0, 1]` | 0 |

Required columns:

- `query_id`
- `condition`
- `ndcg@1`
- `ndcg@10`
- `recall@10`
- `mrr@100`
- `recall@100`

## Statistical Consistency Checks

### H1: Friedman Omnibus

Source: `results/reports/person_d_friedman_ndcg10.csv`

| Retriever | Statistic | p-value | Interpretation |
|---|---:|---:|---|
| BM25 | 40.1296 | 4.07e-08 | Significant omnibus preprocessing effect |
| BGE-m3 | 2.6379 | 0.6201 | No significant omnibus effect |

Paper status: PASS. `paper/sections/05_results.tex` reports H1 as supported for
BM25 but not BGE-m3.

### H2: Naive Strip Harm

Source: `results/reports/person_d_pairwise_ndcg10.csv`

| Retriever | Comparison | Directional hypothesis | Adjusted p-value | Interpretation |
|---|---|---|---:|---|
| BM25 | Keep vs Naive | Naive < Keep | 0.2173 | Not supported |
| BGE-m3 | Keep vs Naive | Naive < Keep | 1.0000 | Not supported |

Paper status: PASS. The paper states H2 is not supported and avoids claiming
measurable naive-strip harm.

### H3: Architecture Interaction

Observed best strategy differs descriptively:

- BM25: `naive_strip`
- BGE-m3: `sentinel`

The observed gaps are small, especially for BGE-m3. The paper explicitly
defers mixed-effects interaction modelling to future work / multi-benchmark
replication.

Paper status: PASS.

### H4: Rule-Resolved Gain

Source: `results/reports/person_d_pairwise_ndcg10.csv`

| Retriever | H4 target | Adjusted p-value | Interpretation |
|---|---|---:|---|
| BM25 | Rule-resolved > Sastrawi clitic | 1.0000 | Not supported |
| BGE-m3 | Rule-resolved > Sentinel | 1.0000 | Not supported |

Paper status: PASS. The paper states the in-house rule-based resolver does not
improve retrieval.

### H5: Sensitivity Stratification

Source: `results/reports/person_d_h5_stratified_pairwise_ndcg10.csv`

Sensitivity tertile counts:

| Tertile | Queries |
|---|---:|
| Low | 331 |
| Mid | 269 |
| High | 360 |

The expected pattern, larger reliable gains on high-sensitivity queries, is not
observed. Corrected significant rows appear only in the low-sensitivity BM25
stratum and reflect Rule-resolved underperformance.

Paper status: PASS.

## Paper Claim Consistency

Checked paper sections:

- `paper/sections/05_results.tex`
- `paper/sections/06_discussion.tex`

Required limitation coverage:

| Required limitation | Status |
|---|---:|
| In-house rule-based resolver explicitly named | PASS |
| Resolver is not neural / not oracle | PASS |
| Sastrawi dictionary coverage limits | PASS |
| Single benchmark / single seed-config limitation | PASS |
| H3 mixed-effects decision | PASS |

Overclaim check:

- PASS: The paper does not claim H2/H4/H5 support.
- PASS: The paper distinguishes BM25 omnibus significance from practical effect
  size.
- PASS: BGE-m3 is described as robust/tied rather than improved by a strategy.

## Verification Commands

Executed:

```text
python -m pytest
```

Result:

```text
110 passed
```

Executed Person D full analysis for all registered metrics:

- `ndcg@1`
- `ndcg@10`
- `recall@10`
- `mrr@100`
- `recall@100`

Outputs use the `results/reports/person_d_*` prefix.

## Residual Risks Before Merge

1. This audit validates metric CSVs, not the original large run files or indices.
   The run/index artifacts are not committed locally, so provenance relies on the
   upstream Person B/C runbooks and handoff.
2. The H5 sensitivity labels are heuristic and not gold coreference annotation.
3. The Results section includes a compact nDCG@10 table only; secondary metric
   tables exist as CSVs but are not yet rendered into the paper.
4. The failure-analysis subsection is still high-level. A final submission should
   inspect representative query-level rank changes.
5. Adversarial sub-agent PR review is not available in this local tool context;
   this document is a self-audit and should be supplemented by reviewer feedback
   before merge.

## Verdict

Stage 2.5 integrity check: PASS with residual risks documented.

The experiment outputs are complete enough for Person D's analysis and paper
drafting. The main blocker remaining is external review/adversarial PR review,
not data or statistics completeness.
