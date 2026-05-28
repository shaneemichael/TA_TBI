# Person D — Analysis, Stats, and Paper Handoff

## Inputs validated

- `results/metrics/bm25/*.csv`: 5 BM25 conditions.
- `results/metrics/bge_m3/*.csv`: 5 BGE-m3 conditions.
- `artifacts/sensitivity_annotations.csv`: H5 tertile labels.

Integrity checks:

- 10 metric CSVs found.
- Each metric CSV has 960 rows and 960 unique query IDs.
- Query IDs are identical across all 10 conditions.
- All metric values are within `[0, 1]`.

## Outputs written

Primary nDCG@10:

- `results/reports/person_d_condition_summary_ndcg10.csv`
- `results/reports/person_d_pairwise_ndcg10.csv`
- `results/reports/person_d_friedman_ndcg10.csv`
- `results/reports/person_d_h5_stratified_pairwise_ndcg10.csv`

Secondary metrics:

- `results/reports/person_d_condition_summary_ndcg1.csv`
- `results/reports/person_d_condition_summary_mrr100.csv`
- `results/reports/person_d_condition_summary_recall10.csv`
- `results/reports/person_d_condition_summary_recall100.csv`
- matching `person_d_pairwise_*`, `person_d_friedman_*`, and
  `person_d_h5_stratified_pairwise_*` CSVs for each metric.

## Primary findings

Mean nDCG@10:

| Condition | Mean nDCG@10 |
|---|---:|
| `bge_m3__sentinel` | 0.5603 |
| `bge_m3__keep` | 0.5603 |
| `bge_m3__sastrawi_clitic` | 0.5600 |
| `bge_m3__naive_strip` | 0.5591 |
| `bge_m3__rule_resolved` | 0.5583 |
| `bm25__naive_strip` | 0.4495 |
| `bm25__sastrawi_clitic` | 0.4490 |
| `bm25__keep` | 0.4486 |
| `bm25__sentinel` | 0.4469 |
| `bm25__rule_resolved` | 0.4413 |

H1:

- BM25 Friedman: statistic `40.1296`, p `4.07e-08`.
- BGE-m3 Friedman: statistic `2.6379`, p `0.6201`.

H2:

- Not supported. Naive strip is not significantly worse than Keep after
  correction for either retriever.

H3:

- Descriptively, best strategies differ by architecture (`naive_strip` for BM25,
  `sentinel` for BGE-m3), but the top-strategy deltas are tiny. Mixed-effects
  interaction modelling is deferred to future work / multi-benchmark replication.

H4:

- Not supported. Rule-resolved does not outperform the best non-resolving
  strategy. It is lower than Sastrawi clitic for BM25 and lower than Sentinel for
  BGE-m3.

H5:

- Not supported. Stratified analysis does not show larger reliable gains on the
  high-sensitivity stratum. Significant corrected effects appear only in the
  low-sensitivity BM25 stratum and reflect Rule-resolved underperformance.

## Paper updates

Updated:

- `paper/sections/05_results.tex`
- `paper/sections/06_discussion.tex`

The write-up explicitly calls out:

- the in-house rule-based resolver limitation,
- Sastrawi dictionary coverage limits,
- single-benchmark / single-configuration constraints.

## Verification

```text
110 passed
```
