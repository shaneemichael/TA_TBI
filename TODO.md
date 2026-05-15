# TODO Until Project Completion

## 1. Environment Setup

- [ ] Create and activate a Python virtual environment.
- [ ] Install lightweight dev dependencies:

  ```bash
  pip install -e ".[dev]"
  ```

- [ ] Install optional experiment dependencies as needed:

  ```bash
  pip install -e ".[preprocessing,data,bm25]"
  pip install -e ".[dense]"
  ```

- [ ] Confirm tests run:

  ```bash
  python -m pytest
  ```

## 2. Day 1 Verification

- [ ] Run preprocessing verification:

  ```bash
  python scripts/verify_preprocessing.py
  ```

- [ ] Confirm Strategy 3 protects false positives:
  - [ ] `punya`
  - [ ] `tanya`
  - [ ] `hanya`
  - [ ] `biasanya`
  - [ ] `Kenya`
  - [ ] `Sonya`
  - [ ] `Tanya`
- [ ] Decide whether Sentinel stays simple regex-based or should use dictionary-guarded detection.
- [ ] Prepare or source an Indonesian root dictionary file.
- [ ] Verify Sastrawi behavior using real PySastrawi, not only the test double.

## 3. Data Preparation

- [ ] Download/load MIRACL-id corpus and dev queries.
- [ ] Prepare processed query and corpus files for all strategies:
  - [ ] Keep
  - [ ] Naive strip
  - [ ] Sastrawi clitic
  - [ ] Sentinel
  - [ ] Rule-resolved
- [ ] Confirm processed files are saved under `data/processed/`.
- [ ] Ensure large raw/processed artifacts are not accidentally committed.

## 4. Rule-Based Resolver

- [ ] Expand Strategy 5 resolver heuristics.
- [ ] Add Stanza POS tagging integration.
- [ ] Test resolver on a small hand-picked Indonesian sample.
- [ ] Create a 100-passage sanity-check sample.
- [ ] Annotate resolver outputs for:
  - [ ] Function classification accuracy
  - [ ] Antecedent-selection accuracy
- [ ] Report resolver limitations honestly.

## 5. Indexing

- [ ] Build BM25 indices for all 5 preprocessing strategies.
- [ ] Encode passages with BGE-m3 for all 5 preprocessing strategies.
- [ ] Build FAISS HNSW dense indices.
- [ ] Log index configuration, timestamps, and config hashes.
- [ ] Confirm all 10 main conditions are available:
  - [ ] 5 strategies x BM25
  - [ ] 5 strategies x BGE-m3

## 6. Retrieval

- [ ] Run top-1000 retrieval for BM25 conditions.
- [ ] Run top-1000 retrieval for BGE-m3 conditions.
- [ ] Save TREC run files under `results/runs/`.
- [ ] Validate run file format.
- [ ] Check that every dev query has retrieved results.

## 7. Evaluation

- [ ] Compute per-query metrics:
  - [ ] nDCG@1
  - [ ] nDCG@10
  - [ ] MRR@100
  - [ ] Recall@10
  - [ ] Recall@100
- [ ] Save metrics under `results/metrics/`.
- [ ] Generate condition-level summary tables.
- [ ] Sanity-check metric ranges and missing values.

## 8. Sensitivity Stratification

- [ ] Compute `-nya` sensitivity score for each dev query.
- [ ] Stratify queries into:
  - [ ] Low sensitivity
  - [ ] Mid sensitivity
  - [ ] High sensitivity
- [ ] Save sensitivity table.
- [ ] Verify tertile balance.

## 9. Statistical Analysis

- [ ] Run Friedman test per retriever on nDCG@10.
- [ ] Run Wilcoxon pairwise tests with Bonferroni correction.
- [ ] Run one-tailed Naive vs Keep test for H2.
- [ ] Run Rule-resolved vs best non-resolving strategy test for H4.
- [ ] Compute Cliff's delta effect sizes.
- [ ] Compute bootstrap confidence intervals.
- [ ] Run H5 sensitivity-stratified analysis.
- [ ] Repeat secondary analysis for MRR@100, nDCG@1, Recall@10, and Recall@100.

## 10. Result Inspection

- [ ] Identify queries with largest rank changes.
- [ ] Compare Keep vs Sastrawi clitic.
- [ ] Compare Keep vs Rule-resolved.
- [ ] Inspect cases where Rule-resolved hurts retrieval.
- [ ] Inspect cases where Sastrawi and Rule-resolved disagree.
- [ ] Write qualitative failure taxonomy.

## 11. Paper Updates

- [ ] Fill in `paper/sections/04_experiments.tex`.
- [ ] Fill in `paper/sections/05_results.tex`.
- [ ] Update `paper/sections/06_discussion.tex` based on actual findings.
- [ ] Update limitations and validity threats.
- [ ] Add implementation details:
  - [ ] Pyserini version
  - [ ] sentence-transformers version
  - [ ] PySastrawi version
  - [ ] FAISS config
  - [ ] hardware
  - [ ] seeds
- [ ] Add tables and figures.
- [ ] Compile paper successfully.

## 12. Reproducibility

- [ ] Update README with final experiment commands.
- [ ] Freeze dependency versions if needed.
- [ ] Ensure configs reproduce every condition.
- [ ] Confirm no large artifacts are committed accidentally.
- [ ] Add final result files or documented download instructions.
- [ ] Tag final experiment commit.
- [ ] Optionally register OSF or use GitHub commit as timestamped pre-registration.

## 13. Final Checks

- [ ] Run full test suite.
- [ ] Run script `--help` checks.
- [ ] Verify all paths in README work.
- [ ] Verify paper builds cleanly.
- [ ] Check citation and bibliography completeness.
- [ ] Review all hypotheses H1-H5 against final results.
- [ ] Prepare final submission/report package.
