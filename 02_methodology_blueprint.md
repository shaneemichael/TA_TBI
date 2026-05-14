# Methodology Blueprint

**Project:** Indonesian *-nya* preprocessing for information retrieval
**Researcher:** Maskrio
**Date:** 2026-05-12 (revised 2026-05-14 — see revision history at end)
**Stage:** Phase 1 — Scoping (deep-research pipeline, `research_architect_agent` output)

> **Revision note (v3, 2026-05-14):** Strategy 5 reconfigured from manual oracle annotation on a 100–150-query subset to fully-automated rule-based anaphora resolution applied to the entire 1.4M-passage MIRACL-id corpus. Rationale: LLM-based resolution is cost-prohibitive at corpus scale (~$3,500 + 24 GPU-h for GPT-4o-class API); manual oracle annotation is out of scope for an S1 thesis. The rule-based pipeline is free, deterministic, reproducible, and resolves on the full corpus rather than a subset — eliminating the "Strategy 5 uses a different population than Strategies 1-4" methodological awkwardness of the original design. Hypothesis H4 reformulated accordingly: rule-based resolution vs. per-retriever-best non-resolving strategy, paired Wilcoxon on the full 960 dev queries.

---

## 1. Research Paradigm

Positivist quantitative experimental research. Controlled within-subjects factorial design with measurable, reproducible retrieval effectiveness metrics. The study tests pre-specified hypotheses against the null with frequentist statistical inference.

## 2. Hypotheses

Stated such that each is falsifiable.

**H1 (main effect of preprocessing):** At least one of the five preprocessing strategies (Keep, Naive strip, Sastrawi clitic, Sentinel, Rule-resolved) yields a significantly different nDCG@10 distribution than the others, separately for each retriever architecture.
- Tested via: Friedman test on per-query nDCG@10
- Reject H0 at α = 0.05

**H2 (naive strip is harmful — directional):** Naive regex *-nya* stripping yields significantly *lower* nDCG@10 than Keep on both retrievers, because of over-stripping high-frequency Indonesian non-clitic words (*punya, tanya, hanya, biasanya*, ...).
- Tested via: one-tailed paired Wilcoxon signed-rank, Naive vs. Keep per retriever
- Pre-committed prediction: ΔnDCG@10 < 0 with non-trivial effect size on BM25 (lexical matching directly hurt); smaller but still negative on BGE-m3 (encoder partially robust to mangled tokens)
- Reject H0 at α = 0.05

**H3 (architecture × preprocessing interaction):** The optimal preprocessing strategy among Keep / Sastrawi clitic / Sentinel / Rule-resolved differs between BM25 and BGE-m3.
- Tested via: comparison of best-performing strategy under each retriever, with bootstrap confidence intervals on the gain difference
- Pre-committed prediction: BM25 benefits from Sastrawi clitic stripping (recall ↑ from morphological collapsing); BGE-m3 either prefers Keep (encoder uses morphological signal) or is statistically indistinguishable across the three (encoder robust)

**H4 (rule-based resolution gain):** Rule-resolved (Strategy 5) yields significantly higher nDCG@10 than the per-retriever-best of Keep, Sastrawi clitic, and Sentinel.
- Tested via: paired Wilcoxon signed-rank, Strategy 5 vs. argmax(Strategies 1, 3, 4) per retriever, on the full 960 dev queries
- Reject H0 at Bonferroni-corrected α = 0.025

**H5 (sensitivity stratification):** Effects of preprocessing strategy on retrieval effectiveness are larger on high-*-nya*-sensitivity queries than on low-sensitivity queries.
- Tested via: interaction term in mixed-effects model on per-query nDCG@10 (strategy × sensitivity stratum), or as stratified Friedman tests with effect-size comparison
- If supported, strengthens claim that observed effects are *-nya*-mediated rather than incidental

**Null framing:** All five nulls are "no difference" — failure to reject is a substantive finding (modern dense retrievers are robust to *-nya* preprocessing) and is to be reported, not buried.

## 3. Experimental Design

Within-subjects factorial: each query is evaluated under all preprocessing × retriever combinations.

| Factor | Levels |
|---|---|
| Preprocessing strategy | 5: Keep / Naive strip / Sastrawi clitic / Sentinel / Rule-resolved |
| Retriever architecture | 2: BM25 / BGE-m3 (dense-mode) |

Optional 3rd retriever (multilingual-e5-base) added in Week 3 if compute permits.

### Conditions Matrix

| | Keep | Naive strip | Sastrawi clitic | Sentinel | Rule-resolved |
|---|---|---|---|---|---|
| **BM25** | 960 dev | 960 | 960 | 960 | 960 |
| **BGE-m3** | 960 dev | 960 | 960 | 960 | 960 |

10 full-set indices = 10 conditions. Estimated compute: ~60 GPU-hr (within 90 GPU-hr Kaggle budget). Strategy 5 itself runs CPU-only.

### Dependent Variables

- **Primary:** nDCG@10
- **Secondary:** MRR@100
- **Diagnostic:** Recall@10, Recall@100
- **Sensitivity-aware:** nDCG@1 (added per DA Checkpoint Issue 2 to defend against ceiling effect on BGE-m3)

## 4. Materials

### Dataset

- **MIRACL-id** v1.0 (Indonesian split)
  - Source: https://huggingface.co/datasets/miracl/miracl
  - Corpus: ~1.4M Wikipedia passages, native Indonesian
  - Queries: 4,070 train / 960 dev / 731 testA / 611 testB
  - Qrels: human-annotated by Indonesian native speakers
  - License: Apache 2.0 / CC-BY-SA (verify before redistribution)
- **Primary evaluation set:** dev (960 queries) — published metrics use this split
- **Held back:** testA reserved for confirmation if main results are surprising

### Models

- **BM25:** pyserini implementation, default k1=0.9, b=0.4 (standard MIRACL baseline)
- **BGE-m3:** `BAAI/bge-m3`, dense-mode embedding only (max-len capped at 512 for compute)
- **Optional:** `intfloat/multilingual-e5-base` (278M params)

### Tooling

- pyserini ≥ 0.21 (BM25 + trec_eval wrapper)
- sentence-transformers ≥ 2.7
- PySastrawi ≥ 1.0 (Strategy 3 — clitic rule isolated)
- ranx or pytrec_eval for metric computation
- FAISS-CPU for dense index (HNSW M=32, efConstruction=200)
- pandas, numpy, scipy.stats for analysis
- statsmodels for mixed-effects models and multiple-comparisons correction
- For Strategy 5: an Indonesian POS tagger (e.g. Stanza or IndoBERT-POS) and a list of Indonesian root words / proper nouns for antecedent selection

### Compute

- Kaggle T4×2 or P100, 30 GPU-hr/week × 3 weeks = 90 GPU-hr total
- Estimated burn (single dense retriever path, 5 strategies): ~60 GPU-hr
- Slack: ~30 GPU-hr for second retriever, debugging, reruns
- Strategy 5 resolver runs CPU-only; estimated 10–30 minutes for the full 1.4M-passage rewrite

## 5. Preprocessing Strategy Specifications

Each strategy is applied **identically to queries and passages** to maintain experimental control. Preprocessing happens before tokenization for BM25 and before encoding for dense retrieval.

### Strategy 1 — Keep

```python
def preprocess_keep(text: str) -> str:
    return text  # identity; UTF-8 normalized to NFC
```

Baseline. Confirms: encoder/tokenizer behavior on raw Indonesian text.

### Strategy 2 — Naive strip (the harmful baseline)

```python
import re
NYA_PATTERN = re.compile(r'(\w+?)nya\b', flags=re.UNICODE)

def preprocess_naive_strip(text: str) -> str:
    return NYA_PATTERN.sub(r'\1', text)
```

The "what NOT to do" condition. Naive regex over-strips high-frequency Indonesian non-clitic words: *punya* → "pu", *tanya* → "ta", *hanya* → "ha", *biasanya* → "biasa" (debatable), *bunyi*-derived forms; loanwords/proper nouns: *Kenya* → "ke", *Sonya* → "so", *Tanya* (name) → "ta".

This condition is explicitly included to **demonstrate the problem** and serves H2's directional prediction. The expected result is nDCG@10 ↓ relative to Keep, with the effect more pronounced on BM25.

### Strategy 3 — Sastrawi clitic rule (linguistically informed)

Use Sastrawi's *-nya* clitic-removal stage **in isolation**, not full Sastrawi stemming, wrapped with an explicit Indonesian-dictionary guard:

```python
from Sastrawi.Stemmer.Filter.RemovePossessivePronoun import RemovePossessivePronoun

remove_possessive = RemovePossessivePronoun()

def preprocess_sastrawi_clitic(text: str, root_dict: set[str]) -> str:
    tokens = text.split()
    out = []
    for tok in tokens:
        stripped = remove_possessive.filter(tok)
        # dictionary guard: commit the strip only if the result is a known root
        out.append(stripped if stripped in root_dict else tok)
    return ' '.join(out)
```

**CRITICAL Day 1 verification:** confirm that the dictionary guard prevents over-strips on the canonical false-positive set (*punya, tanya, hanya, biasanya, Kenya, Sonya, Tanya*) before running on the full corpus.

### Strategy 4 — Sentinel

```python
def preprocess_sentinel(text: str) -> str:
    return NYA_PATTERN.sub(r'\1 <NYA>', text)
```

Replaces *-nya* with a spaced sentinel. Diagnostic condition: removes lexical match while preserving information that *something* was there.

**Caveat:** has the same false-positive problem as Naive strip on *punya, tanya, hanya*. Reported anyway because the contrast Sentinel vs. Naive strip isolates "what's masked" vs. "what's lost" within the same false-positive base.

**Alternative considered:** apply Sastrawi-style disambiguation, then sentinel only on confirmed clitic *-nya*. Decision deferred to Day 1 — start with the simpler version, upgrade if false-positive rate is unacceptable.

### Strategy 5 — Rule-based anaphora resolution (full corpus)

Three-stage pipeline applied to all 1.4M MIRACL-id passages:

**Stage A — Detection.** Use the same dictionary-guarded matcher as Strategy 3 to identify clitic *-nya* occurrences (excluding false-positive non-clitic words).

**Stage B — Functional-class classification.** For each detected clitic, classify the function (possessive / definite / anaphoric / nominaliser) using surface-context heuristics:
- POS of the host word (verb host → likely nominaliser);
- Presence of a salient prior noun phrase in the preceding 1–2 sentences (no antecedent → likely possessive);
- Definite-marker discourse cues (`sebuah`, `tersebut`, etc.);
- Default: possessive.

**Stage C — Antecedent substitution.** For occurrences classified as anaphoric, substitute the clitic with the antecedent string. Antecedent = most recent salient noun phrase in the prior 1–2 sentences. Salience order: proper noun > definite NP > common noun.

Worked example:
> *Sukarno dilahirkan di Surabaya. Pidatonya terkenal.* → *Sukarno dilahirkan di Surabaya. Pidato Sukarno terkenal.*

Implementation: pure Python with regex, dictionary lookup, and POS tagging (Stanza or IndoBERT-POS). No neural inference per document, no API calls. Estimated wall-clock for full corpus: 10–30 minutes single-core CPU.

**Honest framing:** this is best-effort *automated* resolution, not an *oracle ceiling*. The resolver's classifier accuracy and antecedent-selection accuracy are bounded; we report a hand-annotated sanity check on a small held-out sample so readers can interpret retrieval gains relative to known resolver error.

## 6. Procedures

### Pipeline Architecture

```
Raw MIRACL-id corpus + queries
    ↓
Preprocessing (5 strategies, applied to both Q and D)
    ↓
Indexing
    ├── BM25: pyserini index_collection (CPU, ~30 min each × 5 = ~2.5 hr)
    └── BGE-m3: encode → FAISS HNSW (~6 hr/index on T4 × 5 = ~30 hr)
    ↓
Query encoding (BM25 implicit; BGE-m3 explicit, ~1 hr total)
    ↓
Top-1000 retrieval per condition
    ↓
Metric computation (nDCG@1, nDCG@10, MRR@100, R@10, R@100) via pytrec_eval
    ↓
Sensitivity stratification (compute *-nya* sensitivity score per query)
    ↓
Statistical analysis (Friedman + pairwise Wilcoxon-Bonferroni + mixed-effects)
```

### *-nya* Sensitivity Score (new in v2)

Per query, compute a sensitivity score over the gold-relevant passage(s):
```
sensitivity(q) = α · (count of -nya in gold passages)
              + β · (count of anaphoric -nya, heuristically detected by NP-followed-by-clause pattern)
              + γ · (1 if the queried entity is referred to via -nya at least once, else 0)
```

α=1, β=2, γ=3 as starting weights (sensitivity-tuned in pilot, frozen before main analysis). Stratify queries into low (bottom tertile), mid, high (top tertile) sensitivity bins. Report per-stratum effect sizes.

This is the substitute for crafting synthetic queries — it surfaces the queries where *-nya* should matter most without introducing construct-validity holes.

### Reproducibility Controls

- All seeds fixed at 42 (numpy, torch, faiss)
- `requirements.txt` with version pins
- Preprocessing strategies as named pure functions, unit-tested on 10 known-tricky words each
- Index construction logged with timestamps and config hashes
- All run scripts committed to GitHub before execution
- Results saved as TSV with run config as filename

## 7. Statistical Analysis Plan

Pre-committed analyses (run regardless of results):

### Primary Analyses

1. **Omnibus per retriever:** Friedman test on per-query nDCG@10 across the 5 strategies, separately for BM25 and BGE-m3.
2. **Pairwise post-hoc:** Wilcoxon signed-rank tests between every pair of strategies. With 5 strategies, C(5,2) = 10 pairs per retriever × 2 retrievers = 20 tests. Bonferroni correction within each retriever family (α/10 = 0.005).
3. **Effect size:** Cliff's δ for each pairwise comparison (small/medium/large thresholds: |δ| ≥ 0.147 / 0.33 / 0.474 per Romano et al.).
4. **Bootstrap CI:** 10,000 resamples on per-query ΔnDCG@10; report 95% CI for each pairwise comparison.
5. **H2 directional test:** one-tailed Wilcoxon, Naive vs. Keep, per retriever — pre-committed direction (ΔnDCG@10 < 0).
6. **H4 rule-based-gain test:** paired Wilcoxon, Rule-resolved vs. argmax(Keep, Sastrawi clitic, Sentinel) per retriever, Bonferroni-corrected α = 0.025.
7. **H5 stratification:** mixed-effects model `ndcg ~ strategy * sensitivity_bin + (1|query)` with random intercept per query. If model convergence issues, fall back to stratified Friedman tests with effect-size comparison across strata.

### Secondary Analyses

- Same battery on MRR@100, nDCG@1, Recall@10, Recall@100.
- Architecture interaction: difference-in-differences on (best - baseline) gain per retriever.
- Strategy 5 resolver accuracy on a hand-annotated ~100-passage sample: classifier 4-way accuracy and antecedent-selection binary accuracy.

### Exploratory Analyses (clearly labeled)

- Per-query rank changes across all 5 strategies: how many queries flip top-1 between Keep and each of the others?
- Correlation between query *-nya* count and rank change.
- Failure analysis: 20 hand-picked queries where Sastrawi clitic and Rule-resolved disagree most — error-pattern taxonomy.
- BPE fragmentation diagnostic: report passages-per-token-count per strategy.
- Optional: probing analysis on BGE-m3 contextual embeddings of *-nya* tokens (RQ-C territory; defer to follow-up).

## 8. Validity Threats and Mitigations

| Threat | Type | Mitigation |
|--------|------|------------|
| Sastrawi clitic rule in isolation may not disambiguate as expected | Construct | Day 1 verification; add explicit dictionary guard if needed |
| Naive strip is a strawman | Construct | Document its prevalence in the wild; framed as "harmful baseline that practitioners actually use" |
| Strategy 5 resolver error rate inflates or deflates measured gain | Construct | Hand-annotated 100-passage sanity check; report classifier and antecedent accuracy alongside retrieval gain so readers can calibrate |
| BPE fragmentation interacts with strategies in unmeasured ways | Construct | Report tokens-per-passage diagnostic per strategy |
| Ceiling effect on BGE-m3 dense retrieval (high baseline nDCG@10) | Statistical conclusion | Add nDCG@1 as a more sensitive metric; analyze low-baseline-quartile separately |
| Sentinel inherits Naive's false-positive problem | Construct | Acknowledge openly; consider Sastrawi-then-sentinel upgrade if false-positive rate is severe |
| Results don't generalize beyond MIRACL-id | External | Acknowledge as limitation; replicate on Mr. TyDi-id if time |
| Results don't generalize beyond BGE-m3 | External | Add multilingual-e5-base if compute permits |
| nDCG/MRR don't capture *-nya*-relevant retrieval quality | Construct | Multiple metrics; per-query qualitative analysis; sensitivity stratification |
| Multiple comparisons inflate Type I error | Statistical conclusion | Bonferroni within each retriever family; effect sizes alongside p-values |
| Cherry-picking strategies post-hoc | Researcher degrees of freedom | Pre-register all 5 strategies before running |
| Over-fitting to dev set | External | Reserve testA for confirmation only |

## 9. Pre-Registration

**Recommended:** Register on OSF (https://osf.io/registries) before running experiments. Total time: ~30 minutes. Strengthens claims significantly; reviewers cannot accuse post-hoc analysis.

**Lightweight alternative (acceptable for 3-week scope):** commit this methodology blueprint to a public GitHub repo with a verifiable timestamp before any experiments run. Same epistemic guarantee, less ceremony.

**Pre-registration content:**
- Hypotheses H1–H5 verbatim from §2
- Sample: MIRACL-id dev set (960 queries) under all 5 strategies × both retrievers
- Variables and operationalizations (this document, §3 and §5)
- Primary analysis: Friedman + Wilcoxon-Bonferroni + mixed-effects (this document, §7)
- Stopping rule: all 10 main conditions completed
- Data version: MIRACL-id v1.0
- Code commit: GitHub link to be added at registration time

## 10. Ethics Statement

- **Data:** public benchmark, no PII. Verify license terms on MIRACL-id before redistribution.
- **Annotation:** the only human-annotation step is the 100-passage resolver-accuracy sanity check (self-annotation, no IRB required).
- **AI disclosure:** required by ACL/EMNLP/SIGIR per current policy. State explicitly: literature scan and code drafts assisted by Claude (Sonnet/Opus). All experimental design, hypothesis formulation, and analysis decisions made by the human researcher.
- **Compute carbon estimate:** report as supplementary disclosure (~kgCO₂eq from Kaggle GPU hours).
- **Reproducibility statement:** all code and intermediate artifacts released on GitHub under MIT or Apache 2.0.

## 11. Devil's Advocate Pre-Mortem

Failure modes to plan framing for *now*, before running anything:

**Mode 1: Null result across all strategies on BGE-m3.**
- Reframing: "Modern dense retrievers are robust to *-nya* preprocessing on MIRACL-id; preprocessing complexity is unjustified."
- Pivot the discussion to: (a) why robustness exists (BPE may already handle morphology adequately); (b) implications for Indonesian RAG pipelines (you can stop preprocessing); (c) calls for harder benchmarks where the effect would surface.

**Mode 2: Naive strip is *not* significantly worse than Keep.**
- Means the over-stripping problem is theoretical but not empirically harmful at retrieval scale.
- Pivot: discuss why (the over-stripped tokens may not be in queries' content focus; encoders robust to mangled subwords; etc.).
- Still publishable as "naive *-nya* stripping is not as harmful as feared."

**Mode 3: Sastrawi clitic outperforms Keep on BGE-m3.**
- Counterintuitive — encoder benefits from morphological collapsing.
- Pivot to: identify which queries benefit; characterise the gain; argue for "linguistically-informed preprocessing is still useful even for modern dense retrievers."

**Mode 4: Rule-resolved ≪ Keep.**
- Most informative result: rule-based anaphora resolution introduces more noise than signal at retrieval scale.
- Reframes the importance argument: the bottleneck for retrieval improvement is not anaphora resolution. Future neural/LLM-based resolvers would face the same headwind.
- Possible explanation: dense encoders already track entity context implicitly across the passage.

**Mode 5: Strategy 5 produces almost no rewrites because the heuristic classifier under-detects anaphoric *-nya*.**
- Discovered Day 1–2 during pilot.
- Mitigation: tune classifier thresholds against the 100-passage sanity-check sample; relax detection rules until a non-trivial fraction (target 15–25% of *-nya* occurrences) are classified as anaphoric.
- Contingency: if even a relaxed rule-based classifier produces a tiny rewrite count, report the bound as a limitation and reframe Strategy 5 as "lower-bound estimate of automated-resolution gain."

**Mode 6: Sastrawi's isolated `RemovePossessivePronoun` does not perform dictionary disambiguation.**
- Discovered Day 1.
- Mitigation: implement explicit dictionary guard wrapping the call; document.
- Contingency: if dictionary-based protection is infeasible, fall back to full Sastrawi pipeline as Strategy 3 and document the confound.

## 12. Open Decisions Before Execution

Lock these before Day 1:

1. ~~Custom *-nya*-only stripper vs. Sastrawi-default as Strategy 2.~~ **RESOLVED:** 5-condition design with both Naive (Strategy 2) and Sastrawi clitic (Strategy 3).
2. **Pre-register on OSF or via timestamped GitHub commit.** Recommendation: GitHub commit for the 3-week scope (lightweight equivalent).
3. ~~Annotator for oracle subset.~~ **RESOLVED (v3):** Strategy 5 is now rule-based on the full corpus; no annotator needed. A 100-passage sanity-check sample is self-annotated.
4. **Optional second retriever (multilingual-e5-base).** Recommendation: yes if Week 1 + 2 stay on schedule.
5. **GitHub repo public from Day 1 vs. on submission.** Recommendation: public from Day 1 with a "work in progress" notice; reinforces reproducibility intent and serves as the lightweight pre-registration timestamp.
6. **Sentinel implementation: keep simple regex form, or upgrade to Sastrawi-disambiguated sentinel.** Recommendation: ship simple form for now, decide based on Day 1 false-positive rate.
7. **Strategy 5 POS tagger choice (Stanza vs IndoBERT-POS).** Recommendation: Stanza for stable, well-documented pipeline; IndoBERT-POS only if accuracy on the sanity sample is insufficient.

## 13. Deliverables

By end of Week 3:
- Reproducible code repository (GitHub)
- Results CSV with all per-query metrics across all 10 conditions
- Pre-registered analysis report (HTML or PDF)
- Methodology pre-registration on OSF or timestamped GitHub commit (Day 1 of Week 1)
- Draft paper (5–8 pages, suitable for SIGIR/EMNLP short or workshop)
- ***-nya* sensitivity scoring script** (reusable tool — bonus contribution)
- **Rule-based *-nya* resolver as a standalone Python package** (reusable tool — bonus contribution)
- This blueprint, updated with any in-flight protocol changes (audit trail)

---

## Revision History

- **v1 (2026-05-12 morning):** Initial draft. 4 preprocessing strategies (Keep, Strip-nya custom regex, Sentinel, Oracle).
- **v2 (2026-05-12 afternoon):** Expanded to 5 strategies after Devil's Advocate Checkpoint 1 surfaced over-stripping problem in original Strategy 2. Naive strip retained as harmful baseline (Strategy 2); Sastrawi clitic rule added as linguistically-informed condition (Strategy 3). Hypotheses updated to include H2 (directional naive-harm) and H5 (sensitivity stratification). Statistical plan updated for 5-strategy Bonferroni count (α/10 = 0.005). New §6 sub-section on *-nya* sensitivity scoring as substitute for query crafting.
- **v3 (2026-05-14):** Strategy 5 reconfigured from manual oracle annotation on a 100–150-query subset to fully-automated rule-based anaphora resolution on the full 1.4M-passage corpus. Hypothesis H4 reformulated as "rule-based vs per-retriever-best non-resolving strategy" on the full dev set. Conditions matrix updated (Strategy 5 now 960 dev under both retrievers, not subset). Annotator-recruitment open decision retired; sanity-check sample (100 passages, self-annotated) substituted. New deliverable: rule-based resolver as standalone Python package.

---

*Next stage: Phase 2 — Investigation. Begin with Day 1 verification tasks (Sastrawi behavior; BGE-m3 baseline ceiling check; rule-based resolver pilot on 100 passages).*
