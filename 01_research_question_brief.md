# Research Question Brief

**Project:** Indonesian *-nya* preprocessing for information retrieval
**Researcher:** Maskrio
**Date:** 2026-05-12
**Stage:** Phase 1 — Scoping (deep-research pipeline)

---

## Topic

The Indonesian clitic *-nya* is one of the highest-frequency morphological items in the language (~19.5% of suffix occurrences per Denistia & Baayen 2021). It is polysemous, with at least four distinct functions on the same surface form:

1. **Possessive** — third-person singular ("rumahnya" = "his/her house")
2. **Definite marker** — context-dependent definiteness ("rumahnya" = "the house" given prior reference)
3. **Anaphoric pronoun** — referring to a previously mentioned entity ("pidatonya" = "his/her speech")
4. **Nominalizer** — converts verbs to noun-like forms ("datangnya" = "the arrival")
5. **Pragmatic / emphatic** — exclamations ("enaknya!") and discourse particle uses

Despite this complexity, standard Indonesian information-retrieval pipelines treat *-nya* as a generic suffix to be stripped (Sastrawi, Nazief-Adriani algorithm). The interaction between this preprocessing default and modern dense retrieval has not been controlled-tested.

## Research Questions

### RQ-A (Primary)

Among four preprocessing strategies for Indonesian *-nya* — keep, strip, sentinel-replace, and oracle anaphora resolution on a held-out subset — which yields the best retrieval effectiveness on MIRACL-id, and does the optimal strategy depend on retriever architecture (BM25 vs. BGE-m3)?

**Debatable tension:** Conventional Indonesian IR strips *-nya* by default. Two competing predictions:
- **H1a (stripping helps):** Removing *-nya* improves lexical matching by collapsing morphological variants. Should benefit BM25.
- **H1b (preservation helps for dense):** Dense encoders may use morphological signal to disambiguate context. Stripping destroys signal for BGE-m3.

### RQ-B (Embedded as Strategy 4)

Does anaphora-aware document expansion — replacing anaphoric *-nya* with its resolved antecedent — provide an upper bound on attainable improvement from anaphora-aware preprocessing for Indonesian factoid retrieval?

## FINER Scoring

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **F**easible | High | 8 main conditions × 1 benchmark; 90 GPU-hr Kaggle budget sufficient. 3-week scope realistic. |
| **I**nteresting | High | Challenges a default IR preprocessing assumption; both directions of result are publishable. |
| **N**ovel | High | No published study isolates *-nya* as IR dependent variable. Closest prior art is Malay clitic classification (no IR evaluation). |
| **E**thical | Clean | Public benchmark (MIRACL CC-BY-SA); no human subjects; standard AI disclosure required. |
| **R**elevant | High | Indonesian: ~270M speakers; *-nya* in nearly every paragraph; results inform every Indonesian RAG/QA system. |

## Scope Boundaries

**In scope:**
- Preprocessing strategies applied symmetrically to queries and passages
- BM25 sparse retrieval (pyserini)
- BGE-m3 dense retrieval (dense-mode only)
- MIRACL-id Indonesian split (dev set, 960 queries; corpus ~1.4M passages)
- Oracle anaphora resolution on a 100–150 query subset (manually annotated)
- nDCG@10, MRR@100 as primary metrics; Recall@10/100 diagnostic

**Out of scope:**
- Training new stemmers or coreference models
- Other Indonesian NLP tasks (NER, sentiment, MT)
- Other Austronesian languages (Malay results may be discussed)
- Cross-lingual retrieval
- Full-system anaphora resolution (only oracle ceiling)

## Sub-Questions

1. Does the optimal preprocessing strategy differ between sparse and dense retrievers?
2. What proportion of MIRACL-id queries contain at least one *-nya* whose treatment changes ranking?
3. How large is the gap between the best implementable strategy and the oracle ceiling?

## Literature Context

The gap sits at the intersection of three literatures that don't currently talk to each other:

- **Indonesian morphology**: well-described (Denistia & Baayen 2021, IndoMorph 2025, MorphInd) but does not measure IR effects.
- **Indonesian stemming for IR**: well-evaluated overall (Asian & Williams; Nazief-Adriani; Sastrawi at 95.2%) but does not isolate *-nya*.
- **Indonesian retrieval and QA**: emerging (IndoBERT, IndoNLU AACL 2020; MIRACL TACL 2023; Mr. TyDi MRL 2021) but does not control for clitic preprocessing.
- **Anaphora-for-IR (general)**: classical work (Mitkov; QA-anaphora studies) shows resolution helps; not applied to Indonesian.
- **Closest prior art**: Malay clitic *-nya* classification (2021) — accuracy benchmark only, no IR evaluation.

## Importance

1. **Methodological:** The strip-the-clitic default propagates from stemming-era IR into the dense-retrieval era without revalidation. Either direction of result informs Indonesian retrieval system design going forward.
2. **Scale:** ~270M Indonesian speakers; *-nya* is a high-frequency item appearing in nearly every paragraph.
3. **Equity:** Indonesian is high-speaker, low-resource; this study fits the Joshi et al. (ACL 2020) critique of English-centric NLP infrastructure.
4. **Transferability:** Findings may transfer to Malay and other Austronesian languages with cognate clitics.

## Decision: Locked Direction

- **Primary RQ:** RQ-A
- **Secondary RQ:** RQ-B (operationalized as Strategy 4 oracle condition within RQ-A's factorial)
- **Benchmark:** MIRACL-id only (Mr. TyDi-id replication only if compute permits at end of week 3)
- **Compute:** 30 GPU-hr/week × 3 weeks Kaggle = 90 GPU-hr total

---

*Next stage: Methodology Blueprint — see `02_methodology_blueprint.md`.*
