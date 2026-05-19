# Person 3 — Retrieval Adapters + CLI Flow (Progress Report Material)

## 1. Architecture diagram

```
                          Prepared corpus / queries
                          (data/processed/<strategy>/*.jsonl)
                                   |
                                   v
   +---------------------------------------------------------+
   |  src/nya_ir/cli/build_index.py                          |
   |    --retriever bm25      (default = dry-run)            |
   |    --collection-dir ...  (validated; exists?)           |
   |    --index-dir ...                                      |
   |  Builds the Pyserini Lucene-indexer command via         |
   |  build_pyserini_index() (no Pyserini import needed      |
   |  for dry-run). With --execute, subprocess invokes       |
   |  `python -m pyserini.index.lucene`.                     |
   +---------------------------------------------------------+
                                   |
                                   v
                          artifacts/indexes/bm25/<strategy>/
                                   |
                                   v
   +---------------------------------------------------------+
   |  src/nya_ir/cli/run_retrieval.py                        |
   |    --retriever bm25                                     |
   |    --index-dir, --queries, --output, --run-id           |
   |    --hits, --k1, --b                                    |
   |                                                         |
   |  _build_searcher(args)  -- lazy: imports                |
   |  PyseriniBM25Searcher only on the BM25 branch,          |
   |  so missing-Pyserini failures raise                     |
   |  OptionalDependencyError with the                       |
   |  `pip install .[bm25]` hint.                            |
   |                                                         |
   |  run_searches(searcher, queries, run_id, hits)          |
   |    -- pure-Python, takes any Retriever                  |
   |       (used by the smoke test with a stub).             |
   |                                                         |
   |  write_trec_run(output, entries)                        |
   |    -- TREC-6 format: qid Q0 docid rank score run_id     |
   +---------------------------------------------------------+
                                   |
                                   v
                        artifacts/runs/<retriever>__<strategy>.txt
                                   |
                                   v
   (Person 4) evaluate_runs / analyze_results

Optional dependencies (raised lazily, with extras hint):
  pyserini             -> bm25
  sentence-transformers, faiss-cpu  -> dense (scaffold only)
```

## 2. BM25 command flow (reproducible snippet)

```bash
# 0. Prepare the corpus + queries under one preprocessing strategy.
#    Output: data/processed/keep/queries_dev.jsonl
#            data/processed/keep/corpus_train.jsonl
python scripts/prepare_miracl.py --strategy keep --limit 10 --dry-run

# 1. Dry-run the Pyserini indexer command (no Pyserini install required for this).
python scripts/build_index.py \
  --retriever bm25 \
  --collection-dir data/processed/keep/corpus_train.jsonl \
  --index-dir artifacts/indexes/bm25/keep

# 2. Execute the indexer (requires pyserini + JDK).
python scripts/build_index.py \
  --retriever bm25 \
  --collection-dir data/processed/keep/corpus_train.jsonl \
  --index-dir artifacts/indexes/bm25/keep \
  --execute

# 3. Run retrieval over the prepared dev queries.
python scripts/run_retrieval.py \
  --retriever bm25 \
  --index-dir artifacts/indexes/bm25/keep \
  --queries data/processed/keep/queries_dev.jsonl \
  --output artifacts/runs/bm25__keep.txt \
  --run-id bm25__keep \
  --hits 1000 \
  --k1 0.9 --b 0.4
```

## 3. Adapter boundaries (what is and is not implemented)

| Adapter                | File                          | Status                | Optional dep         |
|------------------------|-------------------------------|-----------------------|----------------------|
| `Retriever` protocol   | `retrieval/base.py`           | done                  | —                    |
| Pyserini BM25 searcher | `retrieval/bm25.py`           | done (with `k1/b` in constructor) | `pyserini` (`.[bm25]`) |
| Pyserini index command | `retrieval/bm25.py`           | done                  | —                    |
| BGE-m3 encoder         | `retrieval/dense.py`          | scaffold              | `sentence-transformers` (`.[dense]`) |
| FAISS dense searcher   | `retrieval/dense.py`          | scaffold              | `faiss-cpu` (`.[dense]`) |
| `build_index` CLI      | `cli/build_index.py`          | done (BM25 dry-run + `--execute`); dense path returns non-zero with explicit message | — |
| `run_retrieval` CLI    | `cli/run_retrieval.py`        | done (BM25 path; `run_searches` factored for stubbed tests); dense path raises explicit `SystemExit` | — |

## 4. CLI ergonomics verification (manual checks)

- `python scripts/build_index.py --help` and `python scripts/run_retrieval.py --help` print clean argparse usage without importing Pyserini.
- `python scripts/build_index.py --retriever bm25 --collection-dir does_not_exist ...` exits 1 with `Collection path not found: ...`.
- `python scripts/build_index.py --retriever bge_m3 ...` exits 1 with `Dense indexing (BGE-m3 + FAISS HNSW) is scaffolded only.`
- `python scripts/run_retrieval.py --queries does_not_exist ...` exits 1 with `Queries file not found: ...`.
- Attempting to instantiate `PyseriniBM25Searcher` without Pyserini raises `OptionalDependencyError: Optional dependency 'pyserini' is required for this operation. Install with 'pip install .[bm25]'.`

## 5. Smoke flow test

`tests/test_retrieval.py` exercises the **queries -> run-file** path with a stub
searcher (no Pyserini / JDK / FAISS required):

- `test_iter_query_jsonl_yields_id_text_pairs` — parser correctness.
- `test_run_searches_collects_one_entry_per_hit` — searcher dispatch + run-id wiring.
- `test_run_searches_roundtrip_through_trec_run_file` — full prepare->search->write->read cycle; asserts 6-column TREC-6 format and 6-decimal score precision.
- `test_build_pyserini_index_command_contains_required_flags` — `--storeRaw`, `--storePositions`, `--storeDocvectors`, threads forwarded.
- `test_build_index_cli_dry_runs_bm25` — BM25 dry-run prints the indexer command.
- `test_build_index_cli_errors_on_missing_collection` — missing-collection path exits non-zero.
- `test_build_index_cli_dense_is_not_implemented` — dense scaffold path exits non-zero with explicit message.
- `test_build_index_parser_accepts_both_retriever_choices` — both `RetrieverName` values parse.

Result: **8/8 passing**, total project suite **42/42 passing**.

## 6. Deferred to later phases

- Real Pyserini integration test (requires JDK in CI image).
- Dense pipeline: encode corpus -> build FAISS HNSW -> run dense retrieval.
- Hybrid retrieval (BM25 + BGE-m3 score fusion) — not in scope for this progress report.
