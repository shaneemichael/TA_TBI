"""Dataset records and I/O helpers."""

from nya_ir.data.io import read_jsonl, read_qrels, read_trec_run, write_jsonl, write_trec_run
from nya_ir.data.records import Passage, Qrel, Query, RunEntry

__all__ = [
    "Passage",
    "Qrel",
    "Query",
    "RunEntry",
    "read_jsonl",
    "read_qrels",
    "read_trec_run",
    "write_jsonl",
    "write_trec_run",
]
