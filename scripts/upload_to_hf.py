"""Upload preprocessed MIRACL-id corpora to a HuggingFace dataset repo.

Run on Person A's machine AFTER ``scripts/run_preprocessing_sweep.py`` succeeds.
Stages a temporary directory with the right layout, generates a dataset card,
then calls ``HfApi.upload_folder`` which diffs and only re-uploads changed
files (so re-runs are cheap).

Repo layout on the Hub::

    README.md                       <- auto-generated dataset card
    root_dict.txt                   <- Sastrawi dictionary used (Path A)
    qrels/qrels_dev.txt             <- TREC qrels extracted from MIRACL dev
    keep/queries_dev.jsonl
    keep/corpus_train.jsonl
    naive_strip/...
    sastrawi_clitic/...
    sentinel/...
    rule_resolved/...

Authentication:
    Run ``huggingface-cli login`` once before invoking this script. The script
    explicitly does NOT prompt for a token, so it can be safely re-run in CI.

Usage:
    python scripts/upload_to_hf.py \\
        --repo-id <username>/nya-ir-miracl-id-preprocessed \\
        --private \\
        --commit-message "Initial upload of 5-strategy preprocessing sweep"
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STRATEGIES: tuple[str, ...] = (
    "keep",
    "naive_strip",
    "sastrawi_clitic",
    "sentinel",
    "rule_resolved",
)

# Short prose blurbs for the dataset card. Kept here (not in markdown) so a
# strategy rename in the codebase is caught by grep, not silently stale.
STRATEGY_BLURBS: dict[str, str] = {
    "keep": "Baseline pass-through. Text is preserved exactly as MIRACL ships it.",
    "naive_strip": (
        "Every word ending in `-nya` has the suffix stripped unconditionally. "
        "Includes false positives like *punya*, *tanya*, *hanya*, *Kenya* — "
        "this strategy exists as a deliberately bad baseline."
    ),
    "sastrawi_clitic": (
        "PySastrawi's `RemoveInflectionalPossessivePronoun` visitor, guarded by "
        "the bundled 29,931-root Indonesian dictionary so that *punya/tanya/hanya/Kenya* "
        "are recognised as roots and preserved."
    ),
    "sentinel": (
        "Like sastrawi_clitic, but the stripped suffix is replaced with the "
        "literal token `<NYA>`. Lets a retriever learn that *something* clitic-shaped "
        "was here without committing to a specific reading."
    ),
    "rule_resolved": (
        "Rule-based resolver: when a `-nya` form is preceded within ~2 sentences by "
        "a capitalised antecedent that is itself not a `-nya` form, the suffix is "
        "replaced by that antecedent. Falls through to `keep` when no antecedent is found. "
        "This is an in-house heuristic, NOT the IndoCoref system (Artari et al., 2021); "
        "see the paper's Limitations section."
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace dataset repo id, e.g. 'maskrio/nya-ir-miracl-id-preprocessed'.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the 5 per-strategy subdirs + qrels_dev.txt.",
    )
    parser.add_argument(
        "--root-dict",
        type=Path,
        default=Path("artifacts/root_dict.txt"),
        help="Sastrawi root dictionary used by the preprocessing sweep.",
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=Path("artifacts/sensitivity_annotations.csv"),
        help=(
            "Per-query H5 sensitivity annotations CSV (alpha/beta/gamma "
            "components + tertile). Skipped silently if the file does not "
            "exist — useful when uploading just the preprocessing outputs."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public). Safer during the 4-day sprint.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message for this upload (default: timestamped auto-generated message).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage files and print the manifest, but do not call the Hub.",
    )
    return parser


def _build_readme(repo_id: str, processed_dir: Path) -> str:
    """Generate a dataset card. Strategy blurbs come from STRATEGY_BLURBS."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = [
        "---",
        "license: apache-2.0",
        "language:",
        "  - id",
        "task_categories:",
        "  - text-retrieval",
        "tags:",
        "  - miracl",
        "  - indonesian",
        "  - information-retrieval",
        "  - preprocessing",
        "  - clitic",
        "---",
        "",
        "# MIRACL-id with five `-nya` preprocessing strategies",
        "",
        f"Auto-generated by `scripts/upload_to_hf.py` on {today}. Source repo: "
        "https://github.com/Imbad0202/nya-ir-study (replace with the actual repo URL).",
        "",
        "This dataset is a **derivative work** of MIRACL (Zhang et al., 2023) ",
        "restricted to the Indonesian (`id`) subset, preprocessed five different ways ",
        "to study how Indonesian `-nya` clitic handling affects retrieval quality. ",
        "Licensed under Apache-2.0, matching MIRACL.",
        "",
        "## Preprocessing strategies",
        "",
    ]
    for strategy in STRATEGIES:
        lines.append(f"### `{strategy}`")
        lines.append("")
        lines.append(STRATEGY_BLURBS[strategy])
        lines.append("")

    lines += [
        "## Layout",
        "",
        "```",
        "README.md                       (this file)",
        "root_dict.txt                   Sastrawi root dictionary (29,931 entries, Path A)",
        "sensitivity_annotations.csv     Per-query H5 sensitivity (alpha=1*total + beta=2*anaphoric + gamma=3*entity_ref) + tertile",
        "qrels/qrels_dev.txt             TREC 4-column qrels extracted from MIRACL dev positives",
    ]
    for strategy in STRATEGIES:
        queries = processed_dir / strategy / "queries_dev.jsonl"
        corpus = processed_dir / strategy / "corpus_train.jsonl"
        n_q = _count_lines(queries) if queries.exists() else "?"
        n_c = _count_lines(corpus) if corpus.exists() else "?"
        lines.append(f"{strategy + '/':<25} queries_dev.jsonl ({n_q} rows), corpus_train.jsonl ({n_c} rows)")
    lines += [
        "```",
        "",
        "All JSONL files have two fields: `id` (str) and `contents` (str).",
        "Loadable with `datasets.load_dataset(\"" + repo_id + "\", data_files=...)` ",
        "or with the Pyserini / FlagEmbedding ingestion paths.",
        "",
        "## Citation",
        "",
        "If you use this dataset, please cite both MIRACL and this work:",
        "",
        "```bibtex",
        "@article{zhang2023miracl,",
        "  title={MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages},",
        "  author={Zhang, Xinyu and others},",
        "  journal={Transactions of the ACL},",
        "  year={2023}",
        "}",
        "```",
        "",
        "## Reproducibility",
        "",
        "Everything here is reproducible from the code repo via",
        "`scripts/run_preprocessing_sweep.py` followed by",
        "`scripts/upload_to_hf.py`. The Sastrawi dictionary (`root_dict.txt`) is ",
        "shipped alongside the data so that the dictionary-guarded strategies ",
        "(`sastrawi_clitic`, `rule_resolved`) can be reproduced bit-for-bit even ",
        "if PySastrawi ships a different `kata-dasar.txt` in a future release.",
        "",
    ]
    return "\n".join(lines)


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _stage(
    processed_dir: Path,
    root_dict: Path,
    staging_dir: Path,
    *,
    sensitivity: Path | None = None,
) -> None:
    """Copy the files we want on the Hub into ``staging_dir`` with the desired layout."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    # root_dict at top level
    shutil.copy2(root_dict, staging_dir / "root_dict.txt")

    # Sensitivity annotations at top level next to root_dict, since both are
    # methodology artifacts that travel with the data rather than data per se.
    # Optional — Person A's pipeline produces it in step A-7 but earlier uploads
    # may pre-date it; skip silently if the file does not exist.
    if sensitivity is not None and sensitivity.exists():
        shutil.copy2(sensitivity, staging_dir / "sensitivity_annotations.csv")

    # qrels under qrels/
    qrels_src = processed_dir / "qrels_dev.txt"
    if qrels_src.exists():
        (staging_dir / "qrels").mkdir(exist_ok=True)
        shutil.copy2(qrels_src, staging_dir / "qrels" / "qrels_dev.txt")

    # Each strategy under its own subdir
    for strategy in STRATEGIES:
        src_dir = processed_dir / strategy
        if not src_dir.is_dir():
            raise FileNotFoundError(
                f"Missing preprocessing output for strategy '{strategy}' at {src_dir}. "
                f"Run scripts/run_preprocessing_sweep.py first."
            )
        dst_dir = staging_dir / strategy
        dst_dir.mkdir(exist_ok=True)
        for src in src_dir.glob("*.jsonl"):
            shutil.copy2(src, dst_dir / src.name)


def _print_manifest(staging_dir: Path) -> None:
    total_bytes = 0
    print(f"\n=== upload manifest ({staging_dir}) ===")
    for path in sorted(staging_dir.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total_bytes += size
            rel = path.relative_to(staging_dir)
            print(f"  {size:>13,} bytes  {rel}")
    print(f"  {'-' * 13}")
    print(f"  {total_bytes:>13,} bytes total")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.processed_dir.is_dir():
        print(f"error: --processed-dir {args.processed_dir} does not exist", file=sys.stderr)
        return 2
    if not args.root_dict.exists():
        print(f"error: --root-dict {args.root_dict} not found", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="nya_hf_upload_") as tmp:
        staging_dir = Path(tmp)
        _stage(
            args.processed_dir,
            args.root_dict,
            staging_dir,
            sensitivity=args.sensitivity,
        )
        (staging_dir / "README.md").write_text(
            _build_readme(args.repo_id, args.processed_dir), encoding="utf-8"
        )
        _print_manifest(staging_dir)

        if args.dry_run:
            print("\n--dry-run: skipping Hub call")
            return 0

        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            print(
                "error: huggingface_hub not installed. Run `pip install huggingface_hub` "
                "(should come in with the [data] extra).",
                file=sys.stderr,
            )
            return 2

        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

        commit_message = args.commit_message or (
            f"Preprocessing sweep upload ({datetime.now(timezone.utc).isoformat(timespec='seconds')})"
        )
        api = HfApi()
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

        visibility = "private" if args.private else "public"
        print(f"\n✓ Uploaded to https://huggingface.co/datasets/{args.repo_id} ({visibility})")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
