# Indonesian *-nya* Preprocessing for Information Retrieval

A diagnostic study of preprocessing strategies for the polysemous Indonesian clitic *-nya* in modern dense vs. sparse retrieval, evaluated on the Indonesian split of MIRACL.

## Status

Phase 1 (Scoping) complete. Phase 2 (Investigation) begins after Day 1 verification checks. See `02_methodology_blueprint.md` §12 for open decisions.

## Repository structure

The LaTeX is organised following the Single Responsibility Principle: each file has one reason to change. Style is separated from content; content reused across documents lives in one place.

```
nya-ir-study/
├── style/                              Presentation layer (Single Responsibility)
│   ├── packages.tex                    Reason to change: package set
│   ├── macros.tex                      Reason to change: custom commands
│   └── preamble.tex                    Reason to change: page geometry / spacing
│
├── shared/                             Cross-document content (DRY)
│   ├── hypotheses.tex                  Canonical H1–H5 — change here, both docs update
│   ├── research_questions.tex          Canonical RQ-A and RQ-B
│   └── preprocessing_strategies.tex    Canonical 5-strategy enumeration
│
├── paper/                              Research paper (post-experiment artefact)
│   ├── main.tex                        Thin orchestrator: title + section ordering only
│   └── sections/
│       ├── 00_abstract.tex             Substantive draft
│       ├── 01_introduction.tex         Substantive draft
│       ├── 02_related_work.tex         Substantive draft
│       ├── 03_methodology.tex          Inputs from shared/
│       ├── 04_experiments.tex          Skeleton, fill in Week 1–2
│       ├── 05_results.tex              Skeleton, fill in Week 2–3
│       ├── 06_discussion.tex           Skeleton with limitations pre-written
│       └── 07_conclusion.tex           Skeleton + AI disclosure pre-written
│
├── proposal/                           Project proposal (pre-experiment artefact)
│   ├── main.tex                        Thin orchestrator
│   └── sections/
│       ├── 01_problem_statement.tex
│       ├── 02_research_questions.tex   Inputs from shared/
│       ├── 03_literature_review.tex
│       ├── 04_methodology.tex          Inputs from shared/
│       ├── 05_timeline.tex
│       └── 06_expected_contributions.tex
│
├── 01_research_question_brief.md       Markdown working brief (FINER scoring)
├── 02_methodology_blueprint.md         Full methodology, validity, statistical plan
├── references.bib                      Shared bibliography (paper + proposal)
├── .gitignore                          LaTeX-aware
├── setup_git.sh                        Run once on your Mac to init git
└── README.md                           This file
```

## What "follows SRP" means here, concretely

Each piece of the LaTeX has exactly one reason to change:

- **Adding a `\usepackage`** → only `style/packages.tex`.
- **Adding/renaming a custom command (e.g., `\id{...}`)** → only `style/macros.tex`.
- **Tightening section spacing** → only `style/preamble.tex`.
- **Rephrasing hypothesis H2** → only `shared/hypotheses.tex` — *both* the paper's methodology section and the proposal's RQ section update automatically.
- **Adding a 6th preprocessing strategy** → only `shared/preprocessing_strategies.tex`.
- **Editing the paper's introduction** → only `paper/sections/01_introduction.tex`.
- **Adding a new bibliography entry** → only `references.bib`.

Compare with a monolithic main.tex where any of these would force editing dozens of lines scattered across the file.

## Cross-references between files

```
paper/main.tex
├── \input ../style/packages
├── \input ../style/macros
├── \input ../style/preamble
└── \input sections/03_methodology.tex
                 ├── \input ../shared/hypotheses
                 └── \input ../shared/preprocessing_strategies

proposal/main.tex
├── \input ../style/packages          ← same files as paper/, no duplication
├── \input ../style/macros
├── \input ../style/preamble
└── \input sections/02_research_questions.tex
                 ├── \input ../shared/research_questions
                 └── \input ../shared/hypotheses   ← same canonical text as paper/
```

Note: the `\input` paths use `../shared/` and `../style/` because both `paper/` and `proposal/` are one level deep. If you reorganise, update those relative paths.

## Initial git setup (run once)

The repo was scaffolded inside Cowork, where the mounted folder restricts git's lock-file operations. Open Terminal on your Mac and run:

```bash
cd ~/Documents/Claude/Projects/Research/nya-ir-study
chmod +x setup_git.sh
bash setup_git.sh
```

The script removes the broken init left by scaffolding, runs a clean `git init` on `main`, and makes the first commit. **That first commit serves as your lightweight pre-registration timestamp** (per blueprint §9) — it freezes the methodology before any experiments run.

After that, push to GitHub:

```bash
git remote add origin git@github.com:<your-username>/nya-ir-study.git
git push -u origin main
```

Public from Day 1 is the recommendation in the blueprint — it reinforces reproducibility intent.

## Building the LaTeX

Both `paper/` and `proposal/` reference shared `style/` and `shared/` via `\input{../style/...}` etc. Compile from inside each document folder:

```bash
cd paper/
latexmk -pdf main.tex
```

Or the manual 4-pass:

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

For long Indonesian-language passages with extended Unicode, switch to XeLaTeX:

```bash
xelatex main.tex
```

`latexmk -xelatex` works too.

## Working in branches

Suggested branch hygiene during the experiment:

- `main` — methodology + paper draft, always compiles cleanly
- `experiment/<strategy-name>` — work-in-progress code for each preprocessing condition
- `paper/<section>` — paper section revisions
- `annotation/<batch>` — oracle annotation passes

After Day 1 verification, the first non-trivial branches will likely be `experiment/keep-baseline` and `experiment/sastrawi-clitic`.

## Authoring tips

- Indonesian examples should use `\id{...}` (defined in `style/macros.tex`).
- Add new references to `references.bib`, not inline; both `paper/` and `proposal/` resolve from there.
- When adding a new shared content fragment, follow the `shared/<topic>.tex` pattern; one file per topic.
- Update `02_methodology_blueprint.md` revision history at the bottom whenever the design changes.

## Next steps

See `02_methodology_blueprint.md` §12 (open decisions) and the Day 1 checklist in the conversation history.
