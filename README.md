# CPGTrajBench anonymous review artifact

This artifact contains the inspectable runtime, evaluation code, one demo case,
and lightweight text/JSON packs for 30 public TCGA Lung cases. It is research
software, not a clinical product or medical advice.

## Reproduction tiers

1. **CPU verification:** install `.[dev]` and run `python -m pytest`.
2. **Public Lung verification:** inspect or score the 30 cases in `data/LUNG` and
   the corrected references in `evaluation/references`.
3. **Full 109-case experiments:** not independently reproducible from this
   artifact. Planner weights and the institutional UCEC/NPC case packs are not
   distributed.

The corrected Lung references target `NSCLC_2010@2010` with decision date
`2010-12-31`. Historical Dynamic/Combined paper scores were not recomputed and
must not be presented as exact results from these corrected references.

## Quickstart

```bash
python -m venv .venv
# bash: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest
```

Configuration is read from process environment variables. Copying
`.env.example` does not load it automatically. For bash use
`set -a; source .env; set +a`; for PowerShell set the listed `$env:NAME`
variables explicitly.

See `evaluation/README.md`, `data/README.md`, and
`docs/PLANNER_RELEASE.md` for the release boundaries.

## Licensing note

The third-party guideline text requires a separate redistribution-rights
review. Its presence in this review artifact does not imply that the project
MIT license grants rights to the guideline content.
