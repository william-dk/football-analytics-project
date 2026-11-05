# Archived EDA implementations

These files are historical snapshots, preserved byte for byte. They are not the
default analysis entry points. `archive_manifest.json` records every moved file,
its old/new location, SHA-256 and the five user-protected files. Large archived
Step 2.0 feature CSVs retain their local-only Git exclusion.

## Archived groups

- `Step 2.A Route Subtypes/`: v1 code, tests and results. The current v2 pipeline
  reads the archived assignments/validation for its v1–v2 comparison.
- `Supplementary Data Analysis/CodexOutput v2/`: previous visual builder/results.
- `Supplementary Data Analysis/build_route_analysis.py`,
  `verify_route_analysis.py`, the former README and
  `CodexOuput_route_visualizations/`: first input-only route analysis, including
  historical context comparisons and animations not all reproduced in v3.
- `separation_baseline/scripts/build_route_layer.py`,
  `football_network/routes.py`, `outputs/route_layer_sample/`: early five-game
  route-layer prototype, superseded by the official full-season Step 2.0 builder.
- `Step 2.0 Route Dataset/`: one-off 20° gate audit and 20°–35° migration verifier,
  original review, original five-game outputs and original 20° feature table.

Archived scripts retain their original defaults and provenance. Consult their
recorded source context and use explicit paths/fresh destinations if reproducing
a historical run. They must not be used to regenerate current deliverables.

## Active workflow

The official data builder, defenders, geometry module, 35° feature builder and
their tests stay in `../Step 2.0 Route Dataset/`. Current independent verification
is `verify_break_features.py`, which requires no historical feature table.
Subtype v2 stays in `../Step 2.A Route Subtypes v2/`. Visual v3 stays in
`../Supplementary Data Analysis/CodexOutput v3/`, with its reviewed gallery in
`full_season/index.html`. The separation baseline generator, distance functions,
validators and baseline receiver-play CSV remain active at their original paths.

`../archive_paths.py` resolves an old missing EDA path to the matching archived
path. Active files take precedence; unavailable inputs fail loudly. Current
verifiers use this resolver so saved historical validation JSON files do not
need to be rewritten after a move.

The user explicitly protected these five files, all left in their original
locations and verified unchanged: `EDA/NFLPlayerData.ipynb`,
`EDA/PlayAnimator.ipynb`, `EDA/README.md`, `EDA/phase1_first_game_eda.ipynb`,
`EDA/tracking_5_games.csv`.
