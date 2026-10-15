# Separation baseline: professor-feedback steps 1–4

Source: Local official BDB 2026 Analytics input_2023_w01..w18.csv + supplementary_data.csv. This run uses all 18 available 2023-season weekly input partitions.

## Data coverage and timing

- 4,880,579 tracking rows; 14,108 plays; 272 games.
- 64,752 receiver-play observations; 64,746 satisfy summary eligibility.
- 38,002 eligible WR routes; 8,611 eligible WR targets.
- Input data has no snap, release, arrival, football trajectory, or complete all-22 event tracking. The last input frame is only a release-adjacent boundary proxy. A true snap-to-throw mean and arrival separation cannot be reported from these files.
- Existing supplementary route labels describe the targeted receiver only. Never transfer them to other route runners.

## Definitions

Nearest DB = minimum Euclidean x/y distance to supplied defensive-coverage players with position CB/DB/S/FS/SS. Secondary comparison = nearest supplied defensive-coverage player, including linebackers. Neither identifies a coverage assignment or guarantees all 11 defenders are present.

Main metrics: input-end distance and arithmetic mean across all supplied input frames (10 Hz). Time averages may mix different observed window lengths. Player means give equal weight to eligible receiver-plays. Target and all-route summaries remain separate.

Eligibility: supplementary match, penalty flag N, receiver frames 1..input-end with no gaps, DB observed every frame, finite coordinates. Missing comparisons remain missing. Entire plays with nonfinite coordinates are excluded. Out-of-field positions are flagged and retained without clipping, as players may run out of bounds. Raw records and eligibility flags remain inspectable.

## Quality and independent validation

- Duplicate/null key checks passed weekly; weekly play partitions do not overlap.
- 0 unmatched receiver-plays; 0 incomplete input tracks; 6 incomplete DB windows.
- 0 nonfinite-coordinate rows; 0 out-of-field rows; 156 receiver frames without a DB.
- Independently recomputed nearest IDs and distances for 432 spread-out frame observations; maximum error 0 yards. Rotation invariance checked on one play per week; nearest-coverage distance <= nearest-DB distance checked for every comparable frame.

## Descriptive results, not route-running ability rankings

- WR all-route mean input-end DB distance: 3.735 yards.
- WR targets-only mean input-end DB distance: 4.168 yards.
- All-position targets: completed 6.683 vs incomplete 4.244 yards. These are unadjusted descriptive means, not causal effects or a prediction model.
- In 4,354 eligible target plays, the nearest tracked coverage player at input end was not a DB.
- Pass-result counts: {'C': 9736, 'I': 4032, 'IN': 337}. Incomplete passes remain in the analysis; interceptions remain a separate source label. No invented drop/throw-error labels.
- Player summary files include count, games, mean, median and quartiles. A 30-observation display threshold is an exploratory convention, not a reliability guarantee.
- Small-sample illustration: the largest unfiltered WR target mean belongs to Derius Davis, 11.471 yards across just 6 eligible targets. This is not evidence of the best route runner.

## Scope and next check-in

The dataset selects plays and players; do not generalize to all 2023 NFL routes. Untargeted receivers have no individual completion outcome. Each example is deterministically chosen for illustration, not representative proof. Four cases are nearest to their route/outcome median input-end separation; two highlight the largest DB-vs-coverage endpoint gap and most DB switches.

Next: discuss input-window timing with the professor, review the six trajectory examples, and decide whether existing target route labels suffice before attempting clustering. No clustering or success classifier was added.

## Reproduce

Run `python scripts/run_separation_baseline.py` from the project root with NumPy and pandas installed. See `frames/week_*.csv.gz`, `receiver_play_metrics.csv`, `players_*.csv`, `target_context_summary.csv`, `weekly_quality.csv`, `cases/`, and `audit_and_results.json`. Existing source data and prior analyses are unchanged.
