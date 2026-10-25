# Route visualizations v3

Open `index.html` for the atlas, corrected route distributions, and representative
completions and incompletions. This version copies the v2 visual logic into
`visual_helpers.py` and uses the reviewed 35° geometry. The v2 files are untouched.
The final reviewed delivery is in `full_season/`; root-level initial-render files
are preserved. The final layout moves the ball key below the arrival-separation
box to prevent overlap. Its five-game precursor is `validation_5games_layout_v2/`.

## Changes from v2

- Route-frame lateral coordinates are `(y − y0) × side_corrected` for every
  receiver path and ball landing point. The displayed own sideline is the boundary
  on that corrected side, rather than the nearest boundary. Field views retain
  the canonical right-facing field coordinates, as in v2.
- Break statistics come exclusively from the new feature table at 35°; no legacy
  break column is used. A cut's missing values are not replaced in the summaries.
- Representatives are selected independently within completions (`C`) and
  incompletions (`I`), ranked against the same full-label median as v2. Interceptions
  remain visible in distributions but are not called representative incompletions.
- Representative landing-to-arrival eligibility is 10 yards rather than 5.
  Penalty and flagged-side exclusions remain as in v2. Time/depth IQR and observed
  arrival-defender conditions are relaxed in that order if necessary. All
  relaxations and absent eligible result groups are recorded. If no incompletion
  is eligible, only the completion is shown.
- Accepted v2 subtype medians appear in distinct dash-dot colors on distribution
  plots for split labels. These use actual snap time and at least 50% subtype
  support as in Part 2, with depth expressed from LOS to match the v3 axes.
- `checks.json` adds corrected-side counts, FLAT/OUT inside shares before and
  after correction, every selected pass result, and hashes proving both source
  files and prior-version outputs are unchanged.

## Reading the figures

The horizontal axis in distributions is corrected lateral displacement in yards,
positive toward the corrected own sideline. The vertical axis is depth from the
line of scrimmage, in yards, as in v2. Receiver-snap-relative break depth therefore
need not equal LOS-relative plot depth. Solid blue paths precede the last input
frame (throw proxy), and dashed blue paths continue during ball flight. Orange
marks the throw; green circles mark completed-pass arrivals; red crosses mark
other arrivals. Gold stars show ball landing. The sideline band is the 10th–90th
percentile across plays, and the dotted boundary is its median.

Up to 200 routes are sampled per label, with completion proportions preserved
by largest-remainder allocation. All statistics and medians use the full label.
Paths are interpolated only over observed times onto the shared 0.1-second
clock. Medians stop when support first falls below max(5, ceil(25% × label_n));
for a five-game label with fewer than five routes, the available label count
is the minimum. Durations are never stretched. Landing stars more than 40 yards
from the receiver arrival are omitted from spaghetti plots; that v2 drawing
rule is separate from the 10-yard representative eligibility rule.

Representative field views show the receiver, observed coverage defenders,
input-only QB and other runners, ball landing and nearest-defender separation.
Nearest defenders can differ between throw and arrival. The dotted ball flight
is a straight-line approximation, not tracked ball motion. The atlas uses equal
yard scales and common limits across labels. Arrival is the last tracked frame,
not a verified catch event. Separation is descriptive, not a causal quality score.

## Files and validation

The output filenames follow v2: `spaghetti/`, `representative/`, `index.html`,
`label_summary.csv`, `representative_plays.csv`, `sampled_routes.csv`,
`median_path_support.csv` and `checks.json`. Selected examples include their
pass result, receiver, game/play IDs, distance to median and any relaxation.
`verification.json` independently audits delivered coordinates, medians, sampling,
representative selection and images. `validation_5games/` holds the smoke run
performed before the full season; labels/result groups absent from that subset
are recorded rather than invented.

The builder first requires passed Part 1 and Part 2 validation records. It
checks completion ball-landing geometry separately for left/right plays before
drawing, compares selected receiver output paths with official tracking, verifies
drawn nearest-defender distances against Table F, verifies LOS depth, and checks
source and prior-version hashes. Validation status is `passed` only with an
empty failures list. Existing outputs are never replaced.

```powershell
python -m unittest test_route_visuals_v3.py
python build_route_visuals_v3.py --five-games `
  --break-dir "../../Step 2.0 Route Dataset/break_features_v2_35deg/validation_5games" `
  --subtype-dir "../../Step 2.A Route Subtypes v2/validation_5games" `
  --out validation_5games_NEW
python build_route_visuals_v3.py --out full_season_NEW
```

Use `--data-dir`, `--break-dir`, `--subtype-dir`, `--train-dir`, and
`--supplementary` to override the documented local source paths. The full run
requires the delivered `validation_5games_layout_v2/checks.json` to pass first;
`--subset-validation-dir` can select another validated smoke run. Dependencies
are NumPy, pandas and Matplotlib; seed is 20261025 throughout.
