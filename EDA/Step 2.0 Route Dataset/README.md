# Step 2.0: reusable route dataset

Builds from the **official 2023 weekly input/output CSVs**, joining supplementary context on both `game_id` and `play_id`. It does not read the combined tracking extract or classify routes. Existing Step 1 files and the earlier combined-extract route module are preserved.

```powershell
python build_route_dataset.py --train-dir "PATH/TO/OFFICIAL/train"
python -m unittest discover -s . -p "test_*.py" -v
```

Requires Python, pandas and NumPy. Uses Parquet with Zstandard when PyArrow is available; otherwise gzip CSV. `--format csv.gz` forces the fallback. Supplementary and output locations can be overridden with `--supplementary` and `--output-dir`. `--weeks 1 2` builds a bounded subset; the default is all 18 weeks. `--skip-route-frames` omits the optional pre-throw table.

## Delivered tables

The current `data/` directory contains six ordinary, uncompressed `.csv` tables, plus documentation and audit records. Each table combines all 18 weeks. The directory is ignored by Git. The builder still writes Parquet or gzip CSV on a fresh run; that build-time format is separate from the current storage layout.

```text
Step 2.0 Route Dataset/
  README.md                         # Build and methods guide
  build_route_dataset.py
  defender_dataset.py
  verify_route_dataset.py
  test_route_dataset.py
  test_defender_dataset.py
  data/
    README.md                       # Detailed dataset/field guide
    target_full_paths.csv            # A: full targeted receiver paths
    routes.csv                       # B: route-level features and context
    route_frames.csv                 # C: all runners, input frames only
    target_defender_frames.csv       # D: observed defenders per target frame
    target_separation_frames.csv     # E: nearest-defender separation per frame
    target_separation_summary.csv    # F: targeted-route separation summaries
    manifest.json
    validation.json
    weekly_audit.csv
    target_count_exceptions.csv
    target_label_missing.csv
    defender_validation.json
    defender_validation_5games.json
    defender_baseline_mismatches.csv
```

`manifest.json` and `defender_validation.json` retain the original `csv.gz` build-format records. They are historical build/validation records, not current filename indexes. Read the ordinary CSV filenames shown above; no decompression is needed. The original `verify_route_dataset.py` follows the historical manifest and will need matching filenames or an updated reader before it can validate the current storage layout.

- `target_full_paths`: primary frame table, keyed by `(game_id, play_id, nfl_id, frame_id)`, containing all targeted input and output observations. Includes `phase`, normalized and flipped field coordinates, velocity, speed, acceleration, headings, turn rate, side and side flag.
- `routes`: one row per route runner per play. `is_targeted` identifies the analysis cohort. Includes all requested snap, timing, fixed-time, throw, break and supplementary features. `route_label` is populated only for targeted receivers.
- `route_frames`: optional input-only frame table for all route runners, with the same feature columns except `phase`.
- `weekly_audit.csv`: source/output rows, cohort counts, frame restart checks, missing passer counts and break counts by week.
- `target_count_exceptions.csv`: plays with zero or multiple targeted receivers. Preserve valid tracking rows; do not fabricate a receiver to match the supplementary play count.
- `target_label_missing.csv`: targeted receivers whose supplementary source label is blank. Preserve their paths but exclude them when a later analysis requires a known route type.
- `manifest.json`: source paths/sizes, cohort, format, column lists and row counts. Read this first downstream.
- `validation.json`: results of a separate read-back validation, written only by the verifier.

```python
from pathlib import Path
import pandas as pd

root = Path("data")
routes = pd.read_csv(root / "routes.csv", low_memory=False)
targets = routes.loc[routes.is_targeted].copy()
paths = pd.read_csv(root / "target_full_paths.csv", low_memory=False)
separation = pd.read_csv(root / "target_separation_frames.csv", low_memory=False)
summary = pd.read_csv(root / "target_separation_summary.csv", low_memory=False)
```

## Coordinates and timing

The supplementary scope is `season == 2023`; all supplied tracking plays must match that scope. The 2024 supplementary rows have no tracking and are excluded. Input frame 1 is the snap proxy and the last input frame is the throw proxy; neither is an event tag. The observed final output is the ball-arrival/catch-point proxy, whether the pass was completed or not.

Left-moving raw input **and output** players, including defenders, are rotated with `x = 120-x`, `y = 53.3-y`, matching Step 1's flip rule. The field coordinates saved as `x,y` are these flipped coordinates, in yards. LOS is `120-absolute_yardline_number` on left plays and `absolute_yardline_number` on right plays. Directions `dir,o` are never read.

The QB's frame-1 y is the ball proxy. Each runner has origin at his own frame-1 x/y, with downfield `x_norm = x-x0`, and `y_norm = (y-y0)*side`. `side = sign(y0-qb_y0)`, so positive lateral movement points toward his initial sideline. Missing passer or exact y equality uses side +1. `side_fallback` records these cases. `side_flag` also includes RB/FB and initial lateral distance at most 2 yd. Missing QB means alignment and lateral distance stay missing, rather than imputing an alignment.

Every output player must start at output frame 1 and have continuous frames; the build verifies this in every week. Output is remapped to `last_input_frame + source_output_frame_id`. The first output is therefore 0.1 seconds after the throw. `t = (frame_id-1)/10`; `throw_time_s = (n_input_frames-1)/10`; `ball_flight_s = n_output_frames/10`. Target output length must equal `num_frames_output` in the input metadata. Untargeted output timing fields are null.

## Feature definitions

`vx,vy` are backward coordinate differences times 10, including across the throw boundary. They are missing at frame 1. Input `s,a` preserve the official values. Output speed is the norm of positional velocity, and output acceleration is the signed difference in position-derived speed times 10, including a position-derived preceding speed at the boundary. Units are yd/s and yd/s². A very short input path may leave the first output acceleration missing. Mixed input sensor speed and output positional speed should be considered when interpreting changes across phases.

Headings derive from positional velocity in route coordinates, with 0° downfield and positive angles toward the initial sideline. Sine/cosine, and any dependent heading change, are missing when `s` is below 1 yd/s, when positional displacement is exactly zero, or when the first difference is unavailable. Turn rate wraps consecutive heading differences to [-180°,180°) before multiplying by 10; units are deg/s. No headings are inferred from `dir,o`.

Snap features use the actual frame-1 row. `lateral_from_ball_yd` is an absolute distance; `sideline_dist_yd` is the distance to the nearest field sideline; `depth_from_los_yd` is signed, positive downfield. Motion is official input `s > 1`.

Alignment is a proxy with explicit precedence: backfield if the runner starts at or behind the QB's x; otherwise inline if within 4 yd laterally of the ball and within 1 yd of the LOS; otherwise outermost for the maximum lateral distance among route runners on the same side of the ball, and slot for the rest. Ties at the maximum are both outermost. Ranking includes every supplied route runner, including backfield players; it is not a personnel/formation classification.

Fixed-time columns are named `x_norm_0_5s`, `y_norm_0_5s`, `heading_0_5s`, `speed_0_5s`, etc., through `3_0s`. They use exact 10 Hz observations, full paths for targets and input-only paths for other runners, with NaN beyond observations. Heading columns are degrees. Throw columns end in `_at_throw`, evaluated at the last input frame. `heading_change_last_0_3s_deg` is the wrapped endpoint difference, and `speed_change_last_0_3s` the speed endpoint difference over exactly 0.3 s before the throw; both stay missing when unavailable.

Break detection uses full targeted paths, including ball flight. Smooth x/y with the symmetric kernel [1,4,6,4,1]/16 and edge padding. At each candidate compare 0.3 s displacement windows before and after the candidate; both must cover at least 0.6 yd and the absolute signed turn must be at least 35°. Choose the strongest absolute qualifying turn; ties resolve to the earliest. Report depth/time from the unsmoothed observation and the signed angle from smoothed displacements. Before/after speeds average the three per-frame speeds ending at / following the break. `break_before_throw` is strictly earlier than the throw; it is null when no break is detected. All break fields are null for untargeted runners. This is an exploratory geometric break proxy, not a charted break or a route classifier.

Supplementary outcome and coverage fields describe the play/target context. They are attached to all route rows for context; they must not be interpreted as individual outcomes for untargeted runners. Only `route_label` is assigned to the targeted receiver.

## Validation

The builder rejects duplicate frame keys, gaps, invalid coordinates, missing source files, unknown output players, unequal player throw frames, unexpected output numbering, inconsistent target flight lengths, duplicate supplementary joins, and mismatched play/week coverage. Regression fixtures cover left/right geometry, defender flipping, shared play IDs across games, target-only labels, phase timing, late breaks, side fallbacks and slow noise. The read-back verifier checks every primary row's geometry derivatives and timing, all fixed-clock target positions, route-frame coverage, and independently reconstructs a raw input/output target path from each requested week.

The original three tables retain their original columns. The defender extension now adds three separate tables with observed coverage defenders, frame-level separation and target-level summaries; see below. Post-throw defenders are only the supplied subset, and support counts remain explicit.

## Local full-season build

The 2023 Weeks 1–18 build contains 14,108 plays and targeted receivers, 64,752 total route runners, 557,274 targeted full-path frames, and 1,821,157 pre-throw route frames. There are 14,105 labeled targets and three targets with blank source route labels, recorded in `target_label_missing.csv`. Every play has exactly one tracked targeted receiver. Eleven route rows lack a frame-1 passer and use the documented fallback. The 3,901 supplementary 2024 plays are excluded. The original build used gzip CSV because PyArrow was unavailable; all six delivered tables are now stored as ordinary CSV.

## Defender extension (Tables D/E/F)

All six tables in the current data directory are ordinary `.csv`; their historical build records still record `csv.gz`. Defender-only source discovery checks actual files rather than using that historical extension. It rejects ambiguous duplicate source formats. If rebuilding, use a separate output directory to avoid leaving a second format beside the current tables.

```powershell
# First validate the first five games of week 1 in a separate output folder.
python build_route_dataset.py --train-dir "PATH/TO/OFFICIAL/train" --defenders-only --weeks 1 --max-games 5 --existing-data-dir "data" --output-dir "PATH/TO/TEMP/five_games"
# Then rebuild all 18 weeks into a separate folder, keeping the stored CSVs intact.
python build_route_dataset.py --train-dir "PATH/TO/OFFICIAL/train" --defenders-only --existing-data-dir "data" --output-dir "PATH/TO/NEW/defender_build"
```

`--skip-defenders` skips the extension on a fresh original build. Otherwise a normal fresh build adds the extension after the original pipeline. `--baseline` overrides the reference receiver-play metrics file. The optional original route-frame table is not required by the extension, but is included in preservation checks whenever present.

- `target_defender_frames.csv`: 3,065,233 observed target-frame-defender rows. Includes field kinematics, receiver-relative displacement, travel-direction projections, closing speed and official-output membership.
- `target_separation_frames.csv`: 557,274 target frames, preserving the entire receiver grid and missing distances where no defender is observed. Includes nearest-all/DB distances, support counts, trailing/leverage/cushion features and nearest switches.
- `target_separation_summary.csv`: 14,108 target routes with original target-route context plus exact throw/arrival distances, flight changes, fixed-throw-defender sensitivity, phase switch counts, input extrema and break separation.
- `defender_validation.json`: full-season validation, column lists, counts, defender support distributions and input preservation hashes.
- `defender_validation_5games.json`: successful five-game validation record before the full run.
- `defender_baseline_mismatches.csv`: header-only on this successful run.

Full-season validation checked 500 random frames from raw unflipped positions (maximum distance error 1.36e-14 yards), compared all 14,108 targets to the existing throw-time DB baseline (zero mismatches; three consistently missing DB distances), and read back serialized output keys/counts. SHA-256 checks confirmed that the original three tables were unchanged. Nine regression tests cover the original route logic and defender geometry, ties, source selection, gap handling and slow receivers.

The mean observed defender count is 6.708 per input receiver frame and 2.510 per output frame. The throw-time nearest defender is absent from the official output set in 1,511 plays (10.711% of 14,107 plays with a known throw-time nearest defender); one flag is unknown. At the receiver's final output frame, 1,142 plays have no observed defender, so arrival distance remains missing. Route medians reproduce the expected descriptive ordering: GO 1.588 yards, ANGLE 6.702, FLAT 7.678 and SCREEN 9.769. This ordering is reported, not enforced.

Detailed definitions and reading examples are in `data/README.md`, entirely in English. All generated data remains excluded from Git. No shared Google Drive upload was performed; the output location is the existing OneDrive-backed data directory.
