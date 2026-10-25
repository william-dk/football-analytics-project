# Step 2.0 add-on prompt: defender table (Table D)

Hand this to a coding assistant after the three route tables exist. Replace `<TRAIN_DIR>` with the local path to the official `train/` folder.

````text
# Task: Add Table D (defender frames around each targeted receiver) to the Step 2.0 route dataset

## Context
The Step 2.0 builder (EDA/Step 2.0 Route Dataset/build_route_dataset.py) already produces routes.csv, target_full_paths.csv and route_frames.csv for the 2023 season. Those tables contain no defenders, so separation cannot be computed. Step 3 measures, within each route type, the targeted receiver's separation at the throw, during ball flight, and at the catch point, and splits it into trailing / leverage / cushion components. This task adds one defender table and a per-frame separation summary, built with EXACTLY the same flip, time-stitching and side rules as the existing tables. Extend the existing builder; do not change existing tables or their columns.

## Inputs
- Official weekly files: <TRAIN_DIR>/input_2023_wXX.csv and output_2023_wXX.csv, Weeks 01–18.
- The existing target_full_paths.csv and routes.csv (for the targeted receiver's frames, side, and play keys).

## Verified facts (do not contradict)
- Defenders = player_side == "Defense". Input frames: player_role == "Defensive Coverage", about 7 per frame (pass rushers are not tracked). Output frames: only the defenders the official output file contains (about 2.4 per play, range 1–6), x/y only.
- Output frame_id restarts at 1 in every week; stitched frame_id = last input frame_id + output frame_id (same as target_full_paths).
- Left-moving plays: x = 120 − x, y = 53.3 − y. Use the flipped field coordinates for all distances.
- A defender can appear in input frames but not in output frames. Never forward-fill or impute missing defenders.

## Output 1: target_defender_frames (one row per targeted play × frame × defender)
Keys: game_id, play_id, frame_id (stitched), receiver_nfl_id, defender_nfl_id. Columns:
- phase ("input"/"output"), t (seconds since frame 1, same formula as target_full_paths)
- defender_name, defender_position, is_db (position in CB, DB, S, FS, SS)
- x, y: flipped field coordinates
- s, a: official on input; derived from positions on output (same method as target_full_paths)
- vx_field, vy_field: velocity in flipped field coordinates from position differences
- dist_to_receiver: Euclidean distance to the targeted receiver in the same frame
- rel_x, rel_y: defender position relative to the receiver in that frame, in the receiver's route frame (rel_x = x_def − x_rec; rel_y = (y_def − y_rec) × side, where side is the receiver's side from routes.csv), so +rel_x is downfield of the receiver and +rel_y is toward the receiver's own sideline
- along_track, cross_track: the receiver-to-defender vector projected onto the receiver's direction of travel in that frame (from the receiver's vx, vy) and onto its perpendicular, signed so that +cross_track points toward the receiver's sideline. NaN when receiver speed < 1 yd/s.
- closing_speed: rate of change of dist_to_receiver (negative = defender closing), from consecutive frames of the same defender; NaN when either frame is missing
- in_output_set: whether this defender appears in the official output file for the play

## Output 2: target_separation_frames (one row per targeted play × frame)
Keys: game_id, play_id, frame_id, receiver_nfl_id. Columns:
- phase, t, n_defenders_observed, n_db_observed
- nearest_def_id, nearest_def_position, sep_nearest (min distance over all observed defenders)
- nearest_db_id, sep_nearest_db (min over DBs only)
- trailing_component, leverage_component, cushion_flag for the nearest defender: classify the nearest defender as trailing (along_track < 0), cushion (along_track > 0), and record |cross_track| as leverage distance
- nearest_switch: nearest_def_id differs from the previous frame
Ties resolve to the smaller defender_nfl_id. When no defender is observed, separation stays NaN (never 0 or a large default).

## Output 3: columns appended to a NEW file target_separation_summary.csv (one row per targeted route; do not edit routes.csv)
- sep_at_throw, sep_db_at_throw: at the last input frame
- sep_at_arrival, sep_db_at_arrival: at the last output frame
- sep_change_flight = sep_at_arrival − sep_at_throw
- nearest_at_throw_in_output: whether the nearest defender at the throw is in the output set (if not, arrival separation is measured against a different defender group; keep the flag)
- sep_fixed_throw_def_at_arrival: distance at arrival to the defender who was nearest at the throw, when that defender is in the output set (sensitivity measure)
- n_defenders_at_throw, n_defenders_at_arrival, nearest_switches_input, nearest_switches_output
- max_sep_input, min_sep_input, sep_at_break (at break_time_s from routes.csv, when a break was detected)

## Validation (write defender_validation.json; fail loudly)
1. Keys unique in both frame tables; every targeted route in routes.csv with output frames has rows in Outputs 2 and 3.
2. Recompute sep_nearest for 500 random frames from raw (unflipped) coordinates: absolute error < 1e-6 yd (distance is invariant to the flip).
3. Input-frame sep_nearest_db at the last input frame matches the existing separation baseline (EDA/separation_baseline/outputs/separation_baseline/receiver_play_metrics.csv, input_end_db_yards) for targeted receivers, to 1e-6 yd. Report any mismatches.
4. sep_nearest <= sep_nearest_db wherever both exist.
5. Report the distribution of n_defenders_observed by phase (expected about 7 input, about 2.4 output) and the share of plays where nearest_at_throw_in_output is False.
6. Sanity: median sep_at_throw by route label should reproduce Step 1 order (GO lowest about 1.6 yd, FLAT/SCREEN/ANGLE highest). Report, do not enforce.

## Deliverables
- Builder extension and a --skip-defenders flag; the three new files (Parquet preferred, else gzip CSV); defender_validation.json; README section in the same style as the existing data README.
- Run on a 5-game subset first, then all 18 weeks.
- Keep large data files out of GitHub (shared Drive instead).
````
