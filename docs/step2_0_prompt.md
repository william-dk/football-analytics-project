# Step 2.0 build prompt

Hand this to a coding assistant. Replace `<TRAIN_DIR>` with the local path to the official `train/` folder.

````text
# Task: Build the Step 2.0 route-level dataset (merge tracking + supplementary)

## Context
Research project on NFL receiver separation by route type, using NFL Big Data Bowl 2026 Analytics data (2023 season, Weeks 1–18). The study centers on TARGETED pass plays: separation at the throw, during ball flight, and at the catch point. Route types come directly from the supplementary labels, which cover every targeted receiver, so no route classifier is built here. This step merges tracking with the supplementary labels and context and produces reusable tables that later steps read directly. Step 1 (EDA/Supplementary Data Analysis/build_route_analysis.py) normalized targeted routes for plotting but saved no frame-level table; keep its flip rule.

## Inputs
- Official weekly tracking: <TRAIN_DIR>/input_2023_w01.csv … input_2023_w18.csv (pre-throw) and output_2023_w01.csv … output_2023_w18.csv (post-throw). Use these raw official files, not the combined extract 2025_football_player_tracking.csv.
- EDA/Supplementary Data Analysis/supplementary_data.csv: one row per play, 41 columns, 18,009 rows, two seasons: 2023 (14,108 plays, all with tracking) and 2024 Weeks 14–18 (3,901 plays, no tracking). Keep season == 2023 only.
- Reuse code where it fits: EDA/separation_baseline/football_network/routes.py (first version, on branch claude/exciting-fermi-worf7f, built on the combined extract) and football_network/separation.py (nearest_defenders()).

## Data facts (verified; do not contradict)
- Grain: one tracking row = (game_id, play_id, nfl_id, frame_id), 10 Hz. Join supplementary on BOTH game_id and play_id; play_id alone repeats across games.
- Input roles: Passer, Targeted Receiver, Other Route Runner, Defensive Coverage.
- Output files contain x/y only (no s, a, dir, o) for the targeted receiver and about 2.4 nearby defenders, from the throw until the ball arrives. Confirm on the first run whether output frame_id restarts at 1 and place output frames after the last input frame accordingly.
- Frame 1 is a reliable snap proxy. The last input frame is the throw proxy.
- route_of_targeted_receiver describes ONLY the targeted receiver. Never assign it to other route runners.
- Raw official files: left-moving plays need x = 120 − x and y = 53.3 − y; LOS x = 120 − absolute_yardline_number on left plays, absolute_yardline_number on right plays. Do not use dir or o; derive headings from x/y.
- Many breaks happen after the throw (timing routes), so pre-throw break detection fails for most routes. Breaks are measured on full targeted paths.

## Normalization (route frame)
1. Flip left-moving plays as above, so the offense always moves toward +x. Apply the same transform to defenders.
2. Ball proxy: the Passer's y at frame 1.
3. For each route runner: origin at his frame-1 position; x_norm = downfield yards; y_norm = lateral yards × side, where side = sign(y0 − qb_y0), so +y points toward his own sideline. If there is no passer or y0 == qb_y0, use side = +1 and set side_flag.
4. Also flag RB/FB and anyone within 2 yd laterally of the ball proxy (less reliable side).
5. t = seconds since frame 1.

## Per-frame features
vx, vy (differences of x_norm/y_norm × 10), s and a (given on input frames; compute from positions on output frames), heading_sin, heading_cos (NaN when speed < 1 yd/s), turn_rate (deg/s, wrapped to ±180°).

## Outputs, in priority order (Parquet if available, else gzipped CSV)
1. target_full_paths (PRIMARY): one row per frame for each of the ~14,105 targeted receivers, input + output frames concatenated, with phase ("input"/"output"). Columns: game_id, play_id, nfl_id, frame_id, t, phase, x_norm, y_norm, vx, vy, s, a, heading_sin, heading_cos, turn_rate, raw (flipped) x, y, side.
2. routes (PRIMARY): one row per route runner per play; targeted rows are the analysis set, untargeted rows supply alignment context. Columns:
   - identity: game_id, play_id, nfl_id, player_name, player_position, is_targeted
   - timing: n_input_frames, throw_time_s, n_output_frames and ball_flight_s (targeted only)
   - context at snap: in_motion_at_snap (s > 1 yd/s at frame 1), lateral_from_ball_yd, sideline_dist_yd, depth_from_los_yd, alignment (outermost / slot / inline / backfield: rank route runners on the same side of the ball by lateral distance; inline = within 4 yd of the ball and at the LOS; backfield = at or behind the QB's depth), side_flag
   - fixed-time features from the full path for targets and the input path otherwise: x_norm, y_norm, heading, speed at t = 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 s (NaN beyond the observed path)
   - throw-time features: x_norm, y_norm, heading, speed at the last input frame; heading and speed change over the last 0.3 s before the throw
   - break (targeted, from full path): break_detected (turn of at least 35° over 0.3 s windows), break_depth_yd, break_time_s, break_angle_deg, speed before and after the break, and break_before_throw (bool)
   - supplementary columns: season, week, down, yards_to_go, quarter, offense_formation, receiver_alignment, play_action, dropback_type, pass_result, pass_length, team_coverage_man_zone, team_coverage_type, defenders_in_the_box, play_nullified_by_penalty, yards_gained, expected_points_added
   - route_label = route_of_targeted_receiver on targeted rows, NaN otherwise
3. route_frames (LOWER PRIORITY, for an optional extension to untargeted routes): one row per route runner per pre-throw frame, same columns as target_full_paths without phase.

## Merge rules
- pd.merge(..., on=["game_id", "play_id"], validate="many_to_one", indicator=True); raise if any 2023 tracking play fails to match.
- Drop overlapping columns before merging, or use explicit suffixes; no silent _x/_y columns.
- Keep play_nullified_by_penalty == "Y" rows but leave them identifiable (expected: 1 play).

## Validation (write checks.json; fail loudly on violations)
1. Every 2023 play matches supplementary; every route runner appears exactly once in routes; targeted count ≈ 14,105.
2. All routes start at (0, 0) at t = 0.
3. Median QB depth behind LOS at frame 1 is 1–7 yd for left and right plays separately (expected ≈ 5 yd).
4. Targeted, excluding SCREEN and FLAT: ≥ 95% have x_norm > 0 at t = 1.0 s.
5. Mirroring at the catch point (last output frame): OUT, CORNER, FLAT mostly y_norm > 0; IN, SLANT, POST mostly y_norm < 0 (five-game sample: OUT 92%, CORNER 88%, FLAT 91% outward; IN 96%, SLANT 100%, POST 100% inward). Report the same shares at the throw; low values there (CORNER ≈ 25%) are expected truncation.
6. Left vs right plays: per-label median catch point of normalized targeted paths agrees within about 1 yd.
7. Distance invariance: nearest-defender distance recomputed in normalized coordinates equals the raw value to 1e-6.
8. Report counts: targeted routes by label; side_flag; break_detected and break_before_throw shares by label (from full paths); frame gaps and duplicate keys (expected 0).

## Deliverables
- football_network/routes.py updated to read the raw official files; scripts/build_route_layer.py with --input-dir, --supplementary, --out flags and a --skip-route-frames option; the output tables; checks.json; a README section defining every column.
- Run and verify on a 5-game subset first, then on all 18 weeks.
- Do not modify the Step 1 folder or the supplementary CSV.
````
