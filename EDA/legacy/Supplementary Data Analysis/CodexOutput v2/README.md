# Route Visualizations v2

Full pre-throw and post-throw paths for **14,105 labeled targeted receiver routes**, covering the 2023 season, Weeks 1–18. This folder is the requested output destination and replaces `EDA/Route Visualizations v2/` in the original task description. Step 1 and Step 2.0 source files are not edited.

## Start here

Open **[index.html](index.html)** for a local gallery. Each route has a distribution plot and a real-field example, and every image links to its full-resolution PNG. No web server or internet connection is needed.

Read **[VISUALIZATION_READING_GUIDE.md](VISUALIZATION_READING_GUIDE.md)** for the full English interpretation guide, symbol definitions, route-label orientation, and limitations.

## Files

- `spaghetti/route_<LABEL>.png`: 12 route distributions, each with up to 200 sampled paths. Solid blue is before the throw; dashed blue is after. Orange is the throw proxy, green circles/red crosses are receiver arrival points colored by pass result, and gold stars are ball landing points. The dark median uses every route in the label.
- `spaghetti/all_routes_overview.png`: 3 × 4 atlas with identical limits and equal yard scales. Shows all receiver arrival points and the same-time median paths.
- `representative/route_<LABEL>_<game_id>_<play_id>.png`: 12 selected plays on the real field, including coverage defenders, QB, other runners, ball landing, and nearest-defender separation.
- `label_summary.csv`: one row per label; full-cohort statistics, non-missing counts, sideline quantiles, sample size, and landing exclusions.
- `representative_plays.csv`: selected receiver IDs/names, selection distances, relaxed conditions, results, separation, timing, and selection eligibility fields.
- `checks.json`: cohort counts, direction-specific ball-coordinate validation, exclusions, representative-selection audit, passer availability, and official-output consistency checks.
- `verification.json`: independent delivery audit of the statistics, every median coordinate and support count, reproducible sampling, PNG resolution, local gallery links, and source-file integrity.
- `sampled_routes.csv`: exact keys of every sampled route for reproducibility.
- `median_path_support.csv`: 0.1-second median coordinates and support, including tails that are excluded from the figures.
- `build_route_visuals_v2.py`: reusable builder. Uses only pandas, numpy, matplotlib, and the Python standard library. All PNGs are saved at 160 dpi.

## Methods

**Cohort.** Read the ordinary Step 2.0 CSVs using `low_memory=False`. Keep targeted routes with a non-missing supplementary route label. The three unlabeled targets are excluded. All-route statistics retain flagged-side and penalty-nullified plays; the stricter exclusions below apply only to representative selection.

**Coordinates.** All field paths face right. In official files, left-moving plays use `x = 120 - x`, `y = 53.3 - y`, including ball landing. Table A and D are already flipped and are not flipped again. LOS comes from official `absolute_yardline_number`, applying the corresponding x flip. Route plots use `(y - initial_y) * side` laterally and `x - LOS` for depth. The own-side sideline is recomputed from initial field y and side; the source's nearest-sideline distance is not substituted for it.

**Validation before drawing.** The median landing-to-arrival distance for completions must be below 2 yards separately for left and right plays. Otherwise the builder writes the failed check and stops. It also checks LOS depth against Step 2.0, verifies representative receiver output coordinates against the official output files, and checks drawn nearest-defender distances against Table F.

**Sampling.** Up to 200 routes per label, seed `20261025`. Use largest-remainder allocation across complete versus non-complete routes, then sample without replacement within each group. The completion share is preserved to the nearest possible sampled count. Counts and statistics always use the entire label.

**Median.** Interpolate every route onto a shared 0.1-second clock, only inside its own observed interval. Take coordinate-wise medians. Stop at the first grid point with fewer than `max(5, ceil(0.25 * label_n))` observed paths. Durations are never stretched. The median is a pointwise summary, not necessarily a physically observed route.

**Ball exclusions.** Omit a sampled landing star and its arrival connector when landing is more than 40 yards from receiver arrival. Keep that receiver path, endpoint, and all-route statistics. Counts in the figure refer to the sampled routes; `checks.json` also records the full-cohort count.

**Representative selection.** For each label:

1. Require `play_nullified_by_penalty == "N"`, `side_flag == False`, at least one tracked defender at arrival, landing within 5 yards of receiver arrival, and both throw time and arrival depth within that label's full-cohort interquartile ranges.
2. At shared 0.1-second times within the retained median, compute the candidate's mean Euclidean distance to the label median in lateral/depth coordinates. Select the smallest distance. Break ties by game ID, then play ID, then receiver ID.
3. If no candidate qualifies, drop the time/depth IQR condition first, then the defender condition. Record the relaxation. Penalty, side, and landing-distance restrictions remain mandatory. Pass result is never a selection filter.

**Field view.** Show the window from LOS − 12 yards to 8 yards beyond the farther of receiver arrival and ball landing, at full field width. The builder provides a 20-yard minimum window for unusually short plays. Yard numbers use distance to the relevant goal line, rather than raw x coordinates. Solid/dashed defender paths follow the same pre-/post-throw split as the target. QB and other-runner paths use official input observations only. The dotted ball flight is a straight-line approximation from the QB's observed throw position, not a tracked ball trajectory.

## Rebuild

Install the three dependencies into your Python environment, then run from this folder. Substitute your local official `train` directory if it differs.

```powershell
python .\build_route_visuals_v2.py `
  --data-dir "..\..\Step 2.0 Route Dataset\data" `
  --train-dir "C:\Local\Projects\Datalab Football Project\data\nfl-big-data-bowl-2026-analytics\114239_nfl_competition_files_published_analytics_final\train" `
  --supplementary "..\supplementary_data.csv" `
  --out "."
```

The train directory must contain `input_2023_w01.csv` through `input_2023_w18.csv` and the corresponding output files. Large official and defender files are streamed with selected columns. Re-running replaces generated figures, CSVs, checks, and the gallery in `--out`; the written reading guide and this README are retained.

**Interpretation reminder:** frame 1 is a snap proxy, the last input frame is a throw proxy, and arrival is the last tracked frame, not a verified catch event. Separation is measured against observed coverage defenders and does not establish a causal route-quality ranking.
