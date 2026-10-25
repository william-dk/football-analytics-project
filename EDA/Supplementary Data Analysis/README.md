# Supplementary route EDA

Open `CodexOuput_route_visualizations/route_analysis.html` for the full visual report. It contains embedded images and six animations and can be shared as one offline HTML file.

Outputs include a 12-panel route image, 12 separate spaghetti plots, the four requested comparison groups, feature overlap, depth histograms, WR/TE/RB medians, starting-width and man/zone splits, final-input separation box plots, and a left/right orientation check. The individual plots are PNGs; animations are GIFs.

`descriptive_statistics.csv` has one row per class with median and quartiles. `route_features.csv` has one row per matched targeted receiver. `route_counts.csv` reconciles **2023-only** source and tracking counts. `season_scope_audit.csv` separates excluded seasons; `supplementary_play_match_audit.csv` checks all 2023 plays, including null route labels. `snap_proxy_validation.json` and the first-frame audit document the empirical timing origin. `median_time_support.csv` exports same-time median coordinates, sample counts, observation shares and plot inclusion for route, position, width, coverage and direction groups. `quality_flags.csv`, `routes_for_review.csv`, `orientation_audit.csv`, and the subtype/context CSVs support follow-up analysis. The source CSV is unchanged.

## Reproduce

Requires Python, numpy, pandas, scipy, matplotlib, and Pillow. From this directory:

```powershell
python build_route_analysis.py --tracking-root 'C:\Local\Projects\Datalab Football Project\data\nfl-big-data-bowl-2026-analytics\114239_nfl_competition_files_published_analytics_final\train'
```

The supplementary CSV contains play-level labels for **targeted receivers only**. Filter to `season == 2023` before counting or joining, then join on both game and play IDs and use only the matching Targeted Receiver track. Use official `input_2023_*.csv` files rather than the combined input/output extract. All 14,108 2023 plays match tracking; 14,105 have route labels and 3 do not. The 3,901 plays from 2024 W14–18 (3,900 labeled, 1 unlabeled) are outside this analysis, not matching failures. `--output` optionally selects a different output directory.

## Interpretation

- Normalize direction, translate to the first observed receiver position, and mirror lateral movement by the receiver's initial field side. Positive lateral movement is outward; positive longitudinal movement is downfield. Distances remain in yards.
- Sample up to 250 paths per class with seed 20261025. Use every matched route for statistics. Median paths align to the **same elapsed seconds** from the frame-1 snap proxy, on a 0.1-second grid: `t = (frame_id - 1) / 10`. No stretching by relative duration, endpoint padding, or extrapolation. At each time, only still-observed tracks contribute. Stop a median when support drops below `max(5, ceil(25% of the group's initial n))`; later medians describe a changing survivor population. The support chart and CSV make this attrition visible.
- A heuristic finds the strongest observed turn of at least 35 degrees, comparing smoothed movement vectors over 0.3 seconds before and after the candidate. This is not a charted break point. Pure stops, subtle bends, and turns near the end may be missed. Interpret HITCH timing and depth especially cautiously.
- Frame 1 is an empirically supported **snap proxy**, based on initial route-runner speeds and QB depth behind LOS; the pipeline recomputes and exports that evidence. `break_time_since_snap_proxy_s` measures detected turn time after this origin. `snap_to_throw_proxy_s = (last input frame - 1) / 10` uses the final input frame as a release proxy. Neither endpoint is event-verified; `verified_snap_to_throw_s` stays missing. `break_time_from_window_start_s` is a backward-compatible alias because all target tracks start at frame 1. No post-throw trajectories are used.
- Play-level `receiver_alignment` (e.g. 3x1) cannot establish receiver slot/outside alignment. The context plot uses a starting-width proxy: outside if at least 10 yards from field center, otherwise interior. Interior includes backfield players.
- Depth bands and density modes describe observed paths. They do not establish route subtypes: short observation windows can hide the later break. Review candidate label anomalies manually.
- Separation means distance to the nearest **supplied** defender, with defensive-back distance separately reported. It is not proof of coverage assignment or a causal route effect. Target selection, timing, coverage, and observation length confound class comparisons.

See `CodexOuput_route_visualizations/methodology.txt` and `one_page_summary.txt` for complete definitions and findings. Run `python verify_route_analysis.py` after generation for the time-alignment regression checks and output reconciliation.
