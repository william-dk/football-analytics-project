# Supplementary route EDA

Open `route_visualizations/route_analysis.html` for the full visual report. It contains embedded images and six animations and can be shared as one offline HTML file.

Outputs include a 12-panel route image, 12 separate spaghetti plots, the four requested comparison groups, feature overlap, depth histograms, WR/TE/RB medians, starting-width and man/zone splits, final-input separation box plots, and a left/right orientation check. The individual plots are PNGs; animations are GIFs.

`descriptive_statistics.csv` has one row per class with median and quartiles. `route_features.csv` has one row per matched targeted receiver. `route_counts.csv` reconciles source and tracking counts; unmatched labels are not assigned artificial paths. `quality_flags.csv`, `routes_for_review.csv`, `orientation_audit.csv`, and the subtype/context CSVs support follow-up analysis. The source CSV is unchanged.

## Reproduce

Requires Python, numpy, pandas, scipy, matplotlib, and Pillow. From this directory:

```powershell
python build_route_analysis.py --tracking-root 'C:\Local\Projects\Datalab Football Project\data\nfl-big-data-bowl-2026-analytics\114239_nfl_competition_files_published_analytics_final\train'
```

The supplementary CSV contains play-level labels for **targeted receivers only**. Join on both game and play IDs and use only the matching Targeted Receiver track. Use the official input files rather than the combined input/output extract. Source counts are calculated from the supplied CSV and can differ from counts quoted in previous analyses.

## Interpretation

- Normalize direction, translate to the first observed receiver position, and mirror lateral movement by the receiver's initial field side. Positive lateral movement is outward; positive longitudinal movement is downfield. Distances remain in yards.
- Sample up to 250 paths per class with seed 20261025. Use every matched route for median trajectories and statistics. Median paths use relative observed duration, not a shared clock or verified snap time.
- A heuristic finds the strongest observed turn of at least 35 degrees, comparing smoothed movement vectors over 0.3 seconds before and after the candidate. This is not a charted break point. Pure stops, subtle bends, and turns near the end may be missed. Interpret HITCH timing and depth especially cautiously.
- Snap-relative break time and verified time to throw are unavailable: the data lack event timestamps. Observed span and final-input separation are explicitly labeled proxies. No post-throw trajectories are used.
- Play-level `receiver_alignment` (e.g. 3x1) cannot establish receiver slot/outside alignment. The context plot uses a starting-width proxy: outside if at least 10 yards from field center, otherwise interior. Interior includes backfield players.
- Depth bands and density modes describe observed paths. They do not establish route subtypes: short observation windows can hide the later break. Review candidate label anomalies manually.
- Separation means distance to the nearest **supplied** defender, with defensive-back distance separately reported. It is not proof of coverage assignment or a causal route effect. Target selection, timing, coverage, and observation length confound class comparisons.

See `route_visualizations/methodology.txt` and `one_page_summary.txt` for complete definitions and findings.
