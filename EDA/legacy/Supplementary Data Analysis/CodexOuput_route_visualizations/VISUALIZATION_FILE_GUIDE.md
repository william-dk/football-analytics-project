# Route Visualization Output Guide

## Overview

This directory contains an exploratory analysis of **14,105 matched targeted-receiver routes** from the 2023 season. It covers 12 route classes: ANGLE, CORNER, CROSS, FLAT, GO, HITCH, IN, OUT, POST, SCREEN, SLANT, and WHEEL.

The original output contains **48 files**: 22 PNG visualizations, 6 GIF animations, 15 CSV tables, 2 JSON metadata files, 2 text documents, and 1 HTML report. This guide is an additional documentation file.

Coordinates are normalized so that the targeted receiver begins at `(0, 0)`, positive vertical displacement is downfield, and positive lateral displacement is toward the receiver's initial sideline. Frame 1 is used as an empirical snap proxy. Thick route lines are coordinate-wise medians aligned by actual elapsed time; observations are neither stretched nor extended.

## Static route-trajectory visualizations

### `all_12_routes.png`

A 12-panel overview of every route class. Each panel shows up to 250 sampled trajectories as thin lines, the same-time median as a thick line, and the total number of matched routes. It is the best single image for comparing overall route geometry and variability.

### Individual route plots

Each file below provides an enlarged class-specific view, with up to 250 sampled paths and a median calculated from all routes still observed at each elapsed-time point.

- **`route_angle.png`** — ANGLE routes; emphasizes shallow depth, substantial variability, and the curved movement toward the middle after release.
- **`route_corner.png`** — CORNER routes; shows a vertical stem followed by movement toward the initial sideline.
- **`route_cross.png`** — CROSS routes; shows strong movement across the field, normally producing negative normalized lateral displacement.
- **`route_flat.png`** — FLAT routes; shows rapid lateral movement toward the sideline with limited downfield depth.
- **`route_go.png`** — GO routes; shows primarily vertical movement, with observed depth varying because tracking generally ends near the throw.
- **`route_hitch.png`** — HITCH routes; shows the vertical stem and variation in stopping or return behavior.
- **`route_in.png`** — IN routes; shows a vertical stem followed by movement toward the field interior.
- **`route_out.png`** — OUT routes; shows a vertical stem followed by movement toward the initial sideline.
- **`route_post.png`** — POST routes; shows a vertical stem followed by an inward diagonal component.
- **`route_screen.png`** — SCREEN routes; shows very shallow paths that may include backward downfield displacement.
- **`route_slant.png`** — SLANT routes; shows a shallow inside break, often within a short observation window.
- **`route_wheel.png`** — WHEEL routes; shows the curved transition from lateral movement to a deeper vertical path. All 76 matched routes are displayed.

## Direct route comparisons

### `confusable_routes.png`

A four-panel trajectory comparison covering SLANT versus IN, HITCH versus OUT, POST versus CORNER versus GO, and FLAT versus ANGLE versus WHEEL. Thin lines are sampled routes and thick colored lines are class medians. It shows where route shapes overlap and where depth, direction, or curvature may help distinguish classes.

### `feature_overlap.png`

A four-panel scatterplot of maximum observed depth against final lateral displacement for the same potentially confusable route groups. It tests how much separation is visible from two simple features. It is descriptive, not a held-out classification evaluation.

## Distribution and time-support visualizations

### `depth_subtypes.png`

A 12-panel histogram of maximum observed depth. It shows typical depth, within-class variation, long tails, and possible shallow/deep groupings. The associated peak analysis did not find convincing multiple prominent modes, so the bands are not validated clusters.

### `median_time_support.png`

Shows how many routes remain observed at every tenth of a second after the snap proxy. The dashed line marks the median-path support threshold: the greater of five routes or 25% of the original group. It explains why median paths have different lengths and why later median points represent a changing survivor population.

## Contextual median-trajectory visualizations

### `context_position.png`

Compares median paths by player position, including wide receivers, tight ends, running backs, and fullbacks where available. Only route-position groups with at least five targets are shown.

### `context_alignment_proxy.png`

Compares routes by initial width: outside means at least 10 yards from field center, while interior means less than 10 yards. This is not a verified slot/outside designation, and the interior group may include backfield players.

### `context_coverage.png`

Compares median paths under charted man and zone coverage. Only groups with at least five targets are included. The differences are descriptive and do not establish that coverage caused the observed route geometry.

## Separation visualization

### `separation_boxplots.png`

Contains boxplots of final-input-frame distance to the nearest supplied defender and to the nearest supplied defensive back. Boxes represent the interquartile range and the internal line is the median; outlier points are hidden. These comparisons are affected by depth, timing, coverage, target selection, and defender availability.

## Orientation and normalization validation

### `orientation_check.png`

Compares normalized median routes from plays originally moving left and right. Similar shapes support the direction-normalization procedure; differences identify classes worth reviewing but do not by themselves prove an error.

### `orientation_audit.csv`

The numerical support for the orientation plot. For every route and original direction, it records sample size, median forward speed, median final lateral displacement, and median depth.

## Animated GO-route examples

All six GIFs animate the targeted receiver and supplied defenders. They were selected from low- and high-final-separation GO routes and illustrate extremes rather than typical plays.

- **`animation_low_2023092100_2016.gif`** — Isaiah Hodgins, game 2023092100, play 2016; approximately 0.36 yards from the nearest defender at the final frame.
- **`animation_low_2023100900_228.gif`** — Jayden Reed, game 2023100900, play 228; approximately 0.39 yards from the nearest supplied defender.
- **`animation_low_2023120301_2146.gif`** — Amon-Ra St. Brown, game 2023120301, play 2146; approximately 0.52 yards from the nearest defender.
- **`animation_high_2023102208_1903.gif`** — Jerry Jeudy, game 2023102208, play 1903; approximately 13.08 yards from the nearest defender.
- **`animation_high_2023111204_3537.gif`** — A.T. Perry, game 2023111204, play 3537; approximately 11.08 yards from the nearest defender.
- **`animation_high_2023121001_872.gif`** — Isaiah Likely, game 2023121001, play 872; approximately 15.20 yards from the nearest defender.

### `animation_examples.csv`

Contains metadata for the six animations: player, position, game, play, frames, observed duration, depth, lateral displacement, break features, coverage, formation, separation values, and low/high example designation.

## Consolidated report

### `route_analysis.html`

A self-contained report combining the 12-class overview, descriptive statistics, time support, route comparisons, feature overlap, depth variation, contextual analyses, separation results, normalization checks, animations, and methodology. This is the most complete presentation file in the directory.

## Core analysis tables

### `route_features.csv`

The main route-level analytical dataset. Each row is one matched targeted-receiver route and includes player identity, game/play identifiers, frame range, duration, speed, depth, lateral displacement, initial width, break features, quality flags, context, penalty status, and separation measurements. Most outputs are derived from this table.

### `descriptive_statistics.csv`

Route-class summaries containing counts and median, first-quartile, and third-quartile values for break depth, break time, break angle, maximum speed, duration, throw-time proxy, maximum depth, and final defender separation. Verified snap-to-throw values are blank because event-verified timing was unavailable.

### `route_counts.csv`

Lists supplementary, matched-tracking, and unmatched counts for every route class. All labeled 2023 route classes have zero unmatched records.

### `context_statistics.csv`

Numerical summaries for route, player position, initial-width proxy, and coverage combinations. It reports sample size, median maximum depth, and median final defender separation.

### `median_time_support.csv`

Detailed time-indexed data behind all median paths. It records median coordinates, observed sample count, original group size, observed share, support threshold, and whether each point was included in the plot.

## Depth-subtype supporting tables

### `depth_subtype_bands.csv`

Counts and shares of each route class within descriptive maximum-depth bands. These bins are not validated route subtypes.

### `depth_density_peaks.csv`

Reports prominent peaks from each route class's exploratory maximum-depth density estimate and the number of detected peaks.

## Data-quality and review files

### `quality_flags.csv`

Class-level counts of short windows, detected turns, late turns, returns toward the quarterback, duplicate frames, frame gaps, out-of-bounds coordinates, and implausibly large steps.

### `routes_for_review.csv`

Individual routes satisfying one or more review criteria. It supports manual inspection of unusual geometry or quality flags. Inclusion is not proof of a bad route label.

### `excluded_target_audit.csv`

Records targeted plays excluded because of matching or target-identification problems. It currently contains only the header, indicating no such exclusions in the 2023 cohort.

### `supplementary_play_match_audit.csv`

Play-by-play matching results between supplementary labels and official tracking data, including season, week, game, play, route label, and tracking availability.

### `season_scope_audit.csv`

Documents included and excluded season-week combinations, with play and labeled-route counts. It shows that 2024 weeks 14-18 were intentionally excluded from the 2023 analysis.

## Snap-proxy validation

### `snap_proxy_first_frame_audit.csv`

First-frame observations used to validate frame 1 as a snap proxy. It includes player roles, positions, coordinates, speed, line-of-scrimmage location, and quarterback distance behind the line.

### `snap_proxy_validation.json`

A compact summary of the validation: 14,108 matched 2023 plays, 14,105 labeled routes, three unlabeled plays, median first-frame route-runner speed of 0.01 yards per second, 94.7% below 1 yard per second, and median quarterback position approximately 4.95 yards behind the line of scrimmage.

## Metadata and documentation

### `analysis_manifest.json`

Machine-readable metadata containing the season, counts, data-source paths, random seed, sampling settings, animation count, median-alignment method, support rule, and snap-proxy summary.

### `methodology.txt`

Detailed methodology covering cohort selection, matching, coordinate normalization, alignment proxy, time alignment, median support, turn detection, timing proxies, separation calculations, depth analysis, and limitations.

### `one_page_summary.txt`

An executive summary of coverage, separability, depth patterns, context, quality, separation, timing, and the principal interpretation cautions.

## Recommended reading order

1. Open `route_analysis.html` for the complete report.
2. Review `all_12_routes.png` for the route-shape overview.
3. Use `confusable_routes.png` and `feature_overlap.png` to examine class overlap.
4. Check `depth_subtypes.png` and `median_time_support.png` for distribution and timing context.
5. Review the context plots and `separation_boxplots.png`.
6. Read `one_page_summary.txt`, followed by `methodology.txt` before modeling.
7. Use `route_features.csv` and the audit tables for reproduction or deeper analysis.

## Main interpretation cautions

- The cohort contains targeted receivers only.
- Tracking generally ends near the throw, so an observed path may not contain the complete intended route.
- Maximum observed depth is not necessarily full route depth.
- Frame 1 is a supported snap proxy, but verified event timestamps are unavailable.
- The final input frame is a release proxy, not a verified pass-release timestamp.
- Median paths use real elapsed time and are not duration-normalized.
- Later median points include only routes still observed.
- Initial width is not a verified slot/outside classification.
- Depth bands are descriptive rather than validated clusters.
- Separation is confounded by timing, depth, coverage, target selection, and the supplied defender set.
- Animated examples are selected extremes rather than representative plays.
