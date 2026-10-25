# Step 2.A — within-label route subtypes

2023 targeted routes: **14,105**, across 272 games. Seed 20261025. Full input + output geometry only was used for fitting; separation was joined afterward.

## Step 3 decisions

- **HITCH: split** into HITCH_intermediate, HITCH_intermediate_2, HITCH_short (ARI 0.789).
  - HITCH_intermediate minus HITCH_short, sep_at_throw: **-0.90 yd** (95% CI -1.31 to -0.57).
  - HITCH_intermediate minus HITCH_short, sep_at_arrival: **-0.74 yd** (95% CI -0.96 to -0.44).
  - HITCH_intermediate_2 minus HITCH_short, sep_at_throw: **-0.76 yd** (95% CI -1.07 to -0.51).
  - HITCH_intermediate_2 minus HITCH_short, sep_at_arrival: **-0.52 yd** (95% CI -0.75 to -0.31).

## Labels retained

- **ANGLE**: k=1, ARI=1.000; BIC split failed minimum subtype size; reduced to k=1.
- **CORNER**: k=1, ARI=1.000; BIC split failed minimum subtype size; reduced to k=1.
- **CROSS**: k=1, ARI=1.000; BIC split failed minimum subtype size; reduced to k=1.
- **FLAT**: k=1, ARI=1.000; BIC split failed minimum subtype size; reduced to k=1.
- **GO**: k=1, ARI=1.000; BIC split failed minimum subtype size; reduced to k=1.
- **IN**: k=2, ARI=0.804; no subtype pair meets both separation difference and interval thresholds.
- **OUT**: k=1, ARI=1.000; BIC split failed minimum subtype size; reduced to k=1.
- **POST**: k=2, ARI=0.234; unstable: mean game-bootstrap ARI <0.6.
- **SCREEN**: k=1, ARI=1.000; BIC split failed minimum subtype size; reduced to k=1.
- **SLANT**: k=1, ARI=1.000; BIC split failed minimum subtype size; reduced to k=1.
- **WHEEL**: k=1, ARI=1.000; insufficient sample (<200 eligible or total routes).

## Caveats and review

- **Break-detection dependence:** HITCH_intermediate: 304/304 detected breaks. The accepted split partly reflects the required break-detection feature, so review the geometry cards before giving these groups football names.
- Only 583/14,105 routes have detected breaks. Names use median catch depth unless at least half a subtype has a detected break. Names are provisional geometry bands (<10, 10–<20, >=20 yd), not verified football categories.
- Shape uses 12 arc-length points, standardized within label and reduced to 95% PCA variance. Geometry is also standardized; all-missing break columns use zero for fitting only. k is BIC-selected, then reduced until every subtype has >=100 routes and >=10% of its whole label.
- ARI uses 50 game-resampled refits (including preprocessing), matching each overlapping route once. ARI for k=1 is trivially 1. Silhouette is a k-means cross-check, not the GMM assignment's silhouette.
- Retaining a label after a minimum-size failure does not establish uniform geometry. Rare detected-break indicators can produce small mixture components that force k back to 1.
- Separation intervals use 1,000 paired game resamples per label and all subtype pairs; intervals are exploratory and unadjusted for multiple comparisons. Separation associations may reflect depth, position, coverage or other context; they are not causal effects.
- 1,140 labeled routes lack arrival separation (1,142 across all targets, including two unlabeled routes); each summary reports available counts. 13 paths <1 yd retain their original label (cluster_id=0, no fitted probability).
- **146 outliers** (ceil of 1% per fitted label) remain in all analyses and appear in outlier_review.csv for manual review. Low likelihood alone does not establish a mislabel.
- Cards show up to 200 full paths and throw markers. The median uses actual 0.1-second snap-time bins without stretching or endpoint extrapolation, ending when fewer than half the routes remain.
- Input SHA-256 checks: unchanged. All 14,105 routes are assigned exactly once. Detailed BIC, stability and pairwise comparisons accompany this report.
