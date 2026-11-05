# Route subtypes v2

This version preserves the v1 pipeline's shape sampling, BIC/minimum-size
selection, game bootstrap, separation comparison and final split decision. It
uses the reviewed geometry in `../Step 2.0 Route Dataset/break_features_v2_35deg/`
and reads v1 comparisons from `../legacy/Step 2.A Route Subtypes/`. The archived
v1 code and results retain their original bytes.

Run `build_route_subtypes.py` with a Python environment containing the packages
in `requirements.txt`. The copied numerical and plotting functions live in
`subtype_helpers.py`. `test_route_subtypes.py` covers arc-length sampling,
training-only imputation, diagonal GMM, forced single-component WHEEL,
non-stretched medians and missing-data handling in the separation bootstrap.

## Outputs

- `route_subtypes.csv`: every labeled target once, provisional geometry cluster,
  probability, final retained/split subtype, short-route and outlier flags.
- `route_descriptors.csv`: corrected geometry, continuous total turn, continuous
  fit angle, stem heading, cut/bend/straight class and observed break-strength
  fields. Missing break attributes remain missing.
- `subtype_summary.csv`: counts, geometry, football-name suggestions, position,
  coverage, separation distributions, BIC, stability and the final decision.
- `bic_diagnostics.csv`, `clustering_diagnostics.json`: candidates and sizes,
  chosen k, PCA dimensionality, 50 bootstrap ARIs, convergence information and
  the nested sensitivity diagnostics.
- `sensitivity_comparison.csv`: main/sensitivity k, assignment ARI, sensitivity
  stability, whether the split survives, and whether it disappears to k=1.
- `route_subtypes_sensitivity.csv`: sensitivity assignments for fitted routes;
  integer cluster IDs have no correspondence to main IDs without matching.
- `separation_comparisons.csv`: pairwise throw/arrival median differences,
  1,000 game-bootstrap samples, 95% intervals and the unchanged 0.5-yard rule.
- `comparison_v1_v2.csv`: v1/v2 k and final decision, plus provisional-assignment
  ARI where both versions have multiple geometry clusters.
- `outlier_review.csv`: lowest-likelihood ceil(1%) per eligible label, with play
  descriptions. Outliers remain in all analyses.
- `cluster_cards/`: full corrected paths, throw markers and same-time medians.
- `summary.md`, `validation.json`: decisions, changes from v1 and integrity gates.
- `validation_5games/`: the first five Week 1 games, validated before the season.

## Method details

Clustering uses 12 equally spaced arc-length points (24 values), standardized
within each label and reduced to 95% PCA variance. Additional standardized
features are catch_x, corrected catch_y, max_depth_yd, total_turn_deg and
retreat_yd. break_depth_yd is included only if at least 50% of that label has
`has_break` under the **35° cut rule**. Neither the cut flag, class, stem heading,
continuous fitted angle nor any legacy break indicator is a clustering input.
The class and extra continuous descriptors are retained for interpretation.
Missing values are median-imputed only during fitting, including each bootstrap.

GMM uses diagonal covariance, k=1..4, ten initializations, seed 20261025,
max_iter=500 and v1's regularization/tolerance. WHEEL is fitted only at k=1.
As in v1, take the BIC winner, then reduce k until every subtype has at least
100 routes and 10% of the complete label. Report 50 game-bootstrap refits;
k=1 has trivially perfect ARI and requires no refitting. Final split decisions
require mean ARI >=0.6 and a pairwise separation difference of at least 0.5 yd
with a 95% interval excluding zero. Unstable geometry clusters remain
provisional while the final `route_subtype` retains the label.

The sensitivity analysis removes catch_x, catch_y and max_depth_yd and repeats
selection and the 50-refit stability analysis. Agreement uses adjusted Rand
index on fitted routes. A split survives when k is unchanged, main/sensitivity
assignment ARI >=0.6, and sensitivity stability >=0.6. A main split that reduces
to k=1 is marked mainly pass depth. Shape still contains depth information;
survival does not establish complete independence from pass depth.

Paths shorter than one yard are excluded from fitting. If a label has multiple
geometry clusters, they join the nearest median catch point and are flagged;
they never form their own subtype. Separation testing uses fitted routes as in
v1, while descriptive summaries include all routes. Names use break depth where
available for at least half of the subtype, otherwise catch depth, and then
turn direction; football suggestions are provisional.

## Rebuild into fresh destinations

```powershell
python -m unittest test_route_subtypes.py
python build_route_subtypes.py --five-games `
  --break-dir "../Step 2.0 Route Dataset/break_features_v2_35deg/validation_5games" `
  --output-dir validation_5games_NEW
python build_route_subtypes.py --output-dir full_season_NEW
```

The full-season command requires the delivered `validation_5games/validation.json`
to pass. Both commands require a passed 35° Part 1 validation and refuse to
replace existing subtype outputs. Supply `--input-dir`, `--break-dir` and
`--supplementary` to use other source locations. Separation is joined only after
both main and sensitivity clustering finish. Intervals are exploratory and are
not adjusted for multiple comparisons; separation associations are not causal.
