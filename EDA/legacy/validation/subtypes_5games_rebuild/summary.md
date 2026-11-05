# Step 2.A route subtypes v2

242 labeled targets; 5 games. Seed 20261025.

## Changes from v1

Corrected lateral coordinates throughout. Replaced every legacy break feature and the no_break indicator with total turn and retreat; fitted break depth is included only where at least half of a label has a break. Missing values are median-imputed only inside each fit. Shape retains 12 arc-length samples and PCA to 95% variance. GMM uses diagonal covariance, k=1..4, ten initializations (including all 50 game-bootstrap refits); WHEEL is k=1. The v1 minimum-size reduction and separation decision rules are retained. Short routes join the nearest median catch point of an existing subtype. Names add turn direction, with football names for manual review.

The sensitivity fit removes catch_x, catch_y, and max_depth_yd, repeats model selection and stability, and reports assignment ARI. Shape still encodes depth, so a surviving split is not proof of depth independence. A split reduced to k=1 is marked mainly pass depth.

## Per-label decisions and v1 comparison

- **ANGLE**: v1 k=1 (keep label) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.
- **CORNER**: v1 k=1 (keep label) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.
- **CROSS**: v1 k=1 (keep label) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.
- **FLAT**: v1 k=1 (keep label) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.
- **GO**: v1 k=1 (keep label) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.
- **HITCH**: v1 k=3 (split) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.
- **IN**: v1 k=2 (keep label) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.
- **OUT**: v1 k=1 (keep label) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.
- **POST**: v1 k=2 (keep label) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.
- **SCREEN**: v1 k=1 (keep label) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.
- **SLANT**: v1 k=1 (keep label) → v2 k=1 (keep label); stability ARI 1.000. insufficient sample (<200 eligible routes). Sensitivity k=1, agreement ARI=1.000, survives=True, mainly pass depth=False.

## Interpretation

Separation is loaded only after both geometry fits. Intervals use 1,000 paired game resamples, 95% percentile intervals, and the unchanged >=0.5 yd rule. They are exploratory, unadjusted, and not causal. Geometry clusters with stability below 0.6 remain provisional and do not change the retained route label. All missing break attributes remain missing in descriptors. Football suggestions in subtype_summary.csv need review. Outliers remain in the analysis.

Part 1 uses the reviewed 35-degree cut threshold, with 20–35 degrees labeled bend. Continuous fitted angles remain available but are not extra clustering inputs. Cut depth eligibility is evaluated at 35 degrees, not the superseded 20-degree definition.

Validation: passed; inputs unchanged=True; missing arrival separation=26.
