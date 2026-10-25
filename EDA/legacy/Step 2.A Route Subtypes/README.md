# Step 2.A — supplementary route subtypes

Run from any directory:

```powershell
python "C:\Users\willi\OneDrive\Documents\GitHub\football-analytics-project\EDA\Step 2.A Route Subtypes\build_route_subtypes.py"
```

Tested with **Python 3.12.13**; install the pinned analysis libraries with `python -m pip install -r requirements.txt` from this folder. Defaults resolve relative to the script. Optional `--input-dir`, `--supplementary`, and `--output-dir` accept alternate paths. The complete run uses seed **20261025**, all labeled targeted routes, 50 stability resamples and 1,000 separation resamples. It writes only to the output folder and verifies input SHA-256 hashes afterward. No Step 2.0 table is copied or modified. Execution can take several minutes.

Read **summary.md** first. **route_subtypes.csv** contains the final `route_subtype` for Step 3; `provisional_subtype` preserves exploratory clusters even when the decision is to keep the original label. Cluster IDs are within-label, ordered by median depth, starting at 1. ID 0 means a path shorter than 1 yd: no GMM probability, unchanged final label. `flags` is a semicolon-separated list. Low-likelihood outliers remain in fitting, summaries and comparisons.

**subtype_summary.csv** reports raw-data medians, IQRs, available counts, position/coverage shares, and label-level decisions. WR/TE/RB shares use all routes as denominator; `share_other_position` accounts for FB and other positions. Missing coverage is reported separately. **cluster_cards/** contains one PNG per provisional subtype and one overview per label. Throw markers use actual throw times; the thick median path is aligned on seconds since snap, with no duration stretching and at least 50% route support.

Supporting audit files:

- **bic_diagnostics.csv**: every fitted k, BIC, convergence and subtype-size checks.
- **clustering_diagnostics.json**: PCA dimensions, 50 ARIs, convergence, chosen k and size-rule reductions.
- **separation_comparisons.csv**: all subtype pairs, both measures, signed difference (B minus A), percentile intervals and the separation-rule result. A label must also pass stability to split.
- **route_descriptors.csv**: un-imputed geometry and timing for audit; no 24-point path copies.
- **outlier_review.csv**: exact ceil(1%) lowest fitted-label likelihoods, nearest provisional subtype, and supplementary play descriptions.
- **validation.json**: counts, assignments, input hashes, decisions and package versions.

Method: independently standardize the 24 arc-length shape coordinates and six geometry/indicator features within each label; PCA only the standardized shape block to 95% explained variance, then append standardized geometry. The six features are catch x/y, maximum x, break depth/angle, and a no-break indicator. Undetected break values stay missing in reports. Missing fitting values use the label median, or zero if the whole column is missing. Fit full-covariance GMMs for k=1–4 with five initializations, covariance regularization 0.0001, and up to 500 iterations. Labels with fewer than 200 eligible or total routes use k=1. Select the lowest BIC and reduce k one at a time until every cluster has at least 100 routes and 10% of the entire label (including short routes). K-means silhouette uses the selected feature space and k (up to 2,000 deterministically sampled rows).

Stability resamples the games represented in the label with replacement, preserving multiplicity of all routes in each drawn game, and refits imputation, scaling, PCA and GMM (three initializations). Compare full-data and refitted assignments on the unique overlapping routes; ARI requires no cluster-ID alignment. k=1 has trivially perfect stability and needs no repeated fits. Separation is loaded **only after all labels finish clustering**. Each label's separation bootstrap draws its games jointly for all subtypes, preserves game multiplicities, and calculates differences of route-level medians using available observations. For k>2, any pair can satisfy the user-specified rule; these exploratory intervals have no multiplicity adjustment. A split requires mean ARI >=0.6 and at least one median difference >=0.5 yd in absolute value whose 95% interval excludes zero.

Provisional depth names use median detected break depth only if at least half that subtype has a detected break, otherwise median catch x. Bands are <10 yd (`short`, or `shallow` for CROSS), 10–<20 (`intermediate`), and >=20 (`deep`); numeric suffixes distinguish clusters sharing a band. These are geometry names for human review, not validated football terminology. The analysis neither assesses label accuracy nor trains a route classifier.

Run focused numerical checks with `python -m unittest discover -s "EDA/Step 2.A Route Subtypes" -p "test_*.py"` from the repository root.
