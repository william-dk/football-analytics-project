# Break features: stopped at the requested review gate

The new geometry module and builder leave all existing Step 2.0 tables unchanged.
Five unit tests pass. The Week 1 first-five-game build passes its validation with
242 targeted routes. The full-season feature table has 14,108 targets, including
14,105 labeled targets, each exactly once, with no infinite values. All three
source-table SHA-256 hashes are unchanged. No-break attributes remain missing.

The full-season validation is **failed**, with exactly one failure: the corrected
FLAT median total turn is **−1.3875°**, violating the required positive sign.
OUT, CORNER, IN, SLANT and POST pass their required signs. This gate has not been
weakened. Parts 2 and 3 have not been executed successfully and no downstream
results or refreshed visuals have been produced.

The five-game total-turn medians reproduce the prototype closely: OUT +65.45°,
IN −81.26°, SLANT −55.10°, POST −59.35°, CORNER +73.25°, GO +6.39°.
Detected-break depth medians are SLANT 3.565 yd, OUT 7.290 yd, IN 9.555 yd,
POST 12.500 yd, CORNER 13.810 yd. POST differs from the prototype's 13.7 yd:
the reported median includes only routes passing the specified 20° fit rule.

The requested two-segment TLS rule flags **52.68% of GO routes**, compared with
the expected <15%. At 15° and 25°, the shares are 65.71% and 39.16%, respectively
(see the validation record for exact shares). The threshold remains 20°.
The GO expectation is reported as a diagnostic rather than an extra hard gate;
the explicit FLAT sign requirement is what stops this run.

The side correction flips 313 labeled routes. FLAT arrivals more than five yards
inside fall from 13.565% to 4.690%; OUT falls from 5.646% to 4.878%.
993 targeted routes remain side-uncertain under the prescribed release rule.

`audit_break_gate.py` independently reconstructs FLAT and GO side correction and
total turn from source paths, and fits TLS using covariance eigenvectors rather
than production SVD. Maximum discrepancies are below 6e−14 degrees.
`break_features_review.json` contains the reconstruction results. In FLAT,
median total turn is +8.69° for unflagged sides and −9.24° for flagged sides;
RBs have a −9.84° median. Net turn measures change from the early stem rather
than whether the endpoint lies toward the corrected sideline. These facts help
review the gate but do not justify changing its requirement automatically.

## New files and reproducibility

- `route_geometry.py`: pure side correction, smoothing, total turn, TLS split,
  stop-return and break-strength functions.
- `test_route_geometry.py`: release correction, uncertainty, edge padding,
  exact straight-line tie handling, vertical TLS, mirrored turn signs, overlap,
  stationary paths and retreat checks.
- `build_break_features.py`: fresh-output-only builder, with validation and
  source preservation hashes.
- `validation_5games/`: successful five-game feature table and validation.
- `data/target_break_features.csv`: full-season feature table; its companion
  `data/break_features_validation.json` is failed and must be checked before use.
- `audit_break_gate.py` and `break_features_review.json`: independent review.

Run from this folder with a Python environment containing NumPy and pandas:

```powershell
python -m unittest test_route_geometry.py
python build_break_features.py --five-games --output-dir validation_5games_NEW
python build_break_features.py --output-dir full_season_NEW
```

Existing output files are never overwritten; select a fresh folder for a rerun.
The independent audit likewise refuses to replace its existing review JSON.
Speed uses smoothed displacement times 10 on both phases. Before/after speed
windows exclude the split instant and include their 0.5-second outer bounds.
Sharpness uses the last heading at or below 0% of the fitted signed turn before
its first 80% crossing, searching after the stem window (t >= 1 s); if no 0%
heading is observed there, it starts at the first eligible heading. It remains
missing when 80% is not reached. Break location and stop-return coordinates use
observed positions, while headings and residuals use smoothed positions.

Review the FLAT sign requirement and GO detection behavior before changing any
definition or resuming the downstream subtype and visualization work.
