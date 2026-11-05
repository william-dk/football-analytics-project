# Reviewed break features: 35-degree cuts

The current feature table is
`break_features_v2_35deg/full_season/target_break_features.csv`, with its validation
beside it. The five-game precursor is in `break_features_v2_35deg/validation_5games/`.
Both use the user's reviewed amendment. The original 20° feature table, failed
validation, first five-game run and review records are now preserved under
`../legacy/Step 2.0 Route Dataset/` with their original relative layout.

Hard median total-turn signs apply only to OUT/CORNER (positive) and
IN/SLANT/POST (negative). For FLAT, SCREEN, ANGLE and WHEEL, validation reports
the share with `catch_y_corrected > 0`; FLAT and WHEEL require at least 80%.
A label absent from the five-game subset is recorded as unobserved.

The feature table adds `stem_heading_deg` for the smoothed 0.3–1.0-second
direction, with 0° straight downfield and positive toward the corrected sideline.
It remains available when total turn is missing because the final window
overlaps the stem. `fit_angle_deg` is continuous, independent of the cut rule.
`break_class` is cut at |angle| >=35°, bend at 20°–<35°, and straight otherwise.
`has_break` is true only for cuts; break location, speed and sharpness fields
remain missing for non-cuts. An undefined fit has no cut and a missing fit angle.

Validation reports cut shares at 30°, 35° and 40°. GO is a diagnostic comparison
against OUT, IN and SLANT, with percentage-point differences; it cannot fail the
build. Side correction, smoothing, TLS fitting, earliest ties, speed windows,
sharpness, stop-return features and source hashing otherwise retain the accepted
implementation described in `route_geometry.py` and the archived earlier review.

The full table includes every targeted receiver, including three unlabeled
targets. Subtype and visualization cohorts exclude only those missing labels.
Existing source tables and prior feature outputs are never edited. Run
`python -m unittest test_route_geometry.py`, then `build_break_features.py`
with `--five-games --output-dir FRESH_SUBSET` before the full-season command
with `--output-dir FRESH_FULL`. Downstream tools explicitly require passed
35° validation records, so they cannot silently use the historical feature table.

`verify_break_features.py` independently checks current classification, missing
values, side correction, stem/total turn, endpoints, stop-return fields and cut
speed windows from source paths. It no longer needs the old 20° table. Use
`--feature-dir` for a five-game or full-season output and `--report` for a fresh
validation record. The former `verify_break_features_v2.py` is preserved in legacy
as the one-time 20°–35° migration audit. `../archive_paths.py` lets current
verifiers resolve historical paths without rewriting old validation records.
