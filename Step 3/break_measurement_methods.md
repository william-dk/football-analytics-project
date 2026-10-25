# Measuring route breaks: total turn angle and two-segment fit

## Why the current break detector fails

The Step 2.0 detector compares movement direction over 0.3 s before and after each frame and requires at least 35 degrees, with at least 0.6 yd of movement in each window. It found breaks in only 583 of 14,105 targeted routes, even on full paths (input + output).

In the five-game sample, the reasons are:

- Real breaks are arcs, not corners. An OUT turns about 70 degrees in total, spread over 0.5–1 s; a 0.3 s window sees only about 18 degrees.
- Receivers slow to 1.3–2 yd/s at the break, so 0.3 s covers less than the required 0.6 yd and the frame is skipped. HITCH is hit hardest.
- Smoothing rounds the turn further, and one 35 degree threshold is applied to every route type.

Lengthening the window to 0.8 s raises detection to 63–87% for OUT, IN and HITCH while GO stays at 5%. The two methods below avoid a single short-window threshold altogether.

## Shared preparation

1. Use the targeted receiver's full path (input + output) in route coordinates: origin at the snap position, x downfield, y positive toward his own sideline.
2. Smooth positions with the `[1,4,6,4,1]/16` kernel.
3. Derive speed from smoothed position differences. Output frames have no official speed, so post-throw speed is always position-derived.

## Method 1: total turn angle

**What it does.** Compares where the receiver was heading early in the route with where he is heading at the end.

1. Stem direction `a0`: direction from the first to the last point between 0.3 and 1.0 s after the snap. The first 0.3 s is skipped because heading is unstable at low speed.
2. Final direction `a1`: direction from the first to the last point of the final 0.5 s of the path.
3. Total turn = `a1 − a0`, wrapped to −180..+180 degrees. Positive means toward the receiver's sideline, negative toward the middle.

**Output.** One signed angle per route.

**Strengths**

- Simple and stable; no threshold and no break location needed.
- Captures gradual, arcing turns in full.
- Always defined for any path longer than about 1.5 s.
- Clearly separates route types in the five-game sample (medians): GO +6, CROSS −26, SLANT −55, POST −59, OUT +65, CORNER +73, IN −81 degrees.

**Weaknesses**

- Says nothing about where the break happens, how deep it is, or how sharp it is.
- Sensitive to the choice of stem and end windows. Very short paths (quick screens, early throws) have overlapping windows.
- Routes with two turns (for example a double move) collapse into one net angle.
- For HITCH, the final direction after a stop is noisy at low speed.

**When to use**

- Step 2.A subtype clustering, as a geometric feature replacing the 0/1 break indicator.
- Describing and validating route types: checking that OUT is positive and IN, SLANT and POST are negative, and spotting likely mislabels.
- Any analysis that needs a turn measure for every route without missing values.

## Method 2: two-segment line fit (change point)

**What it does.** Models the route as one straight segment, a break point, and a second straight segment, and finds the break point that fits best.

1. For each candidate break frame k, keeping at least 4 frames on each side:
   - fit a line to the points before k and another to the points after k;
   - use orthogonal (total) least squares via SVD, so the fit works for sideways routes;
   - record the sum of squared perpendicular distances of both fits.
2. The break point is the k with the smallest total error.
3. Read off:
   - break time and break depth (t and x at k) and lateral position (y at k);
   - break angle: signed angle between the two line directions;
   - break strength around k: mean speed in the 0.5 s before, minimum speed near k, mean speed in the 0.5 s after, speed retention = after / before, and time needed to complete most of the turn (sharpness).

**Output.** Break location, depth, angle and break-strength measures per route.

**Strengths**

- Locates the break, which makes depth and timing available (five-game medians: SLANT 3.6 yd, OUT 7.3, IN 9.7, POST 13.7, CORNER 13.8).
- Gives a fixed point for measuring break strength: deceleration, speed retention and sharpness. These describe how well a receiver breaks, not just which route he runs.
- Uses the whole path, so it does not depend on a short window or a fixed angle threshold.

**Weaknesses**

- Always returns a break, even for a nearly straight GO. A rule is needed to decide whether a real break exists, for example |fit angle| at least 20–25 degrees.
- Poor fit for routes that stop and come back (HITCH, curl, comeback): the fit angle is about 20 degrees while the total turn is about 52 degrees. Use maximum depth and the distance retreated from it instead.
- Assumes one break; double moves and wheel routes need a three-segment fit or should be flagged.
- Slower than Method 1 (one fit per candidate frame), though fine for 14,105 routes.
- Break strength after the throw relies on position-derived speed, which is noisier than the official input speed.

**When to use**

- Step 2.A, for break depth as a geometric feature, only where a real break exists.
- Step 3, for break strength: testing whether receivers who keep more speed through the break gain more separation after it, within the same route type.
- Replacing the `break_*` columns in the Step 2.0 route table.

## Angle versus strength

The break angle mainly reflects the route type: an OUT turns about 65 degrees whoever runs it. It is useful for separating routes and subtypes, not for rating receivers. Break strength is how sharply and cleanly the turn is made, measured around the fitted break point:

| Measure | How | Meaning |
| :- | :- | :- |
| Speed retention | mean speed 0.5 s after the break / mean speed 0.5 s before | How much speed survives the turn |
| Deceleration | minimum speed near the break, or the drop from before to minimum | How hard the receiver brakes |
| Sharpness | time to complete 80% of the turn | Short = a sharp cut; long = a rounded route |

Two 65 degree OUTs can differ completely: one completed in 0.3 s with a 10% speed loss, another rounded over 0.9 s with a 40% loss. That difference is what Step 3 uses to explain why a receiver creates separation.

## Summary

| | Total turn angle | Two-segment fit |
| :- | :- | :- |
| Main output | One signed angle | Break location, depth, angle, strength |
| Threshold needed | No | Yes, to decide whether a break exists |
| Gradual turns | Fully captured | Captured; location is the best single corner |
| Stop-and-return routes | Usable | Poor; use max depth and retreat |
| Missing values | Rare | Only where no real break exists |
| Best for | Route-type features, 2.A clustering, label checks | Break depth, Step 3 break strength |

Use both: total turn for every route, and the two-segment fit for location and strength where a real break exists.

A prototype of both methods, run on the five-game sample, produced the numbers above.
