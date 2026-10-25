# How to Read Route Visualizations v2

## What these figures answer

These figures describe how targeted receivers moved before and after the throw in the 2023 tracking sample. They let you compare route geometry, see where receivers and the ball finish, and inspect one geometrically typical example of each route label in its defensive context.

The sample contains **14,105 labeled targets across 12 route labels**, from Weeks 1–18. These are receiver-play records, not unique receivers. Only targeted receivers enter the route distributions. This is a selected tracking sample, so its frequencies and completion rates should not be treated as estimates for every NFL route or every 2023 pass attempt.

Start with [the gallery](index.html), then follow this sequence:

1. Open the **all-routes overview** to compare route shapes and depth on a common scale.
2. Open a label's **spaghetti plot** to see the variation hidden by its median.
3. Open that label's **representative play** to connect the abstract route shape to the real field, defenders, and pass context.
4. Use the CSVs and `checks.json` when you need exact values, sample keys, or selection details.

The most important distinction throughout is between **receiver arrival**, **ball landing**, and **pass result**. Arrival is the last receiver tracking observation. The gold star is the supplied ball landing location. Complete, incomplete, or intercepted comes from the supplementary play result. A receiver reaching the end of a path does not prove that a catch occurred there.

## 1. Read the overview first

Open [all_routes_overview.png](spaghetti/all_routes_overview.png). Each panel represents one route label. Every panel uses the same x and y limits and an equal yard scale, so a yard has the same geometric meaning in both directions.

- **Horizontal axis:** lateral movement from the receiver's initial position. Zero means the original lateral position; positive means toward the receiver's own initial sideline; negative means toward the interior and potentially across the field.
- **Vertical axis:** depth relative to the line of scrimmage, in yards. The horizontal line at zero is the LOS. Positive depth is downfield, and negative depth is behind the LOS.
- **Dark path:** the coordinate-wise median receiver position at the same time since the snap proxy, using all routes in the label.
- **Green dots:** receiver arrival points on completed passes.
- **Red crosses:** receiver arrival points on incompletions or interceptions.

The overview shows **all arrival points**, rather than only the sampled spaghetti routes. Dense regions indicate common arrival locations within this sample. Overlap and unequal label sizes make the darkness of different panels unsuitable as a standardized probability comparison.

Use the atlas to ask: Which labels reach deeper? Which finish farther inside or outside? Which have broad endpoint clouds? Do two labels overlap substantially? The source labels describe football route categories, while the geometry also reflects alignment, timing, coverage, and play design.

The dark path's endpoint is where median support stops. It is **not** the median arrival point and is not a catch marker.

## 2. Read an individual spaghetti plot

Open `spaghetti/route_<LABEL>.png`. It overlays up to **200 routes**, sampled with seed `20261025` while preserving the label's completed/non-completed share as closely as the sample size allows. WHEEL has fewer than 200 routes, so every WHEEL route is drawn. The statistics panel and median use **all routes in the label**.

### Follow one path

The thin solid blue segment shows receiver movement before the throw proxy. The orange dot marks the last input frame, used as the throw proxy. The dashed blue segment continues through the post-throw output tracking. Its endpoint is a green filled circle for a completed pass, or a red cross for an incomplete or intercepted pass.

The gold star shows ball landing in exactly the same transformed coordinates. A thin dotted gold connector joins receiver arrival to ball landing. A short connector indicates spatial proximity at these two supplied endpoints. It does not establish possession, catch timing, catch difficulty, or the actual airborne path of the ball.

Landing stars more than **40 yards** from receiver arrival are omitted, along with their connectors. The receiver's path and result marker remain. The caption gives the number omitted from the plotted sample, and `checks.json` also supplies the corresponding full-label count. A distant landing may reflect a throwaway, an unusual play, or a questionable value; this distance rule does not diagnose which explanation applies.

### Understand the two origins

The lateral origin is the receiver's initial y position. The depth origin is the LOS. As a result, receivers start at lateral zero but can start above or below depth zero. A backfield receiver may begin several yards behind the LOS.

Positive lateral movement always means toward that receiver's own initial sideline after normalization. Thus a negative CROSS route displacement does not mean that every receiver ran to the physical left side of the stadium. Both initial sides of the field have been brought into one comparable route frame.

### Read the sideline reference

The dotted vertical line is the **median distance from the receiver's start to the sideline on the receiver's own side**. The light band spans the 10th–90th percentiles of that distance across the entire label.

This is a summary across different plays and alignments. It is not a single boundary shared by all displayed routes. A path crossing the median line does not by itself prove that the receiver went out of bounds; that play may have a different sideline position. Consult its real-field coordinates before making that judgment.

### Read the median as a same-clock summary

At 2.0 seconds, for example, the median summarizes receiver positions at 2.0 seconds since the snap proxy. A short route is not stretched to match a long route. No path is extrapolated beyond its observed start or finish.

The median stops when fewer than **the larger of five routes or 25% of that label's routes** remain observed. Support is evaluated on a shared 0.1-second grid. This reduces reliance on a small set of long-developing plays, although the mix of observed routes can still change over time. Use `median_path_support.csv` to inspect that support directly.

A coordinate-wise median can combine positions from different players. It need not be a route that any single receiver actually ran, and a median curve can soften individual breaks.

### Read the statistics panel

- **n routes:** every labeled targeted route in this category, including routes not drawn.
- **Completion rate:** the share with source result `C`. `I` means incomplete, and `IN` means intercepted. This is descriptive of this cohort.
- **Median detected break depth:** depth of the geometric break proxy among routes with a detected break and a non-missing break depth. Its `n` is that subset's size. A missing detection does not prove the absence of a football route break. The detector can use the full path, including post-throw motion.
- **Median arrival depth:** receiver x at the final tracked frame minus LOS x. This describes where tracking ends, rather than pass length or yards gained.
- **Median throw time:** elapsed seconds from frame 1 to the last input frame. Both timestamps are proxies, not verified event tags.
- **Median separation at throw / arrival:** nearest observed coverage-defender distance at those moments, in yards. Each `n` is the number of non-missing measurements for that statistic.

The medians are calculated separately. There is no requirement that one play exhibit all of them. In particular, subtracting the two separation medians is not the same as taking the median of each play's separation change.

Individual spaghetti plots can have different axis limits so their details remain legible. Use the common-scale overview when comparing physical dimensions across labels.

## 3. Read a representative play on the real field

Open `representative/route_<LABEL>_<game_id>_<play_id>.png`. This view uses actual flipped field coordinates, with the offense moving **right** for every play.

### Establish the field context

Field y spans 0–53.3 yards. White yard lines are five yards apart, with stronger lines every ten yards. Yard numbers use football distance-to-goal numbering: the midfield 50 is at field x = 60. Raw field x includes the end zones, so field x is not the printed yard number.

The solid dark-blue vertical line is the LOS. The solid yellow line is the first-down marker at LOS plus yards to go, when it is visible in the window. Darker green marks any visible end zone. The horizontal window follows the play, so different examples may have different windows and screen scales; their axes still measure yards with equal aspect.

### Follow the players and ball

- **Thick blue path:** targeted receiver. Solid before the throw, dashed after. The blue start dot, orange throw dot, and result-colored arrival marker identify its three reference positions.
- **Thin red paths:** observed coverage defenders, with a cross at the start. Their post-throw segments are dashed. A brighter, thicker path highlights the defender nearest the target at the throw.
- **Black path:** QB, using input tracking only.
- **Grey paths:** other route runners, using input tracking only.
- **Gold star:** supplied ball landing point.
- **Dotted gold flight line:** straight line from the QB's observed throw position to ball landing. It is explicitly an approximation; there is no observed airborne ball trajectory in this figure.
- **Dashed grey separation segments:** receiver-to-nearest-defender distances at throw and arrival. The annotations report the Table F values and are checked against the drawn coordinates.

The highlighted throw defender and the nearest arrival defender may be different people. A change in the nearest player is not evidence of a charted assignment switch. Only observed coverage defenders are available; missing defenders and untracked pass rushers cannot be reconstructed. Short or absent post-throw defender paths reflect observation availability and should not be interpreted as a defender stopping or leaving the play.

The title reports the label, receiver, and actual pass result. The context lines identify game/play, down and distance, coverage, throw time, ball flight duration, pass length, selection distance, and any relaxed conditions. The play description is shortened to 120 characters; it supplies context rather than a second tracking timeline.

### Understand why this example was selected

The builder seeks a typical **path shape**, rather than the best play or the most successful outcome. A candidate must be non-nullified by penalty, have an unflagged side, have at least one observed defender at arrival, and have ball landing within five yards of receiver arrival. Its throw time and arrival depth must also lie in the label's middle 50%.

For each eligible candidate, the builder averages its Euclidean distance to the label median at their shared observed times. The smallest mean distance wins. Ties go to the earliest game ID and play ID. If no candidate qualifies, the time/depth interquartile restrictions are relaxed first, then the defender restriction. The relaxation is recorded, and the remaining requirements remain in force.

**Pass result does not enter selection.** The chosen example can be an incompletion. Nevertheless, the proximity, side, defender, and timing filters make the eligible subset narrower than the full label. One typical geometry cannot represent every alignment, defensive response, or result in that label. The selection distance measures geometric resemblance in yards, not model confidence or football execution quality.

## 4. Orient yourself to the 12 labels

These brief descriptions are reading aids, not rules used to relabel the data. The figures retain the supplementary labels, including plays whose paths look unusual.

- **HITCH — 2,660 routes:** look for a downfield stem followed by a stop or return toward the passer. Variation in stop depth can spread the endpoint cloud.
- **OUT — 2,214:** look for a stem followed by movement toward the initial sideline, usually positive laterally.
- **FLAT — 1,983:** look for relatively shallow movement toward the sideline. Starting behind the LOS is possible.
- **CROSS — 1,496:** look for sustained movement toward the interior and across the field, usually negative laterally.
- **GO — 1,397:** look for a predominantly downfield path. Lateral adjustments and ball tracking can still produce curvature.
- **IN — 1,107:** look for a downfield stem followed by an inward cut, usually toward negative lateral movement.
- **SLANT — 1,049:** look for earlier diagonal movement inward rather than a long vertical stem.
- **POST — 769:** look for a downfield stem followed by an inward diagonal continuation.
- **ANGLE — 544:** look for an initial release followed by a change inward; starting alignment can strongly affect the shape.
- **CORNER — 509:** look for a downfield stem followed by an outward diagonal continuation, usually positive laterally.
- **SCREEN — 301:** look for a shallow or behind-LOS receiving area. Post-throw movement can materially change the final endpoint.
- **WHEEL — 76:** look for an outward release that turns upfield. With only 76 routes, the observed variety is based on a much smaller cohort.

Different labels can overlap. Do not infer a classification error solely because a single trajectory resembles another label.

## 5. Make a defensible comparison

For a comparison such as OUT versus IN, start with the overview to compare the direction and depth at the same geometric scale. Then open both spaghetti plots to examine variation and the proportion of pre- versus post-throw movement. Check the all-route arrival-depth and timing statistics before attributing an endpoint difference to the route label itself. Finally, use the two real-field examples to see how alignment and observed defenders can produce the patterns.

A useful written observation is: “In this targeted-route sample, label A has a deeper median arrival location than label B; the distributions also differ in timing and alignment.” Support the values with `label_summary.csv`.

Avoid turning that into “A creates more separation” or “B is more effective” from these figures alone. Receiver ability, coverage, quarterback decisions, target selection, pass depth, formation, game context, and tracking availability all influence the observed outcomes. These plots provide descriptions and hypotheses for subsequent analysis.

## 6. Check the supporting files

Use **`label_summary.csv`** for exact full-label statistics and denominators. Use **`representative_plays.csv`** to locate the selected play in other analyses and inspect its eligibility fields. Use **`sampled_routes.csv`** to reproduce which routes appear in the spaghetti plots. Use **`median_path_support.csv`** to see which times have enough observed paths.

Use **`checks.json`** to verify that the build has status `complete`, includes 14,105 labeled targets, passes the left/right ball-coordinate checks, creates 25 PNGs, and records any missing passer or landing exclusions. The completed build also compares the selected receiver output paths with the official weekly output files and verifies the drawn separation markers.

### Results of this build

The direction checks passed: the median landing-to-arrival distance on completions is **1.00 yard for left-moving plays** (4,817 completions) and **0.98 yard for right-moving plays** (4,919 completions). All 12 representative plays meet the original eligibility conditions, so no conditions were relaxed. All 12 have a passer observed at the throw, and both separation markers were verified for each example. The selected examples happen to be completions; completion was not a selection requirement.

One SCREEN landing point is more than 40 yards from receiver arrival and is excluded from the drawing. No other label has a landing exclusion under this rule.

As a worked comparison, GO has a median receiver arrival depth of **22.43 yards**, versus **10.83 yards** for IN. Their median throw times are **2.6 seconds** and **2.8 seconds**, respectively. This illustrates why arrival depth, throw timing, and route geometry should be read separately. The figures do not establish that a deeper route takes longer before the throw or produces a better outcome.

For every figure, keep this statement with any exported slide: **Arrival is the last tracked frame, not a verified catch event.**
