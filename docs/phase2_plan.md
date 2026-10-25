# Next-Phase Plan: Route-Conditioned Receiver Separation

Emory AI.DataLab Sports Analytics Project — William Kong, Jeffery Dou, Jimmy Nguyen

## Research question

After controlling for route type and coverage, **how much separation does a receiver create beyond what is expected, and which receivers do this consistently on which routes?**

Step 2 (route classification) supplies the conditioning variable for Step 3 (separation evaluation). The route model is therefore designed around what Step 3 needs.

Current status:

- **Step 1 (in progress):** Jeff is building play visualizations to develop an intuitive understanding of route types.
- **Step 2 (next):** classify route types, including a BERT-style embedding approach.
- **Step 3:** within each route type, quantify separation and make it player-specific. The core question is what good separation looks like for each route type.

## 1. Data facts that shape the plan (verified)

- **Raw separation is driven mainly by route type.** Median distance from the targeted receiver to the nearest DB at the last pre-throw frame: GO 1.7 yd, SLANT 3.2, OUT 4.2, HITCH 5.0, FLAT 9.8, ANGLE 11.8. Route mix also differs sharply by player: 49 of Davante Adams's 155 targets are HITCH routes, and 38 of Tyreek Hill's 133 are POST routes. Comparing players by raw average separation therefore mostly compares route mix, which is why separation must be route-specific.
- **Only targeted receivers have route labels.** 14,105 targeted routes are labeled across 12 classes (WHEEL has only 76). The other 50,644 route runners are unlabeled. A small labeled set plus a large unlabeled set is exactly the setting where BERT-style self-supervised pretraining helps.
- **Frame 1 is effectively the snap.** In the five-game sample, route runners' median speed at frame 1 is 0.01 yd/s, and 95.5% are below 1 yd/s (most of the rest are in motion). Route shapes can therefore be aligned from frame 1, matching the idea of comparing "what they do from beginning to end".
- **Trajectories are truncated at the throw.** The input window has a median of 2.5 s (IQR 2.1–3.1 s). Many untargeted deep routes have not yet broken when tracking stops.
- **Catch-point (arrival) separation is computable.** The post-throw output frames include the targeted receiver and on average 2.4 nearby defenders (range 1–6), with a median of about 0.9 s of ball flight. The current baseline uses input frames only and does not yet exploit this.
- **Coverage is nearly complete.** About 7 coverage defenders are tracked per frame; the missing players are mostly pass rushers, so nearest-defender distances are reliable.
- **The coordinate convention is mixed** (consistent with the data methods doc). On left-moving plays, x has already been flipped, but `absolute_yardline_number` is still in the original system. Once aligned to the same convention, the QB sits about 5 yd behind the line of scrimmage on both directions. Distances are unaffected, so existing separation values remain usable, but a unified offense-right coordinate layer is required before building route-shape features.
- **Justin Jefferson's sample is small.** In 2023 he played 10 games (Weeks 1–5 and 14–18): 277 routes and 88 targets, which leaves only 2–15 targets per route type. Partial pooling is required, and all routes should be used, not only targets. JJ McCarthy is not in the 2023 data.

## 2. Step 2: Route classification (about 3 weeks)

### 2.0 Route-level data layer (Week 1, prerequisite)

- One row per route runner per play; the sequence runs from frame 1 to the throw. Output frames for targeted receivers are stored separately and used only in Step 3.
- Normalization ("less about absolute position on the field, more about what they do from beginning to end"):
  - Orient the offense toward +x and place the origin at the receiver's position at the snap.
  - Mirror y by the receiver's side of the ball so that +y always points toward the sideline. OUT routes then always go toward +y and SLANT/IN toward −y, so routes from either side are directly comparable.
  - The ball itself is not tracked; the QB's y at the snap stands in for the ball's lateral position.
- Absolute position stays out of the shape features but is kept as context: alignment (outside, slot, inline, backfield), distance to the sideline, motion at the snap, and position (WR/TE/RB).
- Per-frame derived features are computed from x/y differences: velocity vector, acceleration, heading sin/cos, and turn rate. `dir` is not used because its convention is mixed.
- **Visualization deliverable (Jeff):** in normalized coordinates, one spaghetti plot per route label, overlaying a few hundred trajectories each. This is the most direct Step 1 deliverable for understanding route types, and it immediately exposes normalization bugs.

### 2.A Baseline: route similarity (Week 2)

The "play the game of route similarity" option comes first and is the baseline the BERT model must beat.

- **Handcrafted features + gradient boosting:** break depth, break time, signed break angle, lateral displacement, maximum speed, final heading.
- **Trajectory distance:** resample by time, compute DTW or Fréchet distances, and classify with kNN. The same distance matrix supports k-medoids or hierarchical clustering.
- **Evaluation:** game-grouped cross-validation (consistent with the methods doc), macro-F1, and a confusion matrix. Expected difficult pairs: SLANT vs IN, HITCH vs OUT, GO/POST/CORNER routes thrown before the break, and RB FLAT/ANGLE/WHEEL routes.

### 2.B BERT-style masked trajectory model (Weeks 2–3)

Interpretation of the professor's suggestion: randomly mask out frames ("random knockout"), let an encoder compress the sequence into a middle embedding, reconstruct the masked frames ("build back out"), and cluster on the latent embedding.

- **Architecture:** one token per frame (the normalized features above plus a time encoding) and a [CLS] token. A small Transformer encoder with 2–4 layers, d_model 64–128, and a padding mask for variable-length sequences.
- **Pretraining objectives, three kinds of masking:**
  - random masking of 15–30% of frames;
  - contiguous span masking, which forces the model to learn shape rather than interpolate;
  - tail masking, which teaches how routes continue and addresses truncation.

  Reconstruct positions and velocities with an MSE or Huber loss.
- **Pretraining data:** all 64,752 routes. The 50,000 unlabeled routes provide their value here.
- **Input is the receiver's own trajectory only, with no defender information.** Step 3 evaluates separation conditional on route type. If the route embedding contained defender positions, separation information would leak into the conditioning variable and the conclusions would become circular.
- **Evaluation (each compared against 2.A):**
  1. linear probe and kNN on targeted-route labels, reported as macro-F1;
  2. label efficiency: probe performance using 10%, 25%, and 100% of labels, which quantifies the value of unlabeled pretraining;
  3. UMAP visualization of the embeddings, HDBSCAN or GMM clustering, and NMI/ARI against the labels;
  4. a "cluster card" for each cluster built with Jeff's animator (medoid plus random samples), named by manual review.
- **Possible bonus:** clustering may split the 12 NFL classes into finer sub-types, such as quick out vs deep out/comeback, shallow cross vs deep cross, or a 5-yard hitch vs a 12-yard curl. The existing 12 labels serve as the first level and the finer clusters as the second.
- **Final output:** a fine-tuned classifier that assigns every route runner a route type with calibrated confidence (temperature scaling). Low-confidence routes are labeled UNCLEAR/TRUNCATED rather than forced into a class.
- **Go/no-go:** if BERT does not clearly beat 2.A on the probe, the simpler model does the labeling and the BERT embedding is used only for similarity and clustering. Either result is worth reporting.

## 3. Step 3: Route-conditioned separation (Weeks 4–6; define now)

### 3.1 Types of separation ("need to identify different types of separation")

| Dimension | Definition | Football meaning |
| :- | :- | :- |
| Timing | Release (about 1 s after the snap), break (change around the detected break point), throw (last input frame), arrival (last output frame, targeted only) | Release shows winning against press, break shows route running, and arrival is the window the QB actually faces |
| Direction | Project the receiver-to-defender vector onto the receiver's direction of travel and split it into trailing (defender behind), leverage (defender inside or outside), and cushion (defender in front) | GO/POST routes need trailing separation and OUT/SLANT routes need leverage. Three yards of cushion with the defender driving downhill on a HITCH against off-coverage is not the same as three yards of trailing separation |
| Dynamics | Closing speed (relative velocity projected onto the line between the players), combined into a time-to-close measure | Three yards with a defender closing at 6 yd/s is worth far less than three yards with a defender running in parallel |
| Reference defender | Nearest defender (current baseline); in man coverage, an inferred assigned defender (strongest co-movement over the route, e.g. via Hungarian matching) | The nearest defender switches during plays (the baseline counts switches); in man coverage, the assigned defender better shows whether the receiver won the matchup |
| Curve shape | Align separation(t) curves at the break point and cluster them; possible types include early win, break win, steady cushion, and closed down | A data-driven definition of separation types, which is then crossed with route type |

### 3.2 Defining good separation with two anchors

1. **Separation Over Expected (SOE).** Fit an expected-separation model per route with these inputs:
   - route type (from Step 2), man/zone, and coverage type;
   - time to throw, alignment, distance to the sideline, red zone, down and distance, motion, and number of visible defenders.

   Use a GBM or GAM with game-grouped cross-validation. Observed minus expected separation is the SOE, and "good" is defined as a percentile within the route type.
2. **Outcome anchor.** On targeted plays only, fit a logistic GAM for completion probability as a function of arrival separation for each route type, controlling for coverage and air distance. The separation at which each curve flattens is the "sufficient" separation for that route type.
   - This is a descriptive curve. QBs throw to receivers they judge to be open (target selection), so it describes the relationship conditional on being targeted.
   - On the professor's point about a good passer: QB accuracy shifts the curve, because good QBs throw into tight windows. Include a QB random effect or stratify by QB.

### 3.3 Player-specific analysis (Justin Jefferson case study)

- **Hierarchical model:** `SOE ~ fixed effects + (1|receiver) + (1|receiver:route_type) [+ (1|defender)]`, fit with statsmodels MixedLM or bambi/PyMC. Jefferson has only 2 SLANT targets; partial pooling shrinks that cell toward the league mean with a wide interval instead of producing a noise-driven ranking.
- **Sample:** all routes labeled in Step 2 (277 for Jefferson rather than 88 targets), weighted by route-type confidence.
- **Separation profile:** SOE with intervals for each route type, route mix, the mix of separation types, and speed retention through breaks. Comparison players are CeeDee Lamb, Tyreek Hill, Amon-Ra St. Brown, Puka Nacua, and Davante Adams, who each have 339–470 routes.
- **Required stability check:** compute the metric separately on odd and even weeks and report the split-half correlation. If a player's SOE is not stable between halves of the season, it reflects noise rather than skill. Report the stability of raw separation alongside it for comparison.
- **DB perspective:** add a defender random effect to the same model to see which cornerbacks suppress separation on which route types. The assigned defender in man coverage gives the cleanest version.
- **On the QB:** Kirk Cousins was lost for the season in Week 8, so after Jefferson returned in Week 14 his targets came from backup QBs, and his catch outcomes depend heavily on the QB. Separation itself depends far less on the QB, which is a reason to center the study on separation rather than catch results.

## 4. Timeline and suggested roles (adjustable)

| Week | Work | Deliverables |
| :- | :- | :- |
| 1 (Oct 26 – Nov 1) | 2.0 data layer and coordinate validation | Route-level table; 12 spaghetti plots |
| 2 (Nov 2 – Nov 8) | 2.A baseline; first BERT pretraining run | Baseline F1 and confusion matrix |
| 3 (Nov 9 – Nov 15) | BERT evaluation, clustering, cluster cards; check-in with the professor | Probe and clustering comparison; go/no-go decision |
| 4 (Nov 16 – Nov 22) | Label all routes; compute the 3.1 separation types (including arrival) | Decomposed frame-level separation table |
| 5 (Nov 23 – Nov 29) | 3.2 SOE and completion curves | "Good separation" definition per route type |
| 6–7 (Nov 30 – Dec 13) | 3.3 player and DB analysis, stability checks, report | Jefferson case study and final report |

Suggested roles: Jeff continues visualization (spaghetti plots, then cluster cards, then separation timeline animations); one person owns the data layer and the BERT model; another owns the 2.A baseline and the 3.1 separation definitions.

## 5. Actions to start this week

1. Build the normalized route-level table on top of the existing `EDA/separation_baseline/outputs/separation_baseline/frames/*.csv.gz` files and `nearest_defenders()` in `EDA/separation_baseline/football_network/separation.py`.
2. Check more left- and right-moving plays and document the coordinate convention of every field in the data documentation.
3. Jeff adds a normalized route view to the PlayAnimator and produces spaghetti plots grouped by label.
4. Run the 2.A handcrafted-feature baseline first; within a week it shows how separable the 12 labels are.

## 6. Questions for the professor

- Does our interpretation of the BERT approach (masked frame reconstruction, then clustering on the latent embedding) match what he had in mind? In the notes, we read "2027" as "the NFL route definitions already in the data" and "From DB perspective" as "the defensive back's point of view".
- Should good separation be anchored to completion/EPA, or only to percentiles within each route type?
- Should we bring in the Big Data Bowl 2025 data (2022 season, Weeks 1–9)? As far as we know, its `player_play.csv` labels `routeRan` for **every** route runner and includes PFF coverage matchups (who covered whom). That would remove two bottlenecks: labels existing only for targeted receivers, and no way to validate the assigned defender. These field names are from memory and need to be checked after download.
