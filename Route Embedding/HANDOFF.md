# Handoff: route embedding and receiver identification

Updated 2026-11-16 at the end of the third session. Results, tables and figures are in
[README.md](README.md), which is current. This file covers state, decisions and what is open.

## Start here

- **Goal:** the professor's "Final Question" (2026-11-12): "Can we identify a receiver by how they
  run a certain route?" with a self-supervised BERT-style transformer, an embedding of about 16
  dimensions that can rebuild the route, and knocked-out windows. The old plan's steps 2.A, 2.B
  and Step 3 are outdated; plan from the final question directly.
- **Branch:** `claude/supervised-ceiling`, cut from `main` at `c98f463` (PR #6, the pipeline, is
  merged). The third session's work is committed on this branch. **It is not pushed and has no
  PR.** Commit and push only when the user asks.
- **Language:** reply to the user in Chinese, keeping technical terms in English.

## State

- **Everything in the README has been run.** No training or analysis is in progress.
- **Code** (all in `Route Embedding/`): `run_pipeline.py` runs every stage, now including
  `supervised`. 43 unit tests pass.
- **Third session** (the "supervised ceiling", direction 2 of the brainstorm):
  - new `supervised_ceiling.py`: classifiers trained on receiver labels, compared with the
    random-effects model on the same task;
  - `model.py`: `hidden_states()` split out of `encode()` (no change in behaviour);
  - tests, `run_pipeline.py` (`supervised` stage), `compare_runs.py` (`sup_*` columns);
  - README: new section 4, revised "Answer in short" and design section,
    `figures/v2_supervised_ceiling.png`.
- **Git-ignored outputs, only in this checkout:**
  - `data/`: 78,521 sequences (14,048 `full` targeted routes + 64,473 `pre` pre-throw paths of
    all runners). `data/v1_targeted/` is the first session's data; its rows equal the `full` rows.
  - `runs/v2/`: the final model with `eval_report.json`, `embeddings.csv`, `identification/`,
    `style/` (targeted, all, both with `_alignment`) and `supervised/`:
    `all/` (every classifier and the shuffled control), `targeted/`, `all_alignment/`,
    `all_frac25/`, `all_frac50/`, `logs/`.
  - `runs/v1/`, `runs/abl_*`: the first version and the design ablations, each with
    `supervised/all/` (the classifier on z).
  - `runs/comparison.csv`: all runs in one table (`python compare_runs.py`).

## Reproduce

```powershell
python run_pipeline.py --name v2                       # build -> test -> train -> evaluate -> identify -> style -> supervised
python player_style.py --run-dir runs/v2 --population all --condition type+alignment   # alignment control
python supervised_ceiling.py --run-dir runs/v2 --population all --shuffle-control --models mlp_z mlp_all mlp_positions mlp_pca16 mlp_speed tf_scratch tf_finetune
python compare_runs.py                                 # table over all runs
```

- Training v2 takes 23 minutes on 8 threads (`--threads 8`); the style analysis about 15 minutes.
- Classifiers: the MLPs take about 5 minutes per training half on one thread, the Transformers
  about 30 minutes on two. Run one process per model and half
  (`--models tf_finetune --folds odd --no-eval --threads 2`), then one run without `--no-eval` to
  compare; scores are cached in `supervised/<population>/<model>_<half>.npz`.
- Ablations: `python train.py --name abl_base --kinds full --epochs 60 --whole-prob 0` plus one
  changed option each (`--z-dim 8`, `--win-min 5 --win-max 10`, `--inputs x y a`,
  `--win-min 0 --win-max 0`, ...). One thread per process, several processes side by side.

## Findings (details and numbers in the README)

1. A receiver can be identified from how he runs his routes, but only from many routes. Among 251
   receivers (chance 0.4%) a classifier on z names him 2.6% of the time from one route, 35.3% from
   50 and 43.6% from a half-season; the random-effects model 2.0%, 22.7% and 33.3%.
2. The embedding is not the limit: a Transformer trained on receiver labels is no better than the
   classifier on z, and the same classifier on hand-made descriptions is worse.
3. The readout and the data per receiver are limits. The classifier beats the random-effects
   model only with about 100 training routes per receiver; on the targeted routes (about 31) it
   is worse. Both readouts still improve steeply with more training data.
4. The knockout training matters for identification, which shows only with the classifier: no
   knockout 26.9% from 50 routes against 35.4% at the same budget.
5. Within a route type 8% of z's variance is a stable receiver difference: 45% position, 12% team,
   43% individual. Alignment explains about half of the classifier's lead.
6. z rebuilds a route to 0.08 yd and fills a 15-frame window at the break to 0.17 yd. A linear PCA
   code with 16 numbers fills windows as well and copies complete routes better (0.03 yd).
7. Receiver metrics: distinctiveness is a stable trait; consistency mostly reflects usage.

## Decisions and why

- **All route runners, not only targets.** Untargeted routes exist only up to the throw, so every
  runner gets a `pre` sequence. This gives 4-5 times more routes per receiver, which the
  identification needs. Their route type is read from z (kNN trained on the labeled targets).
- **Always report the linear baseline.** PCA with 16 components showed that the first version's
  transformer rebuilt routes 5-10 times worse than a linear code. Every reconstruction number is
  now shown next to PCA and the straight line.
- **Architecture.** A [CLS]-only readout and a Transformer decoder could not reach the precision
  that 16 numbers allow. A linear readout of all frame states, a linear + MLP decoder onto fixed
  eigen-routes, velocity inputs and a higher learning rate reach the first version's error with
  about a tenth of the computation.
- **20% of training routes are shown whole.** A model that always saw a knocked-out window
  embedded complete routes badly, because it had never seen one.
- **Identification model.** Random effects for receiver, game and play within a route type
  (`player_model.py`). Receiver variance comes from products of averages over different games, so
  game effects cannot pass as receiver differences; eigenvalues are measured on receivers that
  did not choose the axes. On simulated data the model recovers the variances and is calibrated.
- **Supervised comparison kept fair.** Logit adjustment by p(receiver | route type), so a
  classifier cannot win by knowing which routes a receiver runs (tested on simulated data, and
  the shuffled-label control stays at chance); tuning only on held-out games of the training
  half; every method scored on the same candidates, test routes and random route sets.
- **Receiver-balanced accuracy, half-season cross-validation** (profiles from odd weeks, tested
  on even weeks, and the reverse), as in the first version.

## Environment

- Windows 11, Python 3.14.6, torch 2.15.0+cpu, 10 cores, no GPU.
- Smart App Control once blocked `import torch` (`WinError 4551`); don't change security
  settings, tell the user if it comes back.
- No `gh` CLI: open PRs in the Claude desktop built-in browser (the user may have to sign in to
  GitHub there first). The repo has no CI.
- Large inputs (`EDA/Step 2.0 Route Dataset/data/`) are git-ignored and exist only through
  OneDrive.

## Open (the user has not chosen)

1. Push `claude/supervised-ceiling` and open a PR.
2. Other directions from the brainstorm, also in the README's "Next steps":
   - explain the 86% play-to-play variation with coverage, alignment and game situation, and
     look for receiver × coverage interactions;
   - route disguise: predict a route's rest from its first second with the knockout model;
   - more data per receiver: another season, profiles across seasons and team changes;
   - relate the receiver metrics to separation and catches.
3. The team's plan Google Doc ("Route-Conditioned Receiver Separation") has not been updated
   with these results.
