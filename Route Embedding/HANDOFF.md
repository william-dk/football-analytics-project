# Handoff: route embedding and receiver identification

Updated 2026-11-14 at the end of the second session. Results, tables and figures are in
[README.md](README.md), which is current. This file covers state, decisions and what is open.

## Start here

- **Goal:** the professor's "Final Question" (2026-11-12): "Can we identify a receiver by how they
  run a certain route?" with a self-supervised BERT-style transformer, an embedding of about 16
  dimensions that can rebuild the route, and knocked-out windows. The old plan's steps 2.A, 2.B
  and Step 3 are outdated; plan from the final question directly.
- **Branch:** `claude/route-embedding-pipeline`, cut from `main` (`b2e1a6e`). The work is
  committed there. **It is not pushed and has no PR.** Commit and push only when the user asks.
- **Language:** reply to the user in Chinese, keeping technical terms in English.

## State

- **Everything in the README has been run.** No training or analysis is in progress.
- **Code** (all in `Route Embedding/`): `run_pipeline.py` runs every stage. See the file table in
  the README. 38 unit tests pass.
- **In the commit:** the pipeline and analysis code, the rewritten `README.md`, six
  `figures/v2_*.png` and this file. New files: `compare_runs.py`, `player_model.py`,
  `player_style.py`, `run_pipeline.py`.
- **Git-ignored outputs, only in this checkout:**
  - `data/`: 78,521 sequences (14,048 `full` targeted routes + 64,473 `pre` pre-throw paths of
    all runners). `data/v1_targeted/` is the first session's data; its rows equal the `full` rows.
  - `runs/v2/`: the final model with `eval_report.json`, `embeddings.csv`, `identification/`,
    `style/targeted`, `style/all`, `style/targeted_alignment`, `style/all_alignment`.
  - `runs/v1/`: the first version, re-analysed with the new code.
  - `runs/abl_*`: eight design ablations plus `abl_always_knockout` (v2 without whole routes).
  - `runs/comparison.csv`: all runs in one table (`python compare_runs.py`).

## Reproduce

```powershell
python run_pipeline.py --name v2                       # build -> test -> train -> evaluate -> identify -> style
python player_style.py --run-dir runs/v2 --population all --condition type+alignment   # alignment control
python compare_runs.py                                 # table over all runs
```

- Training v2 takes 23 minutes on 8 threads (`--threads 8`); the full analysis about 15 minutes.
- Ablations: `python train.py --name abl_base --kinds full --epochs 60 --whole-prob 0` plus one
  changed option each (`--z-dim 8`, `--win-min 5 --win-max 10`, `--inputs x y a`,
  `--win-min 0 --win-max 0`, ...). One thread per process, several processes side by side.

## Findings (details and numbers in the README)

1. A receiver can be identified from how he runs his routes, but only from many routes: among
   251 receivers, 2.0% from one route, 22.7% from 50, 33.3% from a half-season (chance 0.4%).
2. Identification is limited by the routes, not by the embedding: within a route type 8% of the
   embedding variance is a stable receiver difference, and all model variants and baselines
   identify about equally from one route.
3. The receiver part is 45% position, 12% team, 43% individual.
4. z rebuilds a route to 0.08 yd and fills a 15-frame window at the break to 0.17 yd. A linear
   PCA code with 16 numbers fills windows as well and copies complete routes better (0.03 yd).
5. Receiver metrics: distinctiveness is a stable trait; consistency mostly reflects usage.

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
- **Receiver-balanced accuracy, half-season cross-validation** (profiles from odd weeks, tested
  on even weeks, and the reverse), as in the first version.

## Environment

- Windows 11, Python 3.14.6, torch 2.15.0+cpu, 10 cores, no GPU.
- Smart App Control once blocked `import torch` (`WinError 4551`); don't change security
  settings, tell the user if it comes back.
- No `gh` CLI: open PRs in the Claude desktop built-in browser. The repo has no CI.
- Large inputs (`EDA/Step 2.0 Route Dataset/data/`) are git-ignored and exist only through
  OneDrive.

## Open (the user has not chosen)

1. Push the branch and open a PR.
2. Next analyses, in the README's "Next steps": profiles across seasons and team changes; role
   against habit (condition on alignment and coverage together); relate the receiver metrics to
   separation and catches; sharper reconstruction.
3. The team's plan Google Doc ("Route-Conditioned Receiver Separation") has not been updated
   with these results.
