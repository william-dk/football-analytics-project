# Receiver separation baseline

Professor-feedback steps 1–4, using the available 2023-season tracking inputs.

## Start here

- [Findings and definitions](outputs/separation_baseline/README.md)
- [Visual report](outputs/separation_baseline/report.html) — download/open locally; GitHub shows HTML source.
- [Six trajectory and distance examples](outputs/separation_baseline/cases/)
- [WR summaries: all observed routes](outputs/separation_baseline/players_wr_all_routes.csv)
- [WR summaries: targets only](outputs/separation_baseline/players_wr_targets.csv)
- [Receiver-play metrics](outputs/separation_baseline/receiver_play_metrics.csv)
- [Weekly compressed frame tables](outputs/separation_baseline/frames/)
- [Output validation](outputs/separation_baseline/output_validation.json)

## Scope

18 weeks, 272 games, 14,108 plays, 1,821,157 receiver-frame records. Eligible summaries include 38,002 WR routes and 8,611 WR targets. All generated analysis outputs are included; the original weekly input datasets are external inputs and are not duplicated here.

The last supplied input frame is a timing proxy, not event-verified release. There is no verified snap or arrival in these inputs. Nearest DB uses CB/DB/S/FS/SS among supplied coverage players; the secondary comparison includes all supplied coverage positions. Neither guarantees all 11 defenders. Raw distance means do not establish route-running ability. Source route labels apply only to the targeted receiver.

## Reproduce

From the repository root:

```bash
cd EDA/separation_baseline
python -m pip install -r requirements.txt
python scripts/verify_separation_baseline.py
python scripts/verify_separation_outputs.py
```

The second verification reconciles all bundled frame tables and player summaries and refreshes the report and SVG figures. Recompute everything from your local original dataset with:

```bash
python scripts/run_separation_baseline.py --source "/path/to/114239_nfl_competition_files_published_analytics_final"
```

The source directory must contain `supplementary_data.csv` and `train/input_2023_w01.csv` through `input_2023_w18.csv`. The default output remains `outputs/separation_baseline` relative to this package directory.

PNG copies are bundled. To regenerate them, Node.js and the optional `sharp` package are needed:

```bash
npm install --no-save sharp
node scripts/render_separation_cases.cjs
```

## Integrity and layout

`migration_manifest.json` lists the SHA-256 and size of every copied source/result file. `football_network/separation.py` contains the reusable nearest-distance function. `scripts/` contains the analysis, geometry verification, output reconciliation, and optional PNG rendering code. `outputs/separation_baseline/` preserves the complete generated result hierarchy and relative report links.

The pre-existing EDA notebooks and tracking sample are separate analyses and are unchanged by this addition.
