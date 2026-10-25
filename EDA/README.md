# Exploratory Data Analysis

## Receiver separation baseline

[Open the separation analysis](separation_baseline/README.md): professor-feedback steps 1–4, including data checks, nearest-DB distances, six trajectory/distance examples, receiver summaries with sample sizes, and reproducible code.

- [Findings and methodology](separation_baseline/outputs/separation_baseline/README.md)
- [Figures](separation_baseline/outputs/separation_baseline/cases/)
- [All generated outputs](separation_baseline/outputs/separation_baseline/)

## Step 2.0 route layer

`separation_baseline/football_network/routes.py` turns tracking into normalized route frames (origin at the snap position, +x downfield, +y toward the receiver's sideline) and a one-row-per-route table joined to 2023 supplementary context. Route labels are kept only for targeted receivers. Build the five-game sample from the repository root with `python EDA/separation_baseline/scripts/build_route_layer.py`; pass `--tracking` for the full file. Outputs and validation checks are in `separation_baseline/outputs/route_layer_sample/`.

## Existing analyses

- [First-game EDA notebook](phase1_first_game_eda.ipynb)
- [NFL player data notebook](NFLPlayerData.ipynb)
- [Five-game tracking sample](tracking_5_games.csv)
