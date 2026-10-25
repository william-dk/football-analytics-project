# Exploratory Data Analysis

## Receiver separation baseline

[Open the separation analysis](separation_baseline/README.md): professor-feedback steps 1–4, including data checks, nearest-DB distances, six trajectory/distance examples, receiver summaries with sample sizes, and reproducible code.

- [Findings and methodology](separation_baseline/outputs/separation_baseline/README.md)
- [Figures](separation_baseline/outputs/separation_baseline/cases/)
- [All generated outputs](separation_baseline/outputs/separation_baseline/)

## Step 2.0 route layer

The current full-season dataset is in `Step 2.0 Route Dataset/data/`: six ordinary CSV tables covering 2023 Weeks 1–18, including targeted receiver paths, all-runner route context, observed defenders, frame-level separation and targeted-route summaries. Route labels are assigned only to targeted receivers.

- [Dataset fields, joins and reading examples](Step%202.0%20Route%20Dataset/data/README.md)
- [Builder, methods and storage layout](Step%202.0%20Route%20Dataset/README.md)

The earlier five-game route-layer sample remains in `separation_baseline/outputs/route_layer_sample/`, built by `separation_baseline/scripts/build_route_layer.py`. It is separate from the current full-season dataset.

## Existing analyses

- [First-game EDA notebook](phase1_first_game_eda.ipynb)
- [NFL player data notebook](NFLPlayerData.ipynb)
- [Five-game tracking sample](tracking_5_games.csv)
