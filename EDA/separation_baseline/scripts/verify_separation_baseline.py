"""Small geometry contract checks independent of NFL data distribution."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_network.separation import nearest_defenders


def main():
    rows = []
    for frame in [1, 2]:
        for pid, position, side, role, x, y in [
            (1, "WR", "Offense", "Targeted Receiver", 0, 0),
            (2, "CB", "Defense", "Defensive Coverage", 3, 4),
            (3, "FS", "Defense", "Defensive Coverage", -3, -4),
            (4, "LB", "Defense", "Defensive Coverage", 1, 0),
        ]:
            if frame == 2 and position in ["CB", "FS"]:
                continue
            rows.append(dict(game_id=1,play_id=1,nfl_id=pid,frame_id=frame,
                player_name=str(pid),player_position=position,player_side=side,
                player_role=role,x=x,y=y))
    source = pd.DataFrame(rows)
    result = nearest_defenders(source)
    assert len(result) == 2
    assert result.iloc[0].separation_db_yards == 5
    assert result.iloc[0].nearest_db_id == 2  # Exact tie uses smaller ID.
    assert result.iloc[0].separation_coverage_yards == 1
    assert result.iloc[0].nearest_coverage_id == 4
    assert np.isnan(result.iloc[1].separation_db_yards)
    assert result.iloc[1].n_db_observed == 0
    try:
        nearest_defenders(pd.concat([source,source.iloc[:1]],ignore_index=True))
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate keys were accepted")
    print("PASS: 3-4-5 geometry, LB vs DB, deterministic ties, missing DB, duplicate rejection")


if __name__ == "__main__":
    main()
