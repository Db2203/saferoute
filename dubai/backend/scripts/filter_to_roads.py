"""Apply the road-proximity filter to collisions.parquet (run AFTER preprocessing,
BEFORE aggregates / model / edge-blackspots). Run from dubai/backend:
    python -m scripts.filter_to_roads
"""
import pandas as pd

from app.data.aggregates import PARQUET
from app.data.road_filter import MAX_ROAD_DIST_M, build_filter_graph, filter_to_roads


def main() -> None:
    df = pd.read_parquet(PARQUET)
    graph = build_filter_graph()
    out = filter_to_roads(df, graph)
    out.to_parquet(PARQUET, index=False)
    dropped = len(df) - len(out)
    print(
        f"road-proximity filter (<= {MAX_ROAD_DIST_M:.0f} m): "
        f"{len(df):,} -> {len(out):,} ({dropped:,} dropped, {dropped / len(df) * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
