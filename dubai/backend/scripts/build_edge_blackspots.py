"""Snap collisions to edges and write edge_blackspots.json. Run from dubai/backend:
    python -m scripts.build_edge_blackspots
"""
import pandas as pd

from app.data.aggregates import PARQUET
from app.models.edge_blackspots import (
    EDGE_BLACKSPOTS_JSON,
    build_edge_blackspots,
    write_edge_blackspots,
)
from app.models.graph import build_dubai_graph


def main() -> None:
    df = pd.read_parquet(PARQUET)
    graph = build_dubai_graph()
    per, threshold, n = build_edge_blackspots(graph, df)
    write_edge_blackspots(per, threshold, n)
    n_black = sum(1 for e in per.values() if e["blackspot"])
    print(
        f"snapped {n:,} collisions -> {len(per):,} edges; "
        f"{n_black:,} blackspot edges (wsum >= {threshold:.1f})"
    )
    print(f"wrote {EDGE_BLACKSPOTS_JSON.name}")


if __name__ == "__main__":
    main()
