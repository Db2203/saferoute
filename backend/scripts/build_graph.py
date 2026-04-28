"""Build / refresh the London road graph cache.

    cd backend
    .venv/Scripts/python -m scripts.build_graph             # uses cache if present
    .venv/Scripts/python -m scripts.build_graph --rebuild   # force redownload
"""
from __future__ import annotations

import argparse
import time

from app.models.graph import build_london_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if cache exists")
    args = parser.parse_args()

    started = time.perf_counter()
    graph = build_london_graph(force=args.rebuild)
    elapsed = time.perf_counter() - started

    print(f"graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
