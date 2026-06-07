"""Central-Dubai driveable road graph build + cache.

OSMnx pulls the network from OpenStreetMap's Overpass API (slow, needs
internet). We build once, pickle the networkx graph, and reuse from cache.
The bbox is central Dubai (Marina/JLT up through Bur Dubai/Deira) — the main
urban routing area, validated to build in ~30s (~41k nodes / ~83k edges).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import osmnx as ox

DUBAI_ROOT = Path(__file__).resolve().parents[3]
CACHE_PATH = DUBAI_ROOT / "data" / "cache" / "dubai_graph.pkl"
ox.settings.cache_folder = str(DUBAI_ROOT / "data" / "cache" / "osmnx")

DUBAI_NORTH = 25.30
DUBAI_SOUTH = 25.05
DUBAI_EAST = 55.45
DUBAI_WEST = 55.10


def build_dubai_graph(cache_path: Path | None = None, *, force: bool = False) -> nx.MultiDiGraph:
    cache = cache_path or CACHE_PATH
    if cache.exists() and not force:
        return load_dubai_graph(cache)
    graph = ox.graph_from_bbox(
        bbox=(DUBAI_NORTH, DUBAI_SOUTH, DUBAI_EAST, DUBAI_WEST),
        network_type="drive",
        simplify=True,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(graph, f)
    return graph


def load_dubai_graph(cache_path: Path | None = None) -> nx.MultiDiGraph:
    cache = cache_path or CACHE_PATH
    with cache.open("rb") as f:
        return pickle.load(f)
