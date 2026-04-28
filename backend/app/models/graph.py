"""London driveable road graph build + cache.

OSMnx pulls the network from OpenStreetMap's Overpass API, which is slow
(~1-3 min for the London bbox) and needs internet. We download once, pickle
the resulting `networkx.MultiDiGraph` to disk, and reuse via `load_london_graph`.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import osmnx as ox

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "cache" / "london_graph.pkl"

# Pin OSMnx's overpass-response cache under data/cache/ so it doesn't dump
# into whatever the cwd happens to be at run time.
ox.settings.cache_folder = str(REPO_ROOT / "data" / "cache" / "osmnx")

# Matches LONDON_BBOX in app.data.preprocessing
LONDON_NORTH = 51.69
LONDON_SOUTH = 51.28
LONDON_EAST = 0.33
LONDON_WEST = -0.51


def build_london_graph(cache_path: Path | None = None, *, force: bool = False) -> nx.MultiDiGraph:
    cache = cache_path or DEFAULT_CACHE_PATH
    if cache.exists() and not force:
        return load_london_graph(cache)

    graph = ox.graph_from_bbox(
        bbox=(LONDON_NORTH, LONDON_SOUTH, LONDON_EAST, LONDON_WEST),
        network_type="drive",
        simplify=True,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(graph, f)
    return graph


def load_london_graph(cache_path: Path | None = None) -> nx.MultiDiGraph:
    cache = cache_path or DEFAULT_CACHE_PATH
    with cache.open("rb") as f:
        return pickle.load(f)
