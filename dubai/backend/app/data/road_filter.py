"""Drop collisions whose coordinates are not on/near a real road.

Dubai Pulse has a tail (~3%) of incidents geocoded into water, desert, or stacked
on placeholder/default coordinates. A real collision happens on a road, so we
keep only those within MAX_ROAD_DIST_M of the OpenStreetMap vehicle-road network
(measured as true point-to-line distance in metres on a UTM projection).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer

from app.models.graph import DUBAI_ROOT

FILTER_GRAPH_PATH = DUBAI_ROOT / "data" / "cache" / "dubai_filter_graph.pkl"
# Full collision bbox + small margin (N, S, E, W) so boundary points aren't
# falsely dropped. drive_service = vehicle roads incl. service roads/parking.
FILTER_BBOX = (25.55, 24.65, 55.85, 54.75)
MAX_ROAD_DIST_M = 150.0


def build_filter_graph(cache_path: Path | None = None, *, force: bool = False):
    cache = cache_path or FILTER_GRAPH_PATH
    if cache.exists() and not force:
        with cache.open("rb") as f:
            return pickle.load(f)
    n, s, e, w = FILTER_BBOX
    graph = ox.graph_from_bbox(bbox=(n, s, e, w), network_type="drive_service", simplify=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(graph, f)
    return graph


def road_distances_m(df: pd.DataFrame, graph) -> np.ndarray:
    """True distance (metres) from each (lat, lng) to the nearest road edge."""
    projected = ox.project_graph(graph)
    transformer = Transformer.from_crs("EPSG:4326", projected.graph["crs"], always_xy=True)
    xp, yp = transformer.transform(df["lng"].to_numpy(), df["lat"].to_numpy())
    _, dist = ox.distance.nearest_edges(projected, X=xp, Y=yp, return_dist=True)
    return np.asarray(dist, dtype=float)


def filter_to_roads(df: pd.DataFrame, graph, max_dist_m: float = MAX_ROAD_DIST_M) -> pd.DataFrame:
    keep = road_distances_m(df, graph) <= max_dist_m
    return df[keep].reset_index(drop=True)
