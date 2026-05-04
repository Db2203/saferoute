"""Compute a 0-100 risk score per OSM road segment.

Pipeline:
  1. Snap each accident to its single nearest OSM edge — accidents are
     point events that happened on one specific road, so single-nearest
     is the correct semantic.
  2. For each AADT count point, find every edge whose midpoint lies
     within 500 m via a BallTree haversine query — count points are
     area-level traffic estimates, so they should broadcast to all
     nearby edges. This is the difference between ~3% direct AADT
     coverage and ~50%+.
  3. Per edge with at least one accident, compute:
        raw = (accident_count * sum(severity_weight)) / aadt
     where severity_weight is Fatal=3, Serious=2, Slight=1 (inverted
     from the STATS19 codes via `4 - severity`).
  4. For edges that no AADT point reached, fall back to the median
     AADT for that road class (e.g. "primary", "residential"); if even
     that is missing, fall back to the global median.
  5. Normalize raw scores to 0-100.

The aggregation/scoring helpers are pure pandas so they're trivially
unit-tested. The spatial steps live in their own functions and are
exercised end-to-end by `build_risk_scores.py` plus a test that uses
a small synthetic networkx graph.
"""
from __future__ import annotations

import numpy as np
import osmnx as ox
import pandas as pd
from sklearn.neighbors import BallTree

# STATS19 severity codes are inverted: 1=fatal, 3=slight. The risk formula
# wants "more severe = bigger weight," so we flip with 4 - severity.
SEVERITY_WEIGHT = {1: 3, 2: 2, 3: 1}

EARTH_RADIUS_M = 6_371_000.0


def snap_accidents_to_edges(graph, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Return an (N, 3) array of (u, v, key) for the nearest edge to each accident."""
    edges = ox.distance.nearest_edges(graph, X=longitudes, Y=latitudes)
    return np.asarray(edges)


def aggregate_accidents_to_edges(accidents_with_edges: pd.DataFrame) -> pd.DataFrame:
    df = accidents_with_edges.copy()
    df["severity_weight"] = df["collision_severity"].map(SEVERITY_WEIGHT)
    return (
        df.groupby(["u", "v", "key"])
        .agg(
            accident_count=("collision_severity", "size"),
            severity_sum=("severity_weight", "sum"),
        )
        .reset_index()
    )


def _edge_midpoints(graph) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    edge_ids: list[tuple[int, int, int]] = []
    coords: list[tuple[float, float]] = []
    for u, v, key in graph.edges(keys=True):
        un = graph.nodes[u]
        vn = graph.nodes[v]
        edge_ids.append((u, v, key))
        coords.append(((un["y"] + vn["y"]) / 2.0, (un["x"] + vn["x"]) / 2.0))
    return edge_ids, np.asarray(coords)


def aadt_to_edges_within_radius(
    graph,
    aadt_df: pd.DataFrame,
    radius_m: float = 500.0,
) -> pd.DataFrame:
    """For each (u, v, key): mean of AADT count points within radius_m of the edge midpoint."""
    edge_ids, midpoint_coords = _edge_midpoints(graph)
    if not edge_ids or aadt_df.empty:
        return pd.DataFrame(columns=["u", "v", "key", "aadt"])

    midpoints_rad = np.radians(midpoint_coords)
    aadt_coords_rad = np.radians(aadt_df[["latitude", "longitude"]].to_numpy())

    tree = BallTree(midpoints_rad, metric="haversine")
    indices = tree.query_radius(aadt_coords_rad, r=radius_m / EARTH_RADIUS_M)

    aadt_values = aadt_df["all_motor_vehicles"].to_numpy()
    rows: list[tuple[int, int, int, float]] = []
    for aadt_i, edge_idxs in enumerate(indices):
        val = float(aadt_values[aadt_i])
        for ei in edge_idxs:
            u, v, key = edge_ids[ei]
            rows.append((u, v, key, val))

    if not rows:
        return pd.DataFrame(columns=["u", "v", "key", "aadt"])

    df = pd.DataFrame(rows, columns=["u", "v", "key", "aadt"])
    return df.groupby(["u", "v", "key"])["aadt"].mean().reset_index()


def build_road_class_lookup(graph) -> dict[tuple[int, int, int], str | None]:
    lookup: dict[tuple[int, int, int], str | None] = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        hw = data.get("highway")
        if isinstance(hw, list):
            hw = hw[0] if hw else None
        lookup[(u, v, key)] = hw
    return lookup


def compute_edge_scores(
    edge_accidents: pd.DataFrame,
    edge_aadt: pd.DataFrame,
    road_class_lookup: dict[tuple[int, int, int], str | None],
) -> pd.DataFrame:
    if edge_accidents.empty:
        return pd.DataFrame(
            columns=["u", "v", "key", "accident_count", "severity_sum", "aadt", "aadt_is_fallback", "raw_score", "risk_score"]
        )

    aadt_with_class = edge_aadt.copy()
    aadt_with_class["highway"] = [
        road_class_lookup.get((u, v, k))
        for u, v, k in zip(aadt_with_class["u"], aadt_with_class["v"], aadt_with_class["key"])
    ]
    class_median_aadt = (
        aadt_with_class.dropna(subset=["highway"])
        .groupby("highway")["aadt"]
        .median()
        .to_dict()
    )
    global_median = float(edge_aadt["aadt"].median()) if not edge_aadt.empty else 1.0

    merged = edge_accidents.merge(edge_aadt, on=["u", "v", "key"], how="left")
    merged["aadt_is_fallback"] = merged["aadt"].isna()

    if merged["aadt_is_fallback"].any():
        highway_lookup = [
            road_class_lookup.get((u, v, k))
            for u, v, k in zip(merged["u"], merged["v"], merged["key"])
        ]
        fallback_series = pd.Series(highway_lookup, index=merged.index).map(class_median_aadt).fillna(global_median)
        merged["aadt"] = merged["aadt"].fillna(fallback_series)

    merged["aadt"] = merged["aadt"].clip(lower=1.0)
    merged["raw_score"] = (merged["accident_count"] * merged["severity_sum"]) / merged["aadt"]

    max_raw = float(merged["raw_score"].max())
    if max_raw > 0:
        merged["risk_score"] = (merged["raw_score"] / max_raw) * 100.0
    else:
        merged["risk_score"] = 0.0

    return merged[
        ["u", "v", "key", "accident_count", "severity_sum", "aadt", "aadt_is_fallback", "raw_score", "risk_score"]
    ]


def score_road_segments(
    graph,
    accidents: pd.DataFrame,
    aadt: pd.DataFrame,
    aadt_radius_m: float = 500.0,
) -> pd.DataFrame:
    acc_edges = snap_accidents_to_edges(
        graph, accidents["latitude"].to_numpy(), accidents["longitude"].to_numpy()
    )
    accidents_with_edges = accidents.assign(
        u=acc_edges[:, 0], v=acc_edges[:, 1], key=acc_edges[:, 2]
    )

    edge_accidents = aggregate_accidents_to_edges(accidents_with_edges)
    edge_aadt = aadt_to_edges_within_radius(graph, aadt, radius_m=aadt_radius_m)
    road_class_lookup = build_road_class_lookup(graph)

    return compute_edge_scores(edge_accidents, edge_aadt, road_class_lookup)
