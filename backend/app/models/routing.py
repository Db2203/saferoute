"""Modified Dijkstra routing for safety-aware navigation.

Each edge weight is `alpha * travel_time_s + (1 - alpha) * risk_score * temporal_multiplier`:

- `alpha = 1.0` → pure fastest (Google-Maps style)
- `alpha = 0.0` → pure safest (route avoids high-risk roads regardless of time)
- `alpha = 0.5` → balanced default
- `alpha = 0.3` → the "safest" preset shipped to the frontend

`temporal_multiplier` comes from `app.models.temporal.predict_risk_multiplier`
and adjusts the apparent risk by current time/weather context. Pass `1.0`
to ignore the temporal layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import osmnx as ox
from sqlalchemy import text

from app.db.connection import engine

EdgeKey = tuple[int, int, int]

# Fallback travel speed when an edge has no length-derived speed at all.
_FALLBACK_SPEED_MPS = 8.33  # ~30 km/h


@dataclass
class RouteResult:
    nodes: list[int]
    edges: list[EdgeKey]
    geometry: list[tuple[float, float]]  # (lng, lat) sequence — GeoJSON-ready
    total_time_s: float
    total_risk: float
    total_distance_m: float


def load_risk_scores_from_db() -> dict[EdgeKey, float]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT u, v, key, risk_score FROM road_risk_scores")
        ).fetchall()
    return {(int(r.u), int(r.v), int(r.key)): float(r.risk_score) for r in rows}


def to_routing_scores(risk_scores: dict[EdgeKey, float]) -> dict[EdgeKey, float]:
    """Convert raw risk_scores to percentile ranks (0-100) for routing.

    Raw scores from `road_risk_scores` are heavily long-tailed (median <0.01,
    p99 <0.25, max=100), so travel_time dominates the cost function and the
    alpha lever has no continuous range — at α=0.3 routes are identical to
    α=1.0, at α=0.0 routes detour absurdly through residential streets.
    Percentile-ranking gives every scored edge a meaningful position in
    [0, 100] and restores the smooth time-vs-risk tradeoff.

    Edges absent from the input dict (no accidents in the data) keep their
    `risk_scores.get((u,v,k), 0.0)` default in `compute_route`, so they're
    treated as the safest option — a documented MVP simplification.
    """
    if not risk_scores:
        return {}
    keys = list(risk_scores.keys())
    values = np.fromiter((risk_scores[k] for k in keys), dtype=float, count=len(keys))
    ranks = values.argsort().argsort()
    pct = ranks / max(len(values) - 1, 1) * 100.0
    return {k: float(p) for k, p in zip(keys, pct)}


def ensure_travel_times(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if graph.graph.get("_travel_times_added"):
        return graph
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    graph.graph["_travel_times_added"] = True
    return graph


def _edge_weight(travel_time: float, risk: float, alpha: float, mult: float) -> float:
    return alpha * travel_time + (1.0 - alpha) * risk * mult


def _travel_time(attrs: dict) -> float:
    tt = attrs.get("travel_time")
    if tt is not None:
        return float(tt)
    length = float(attrs.get("length", 0.0))
    return length / _FALLBACK_SPEED_MPS


def _make_weight_fn(risk_scores: dict[EdgeKey, float], alpha: float, mult: float):
    def weight_fn(u, v, edge_data):
        best = float("inf")
        for key, attrs in edge_data.items():
            travel_time = _travel_time(attrs)
            risk = risk_scores.get((u, v, key), 0.0)
            w = _edge_weight(travel_time, risk, alpha, mult)
            if w < best:
                best = w
        return best
    return weight_fn


def _best_edge_key(
    graph: nx.MultiDiGraph,
    u: int,
    v: int,
    risk_scores: dict[EdgeKey, float],
    alpha: float,
    mult: float,
) -> int:
    edge_data = graph[u][v]
    best_key = next(iter(edge_data))
    best_w = float("inf")
    for key, attrs in edge_data.items():
        travel_time = _travel_time(attrs)
        risk = risk_scores.get((u, v, key), 0.0)
        w = _edge_weight(travel_time, risk, alpha, mult)
        if w < best_w:
            best_w = w
            best_key = key
    return best_key


def _route_geometry(graph: nx.MultiDiGraph, edges: list[EdgeKey]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for i, (u, v, key) in enumerate(edges):
        attrs = graph.edges[u, v, key]
        if "geometry" in attrs:
            segment = list(attrs["geometry"].coords)
        else:
            segment = [
                (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                (graph.nodes[v]["x"], graph.nodes[v]["y"]),
            ]
        if i == 0:
            coords.extend(segment)
        else:
            # skip the duplicate junction node shared with the previous edge
            coords.extend(segment[1:])
    return coords


def compute_route(
    graph: nx.MultiDiGraph,
    origin: tuple[float, float],
    dest: tuple[float, float],
    risk_scores: dict[EdgeKey, float],
    *,
    alpha: float = 0.5,
    temporal_multiplier: float = 1.0,
) -> RouteResult:
    """Compute a route from origin (lat, lng) to dest (lat, lng) balancing time vs risk."""
    o_lat, o_lng = origin
    d_lat, d_lng = dest

    o_node = int(ox.distance.nearest_nodes(graph, X=o_lng, Y=o_lat))
    d_node = int(ox.distance.nearest_nodes(graph, X=d_lng, Y=d_lat))

    weight_fn = _make_weight_fn(risk_scores, alpha, temporal_multiplier)
    node_path = nx.shortest_path(graph, o_node, d_node, weight=weight_fn)

    edges: list[EdgeKey] = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        edges.append((u, v, _best_edge_key(graph, u, v, risk_scores, alpha, temporal_multiplier)))

    total_time_s = 0.0
    total_risk = 0.0
    total_distance_m = 0.0
    for u, v, key in edges:
        attrs = graph.edges[u, v, key]
        total_time_s += _travel_time(attrs)
        total_risk += risk_scores.get((u, v, key), 0.0)
        total_distance_m += float(attrs.get("length", 0.0))

    return RouteResult(
        nodes=node_path,
        edges=edges,
        geometry=_route_geometry(graph, edges),
        total_time_s=total_time_s,
        total_risk=total_risk,
        total_distance_m=total_distance_m,
    )


def compute_fastest(
    graph: nx.MultiDiGraph,
    origin: tuple[float, float],
    dest: tuple[float, float],
    risk_scores: dict[EdgeKey, float],
    *,
    temporal_multiplier: float = 1.0,
) -> RouteResult:
    return compute_route(graph, origin, dest, risk_scores, alpha=1.0, temporal_multiplier=temporal_multiplier)


def compute_safest(
    graph: nx.MultiDiGraph,
    origin: tuple[float, float],
    dest: tuple[float, float],
    risk_scores: dict[EdgeKey, float],
    *,
    temporal_multiplier: float = 1.0,
) -> RouteResult:
    return compute_route(graph, origin, dest, risk_scores, alpha=0.3, temporal_multiplier=temporal_multiplier)
