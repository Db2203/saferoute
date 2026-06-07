"""Snap collisions to road edges and flag the worst as blackspots.

Used by the 'check your route' feature: count how many blackspot edges a route
crosses. The snapping (osmnx) is separated from the pure aggregation so the
aggregation is unit-testable without a real graph.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

from app.models.graph import (
    DUBAI_EAST,
    DUBAI_NORTH,
    DUBAI_ROOT,
    DUBAI_SOUTH,
    DUBAI_WEST,
)

EDGE_BLACKSPOTS_JSON = DUBAI_ROOT / "data" / "processed" / "edge_blackspots.json"


def aggregate_edge_blackspots(edges, severe, blackspot_pct: float = 0.05):
    """edges: iterable of (u, v, key); severe: iterable of bool.
    Returns (per_edge dict, blackspot wsum threshold). wsum = count + severe
    (minor weight 1, severe weight 2)."""
    per: dict[tuple, dict] = {}
    for (u, v, k), sev in zip(edges, severe):
        e = per.setdefault((u, v, k), {"count": 0, "severe": 0})
        e["count"] += 1
        if sev:
            e["severe"] += 1
    for e in per.values():
        e["wsum"] = e["count"] + e["severe"]
    threshold = (
        float(np.quantile([e["wsum"] for e in per.values()], 1 - blackspot_pct))
        if per
        else 0.0
    )
    for e in per.values():
        e["blackspot"] = bool(e["wsum"] >= threshold)
    return per, threshold


def route_blackspots(node_path, edge_index: dict, max_k: int = 6) -> dict:
    """Count the blackspots a routed node path crosses.

    Direction-agnostic: a collision snapped to edge (a->b) must still be found
    when a route traverses (b->a) — same physical road. Per hop we take the
    worst (max-wsum) of the two directions so a two-way segment isn't counted
    twice. Returns the blackspot segments crossed + total risk exposure.
    """
    crossed = []
    risk = 0.0
    for a, b in zip(node_path, node_path[1:]):
        entries = [
            edge_index[key]
            for k in range(max_k)
            for key in (f"{a}_{b}_{k}", f"{b}_{a}_{k}")
            if key in edge_index
        ]
        if not entries:
            continue
        worst = max(entries, key=lambda e: e["wsum"])
        risk += worst["wsum"]
        if worst["blackspot"]:
            crossed.append({"u": a, "v": b, "count": worst["count"], "severe": worst["severe"], "wsum": worst["wsum"]})
    return {"n_blackspots": len(crossed), "risk_exposure": round(risk, 1), "blackspots": crossed}


def _in_bbox(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["lat"].between(DUBAI_SOUTH, DUBAI_NORTH) & df["lng"].between(DUBAI_WEST, DUBAI_EAST)]


def build_edge_blackspots(graph, df: pd.DataFrame, blackspot_pct: float = 0.05):
    sub = _in_bbox(df)
    ne = np.asarray(ox.distance.nearest_edges(graph, X=sub["lng"].to_numpy(), Y=sub["lat"].to_numpy()))
    edges = [(int(u), int(v), int(k)) for u, v, k in ne]
    severe = sub["severity"].to_numpy() == "severe"
    per, threshold = aggregate_edge_blackspots(edges, severe, blackspot_pct)
    return per, threshold, len(sub)


def write_edge_blackspots(per: dict, threshold: float, n_snapped: int, path: Path | None = None) -> Path:
    out = {
        "meta": {
            "bbox": [DUBAI_SOUTH, DUBAI_WEST, DUBAI_NORTH, DUBAI_EAST],
            "snapped_collisions": int(n_snapped),
            "edges_with_collisions": len(per),
            "blackspot_threshold_wsum": round(threshold, 2),
            "n_blackspots": sum(1 for e in per.values() if e["blackspot"]),
        },
        "edges": {f"{u}_{v}_{k}": e for (u, v, k), e in per.items()},
    }
    dest = path or EDGE_BLACKSPOTS_JSON
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out), encoding="utf-8")
    return dest
