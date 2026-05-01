"""DBSCAN clustering of accident locations to find spatial hotspots.

Runs DBSCAN on (latitude, longitude) in degree space. For the London bbox
the bias from Earth curvature is negligible: eps=0.001 degrees is roughly
100 m in real distance.

Defaults are tuned for London accident density (~70 accidents/km²):
the blueprint's eps=0.01 / min_samples=5 collapsed nearly everything into
one mega-cluster. eps=0.001 + min_samples=10 picks out genuine
intersection-scale concentrations.

`avg_severity_weight` follows the blueprint's risk formula where Fatal=3,
Serious=2, Slight=1 (higher = more severe). STATS19 ships severity inverted
(1=fatal, 3=slight) so we flip with `4 - severity`.
"""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN


def cluster_hotspots(
    accidents: pd.DataFrame,
    eps: float = 0.001,
    min_samples: int = 10,
) -> pd.DataFrame:
    coords = accidents[["latitude", "longitude"]].to_numpy()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    out = accidents.copy()
    out["cluster_id"] = labels
    return out


def compute_cluster_centroids(clustered: pd.DataFrame) -> pd.DataFrame:
    valid = clustered[clustered["cluster_id"] != -1]
    if valid.empty:
        return pd.DataFrame(
            columns=["cluster_id", "latitude", "longitude", "accident_count", "avg_severity_weight"]
        )
    return (
        valid.groupby("cluster_id")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            accident_count=("collision_index", "count"),
            avg_severity_weight=("collision_severity", lambda s: (4 - s).mean()),
        )
        .reset_index()
    )
