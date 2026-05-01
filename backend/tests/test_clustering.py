import pandas as pd

from app.models.clustering import cluster_hotspots, compute_cluster_centroids


def _accidents(points):
    return pd.DataFrame(
        [
            {
                "latitude": lat,
                "longitude": lng,
                "collision_severity": sev,
                "collision_index": idx,
            }
            for lat, lng, sev, idx in points
        ]
    )


def test_dense_points_form_a_single_cluster():
    pts = [
        (51.500, -0.100, 3, "A"),
        (51.501, -0.101, 3, "B"),
        (51.502, -0.099, 2, "C"),
        (51.500, -0.102, 3, "D"),
        (51.503, -0.100, 3, "E"),
        (51.501, -0.098, 1, "F"),
    ]
    out = cluster_hotspots(_accidents(pts), eps=0.01, min_samples=5)
    assert (out["cluster_id"] != -1).sum() == 6
    assert out["cluster_id"].nunique() == 1


def test_isolated_points_are_all_noise():
    pts = [
        (51.500, -0.100, 3, "A"),
        (53.000, -2.000, 3, "B"),  # manchester-ish
        (54.000, -1.500, 3, "C"),  # york-ish
    ]
    out = cluster_hotspots(_accidents(pts), eps=0.01, min_samples=5)
    assert (out["cluster_id"] == -1).all()


def test_centroids_exclude_noise_points():
    pts = [
        (51.500, -0.100, 3, "A"),
        (51.501, -0.101, 3, "B"),
        (51.502, -0.099, 2, "C"),
        (51.500, -0.102, 3, "D"),
        (51.503, -0.100, 3, "E"),
        (51.501, -0.098, 1, "F"),
        (53.000, -2.000, 3, "noise"),
    ]
    clustered = cluster_hotspots(_accidents(pts), eps=0.01, min_samples=5)
    centroids = compute_cluster_centroids(clustered)
    assert len(centroids) == 1
    assert centroids.iloc[0]["accident_count"] == 6


def test_centroid_lat_lng_is_mean_of_cluster():
    pts = [
        (51.500, -0.100, 3, "A"),
        (51.510, -0.110, 3, "B"),
        (51.490, -0.090, 3, "C"),
        (51.500, -0.100, 3, "D"),
        (51.500, -0.100, 3, "E"),
    ]
    clustered = cluster_hotspots(_accidents(pts), eps=0.05, min_samples=5)
    centroids = compute_cluster_centroids(clustered)
    assert len(centroids) == 1
    assert abs(centroids.iloc[0]["latitude"] - 51.500) < 1e-3
    assert abs(centroids.iloc[0]["longitude"] + 0.100) < 1e-3


def test_avg_severity_weight_uses_inverted_codes():
    # severity 1 (fatal) -> weight 3
    pts = [(51.500, -0.100, 1, str(i)) for i in range(5)]
    clustered = cluster_hotspots(_accidents(pts), eps=0.01, min_samples=5)
    centroids = compute_cluster_centroids(clustered)
    assert centroids.iloc[0]["avg_severity_weight"] == 3.0


def test_empty_input_returns_empty_centroids():
    pts = [
        (51.500, -0.100, 3, "A"),
        (53.000, -2.000, 3, "B"),
    ]
    clustered = cluster_hotspots(_accidents(pts), eps=0.01, min_samples=5)
    centroids = compute_cluster_centroids(clustered)
    assert centroids.empty
