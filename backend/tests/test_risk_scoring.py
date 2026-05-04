import networkx as nx
import pandas as pd

from app.models.risk_scoring import (
    aadt_to_edges_within_radius,
    aggregate_accidents_to_edges,
    compute_edge_scores,
)


def test_accident_aggregation_counts_and_severity_sum():
    df = pd.DataFrame(
        [
            {"u": 1, "v": 2, "key": 0, "collision_severity": 1, "collision_index": "A"},  # weight 3
            {"u": 1, "v": 2, "key": 0, "collision_severity": 2, "collision_index": "B"},  # weight 2
            {"u": 1, "v": 2, "key": 0, "collision_severity": 3, "collision_index": "C"},  # weight 1
            {"u": 2, "v": 3, "key": 0, "collision_severity": 3, "collision_index": "D"},  # weight 1
        ]
    )
    out = aggregate_accidents_to_edges(df)
    out_sorted = out.sort_values(["u", "v"]).reset_index(drop=True)

    assert list(out_sorted["accident_count"]) == [3, 1]
    assert list(out_sorted["severity_sum"]) == [6, 1]


def _tiny_graph_with_two_separated_edges():
    # edge (1, 2): around (51.5000, -0.1000), ~50m long
    # edge (3, 4): around (51.5500, -0.1500), ~50m long, several km away
    g = nx.MultiDiGraph()
    g.add_node(1, x=-0.1000, y=51.5000)
    g.add_node(2, x=-0.0995, y=51.5005)
    g.add_node(3, x=-0.1500, y=51.5500)
    g.add_node(4, x=-0.1495, y=51.5505)
    g.add_edge(1, 2, key=0)
    g.add_edge(3, 4, key=0)
    return g


def test_aadt_radius_assigns_only_to_nearby_edges():
    g = _tiny_graph_with_two_separated_edges()
    # AADT point near edge (1,2) — should hit it but not (3,4) since they're ~6km apart
    aadt = pd.DataFrame(
        [{"latitude": 51.5002, "longitude": -0.0997, "all_motor_vehicles": 10000}]
    )
    out = aadt_to_edges_within_radius(g, aadt, radius_m=500)

    assert len(out) == 1
    row = out.iloc[0]
    assert (row["u"], row["v"], row["key"]) == (1, 2, 0)
    assert row["aadt"] == 10000


def test_aadt_radius_averages_when_multiple_points_cover_an_edge():
    g = _tiny_graph_with_two_separated_edges()
    aadt = pd.DataFrame(
        [
            {"latitude": 51.5002, "longitude": -0.0997, "all_motor_vehicles": 10000},
            {"latitude": 51.5001, "longitude": -0.0998, "all_motor_vehicles": 14000},
        ]
    )
    out = aadt_to_edges_within_radius(g, aadt, radius_m=500)
    edge12 = out[(out["u"] == 1) & (out["v"] == 2)].iloc[0]
    assert edge12["aadt"] == 12000  # mean of 10000 and 14000


def test_aadt_radius_one_point_can_cover_multiple_nearby_edges():
    g = nx.MultiDiGraph()
    # three edges all clustered within a few hundred metres
    g.add_node(1, x=-0.1000, y=51.5000)
    g.add_node(2, x=-0.0995, y=51.5005)
    g.add_node(3, x=-0.1005, y=51.5005)
    g.add_node(4, x=-0.0998, y=51.5008)
    g.add_edge(1, 2, key=0)
    g.add_edge(1, 3, key=0)
    g.add_edge(2, 4, key=0)

    aadt = pd.DataFrame(
        [{"latitude": 51.5003, "longitude": -0.1000, "all_motor_vehicles": 8000}]
    )
    out = aadt_to_edges_within_radius(g, aadt, radius_m=500)
    assert len(out) == 3
    assert (out["aadt"] == 8000).all()


def test_aadt_radius_returns_empty_when_no_overlap():
    g = _tiny_graph_with_two_separated_edges()
    aadt = pd.DataFrame(
        [{"latitude": 50.0, "longitude": 1.0, "all_motor_vehicles": 5000}]  # off-graph
    )
    out = aadt_to_edges_within_radius(g, aadt, radius_m=500)
    assert out.empty


def test_compute_edge_scores_uses_aadt_and_normalizes_to_100():
    edge_accidents = pd.DataFrame(
        [
            {"u": 1, "v": 2, "key": 0, "accident_count": 10, "severity_sum": 20},  # raw = 200/1000 = 0.2
            {"u": 3, "v": 4, "key": 0, "accident_count": 1, "severity_sum": 1},   # raw = 1/1000 = 0.001
        ]
    )
    edge_aadt = pd.DataFrame(
        [
            {"u": 1, "v": 2, "key": 0, "aadt": 1000.0},
            {"u": 3, "v": 4, "key": 0, "aadt": 1000.0},
        ]
    )
    road_class = {(1, 2, 0): "primary", (3, 4, 0): "residential"}

    out = compute_edge_scores(edge_accidents, edge_aadt, road_class).sort_values("u").reset_index(drop=True)

    # max raw = 0.2 → that edge becomes 100; the other edge is (0.001/0.2)*100 = 0.5
    assert out.iloc[0]["risk_score"] == 100.0
    assert abs(out.iloc[1]["risk_score"] - 0.5) < 1e-6
    assert not out["aadt_is_fallback"].any()


def test_compute_edge_scores_falls_back_to_road_class_median():
    # edge (1,2,0) has its own AADT; edge (5,6,0) has none → uses primary class median
    edge_accidents = pd.DataFrame(
        [
            {"u": 1, "v": 2, "key": 0, "accident_count": 5, "severity_sum": 10},
            {"u": 5, "v": 6, "key": 0, "accident_count": 1, "severity_sum": 1},
        ]
    )
    edge_aadt = pd.DataFrame(
        [
            {"u": 1, "v": 2, "key": 0, "aadt": 1000.0},
            {"u": 7, "v": 8, "key": 0, "aadt": 4000.0},  # other primary edge contributes to median
        ]
    )
    road_class = {(1, 2, 0): "primary", (5, 6, 0): "primary", (7, 8, 0): "primary"}

    out = compute_edge_scores(edge_accidents, edge_aadt, road_class).sort_values("u").reset_index(drop=True)

    edge_56 = out[(out["u"] == 5) & (out["v"] == 6)].iloc[0]
    assert edge_56["aadt_is_fallback"] is True or edge_56["aadt_is_fallback"] == True  # numpy bool quirk
    # primary median = median of [1000, 4000] = 2500
    assert edge_56["aadt"] == 2500.0


def test_compute_edge_scores_global_fallback_when_class_unknown():
    # edge (5,6,0) has no AADT and unknown road class → uses global median
    edge_accidents = pd.DataFrame(
        [{"u": 5, "v": 6, "key": 0, "accident_count": 1, "severity_sum": 1}]
    )
    edge_aadt = pd.DataFrame(
        [
            {"u": 1, "v": 2, "key": 0, "aadt": 1000.0},
            {"u": 3, "v": 4, "key": 0, "aadt": 5000.0},
        ]
    )
    road_class = {(1, 2, 0): "primary", (3, 4, 0): "secondary"}  # (5,6,0) absent

    out = compute_edge_scores(edge_accidents, edge_aadt, road_class)
    # global median = median of [1000, 5000] = 3000
    assert out.iloc[0]["aadt"] == 3000.0
    assert bool(out.iloc[0]["aadt_is_fallback"]) is True


def test_compute_edge_scores_zero_aadt_is_clipped():
    edge_accidents = pd.DataFrame(
        [{"u": 1, "v": 2, "key": 0, "accident_count": 5, "severity_sum": 10}]
    )
    edge_aadt = pd.DataFrame(
        [{"u": 1, "v": 2, "key": 0, "aadt": 0.0}]  # zero traffic recorded
    )
    out = compute_edge_scores(edge_accidents, edge_aadt, {(1, 2, 0): "primary"})

    # raw should be 50/1.0 (clipped) = 50, normalized to 100
    assert out.iloc[0]["raw_score"] == 50.0
    assert out.iloc[0]["risk_score"] == 100.0


def test_compute_edge_scores_handles_empty_input():
    out = compute_edge_scores(
        pd.DataFrame(columns=["u", "v", "key", "accident_count", "severity_sum"]),
        pd.DataFrame(columns=["u", "v", "key", "aadt"]),
        {},
    )
    assert out.empty
    assert list(out.columns) == ["u", "v", "key", "accident_count", "severity_sum", "aadt", "aadt_is_fallback", "raw_score", "risk_score"]
