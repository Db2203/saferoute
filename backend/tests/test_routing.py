import networkx as nx
import pytest

from app.models.routing import (
    compute_fastest,
    compute_route,
    compute_safest,
    to_routing_scores,
)


def _toy_graph():
    """Two parallel paths between node 1 and node 3:

        1 → 2 → 3   long but zero-risk     (240s travel, 0 risk)
        1 → 4 → 3   short but high-risk    (80s travel, 180 risk)
    """
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(1, x=-0.10, y=51.500)
    g.add_node(2, x=-0.09, y=51.510)
    g.add_node(3, x=-0.08, y=51.500)
    g.add_node(4, x=-0.09, y=51.490)

    g.add_edge(1, 2, key=0, length=1500, travel_time=120)
    g.add_edge(2, 3, key=0, length=1500, travel_time=120)
    g.add_edge(1, 4, key=0, length=500, travel_time=40)
    g.add_edge(4, 3, key=0, length=500, travel_time=40)

    risk_scores = {
        (1, 2, 0): 0.0,
        (2, 3, 0): 0.0,
        (1, 4, 0): 90.0,
        (4, 3, 0): 90.0,
    }
    return g, risk_scores


def test_fastest_picks_short_high_risk_path():
    g, risk = _toy_graph()
    res = compute_fastest(g, (51.500, -0.10), (51.500, -0.08), risk)
    assert [u for u, v, k in res.edges] == [1, 4]
    assert res.total_time_s == pytest.approx(80.0)
    assert res.total_risk == pytest.approx(180.0)
    assert res.total_distance_m == pytest.approx(1000.0)


def test_safest_picks_long_zero_risk_path():
    g, risk = _toy_graph()
    res = compute_safest(g, (51.500, -0.10), (51.500, -0.08), risk)
    assert [u for u, v, k in res.edges] == [1, 2]
    assert res.total_time_s == pytest.approx(240.0)
    assert res.total_risk == pytest.approx(0.0)


def test_alpha_lever_swaps_chosen_path():
    g, risk = _toy_graph()

    pure_time = compute_route(g, (51.500, -0.10), (51.500, -0.08), risk, alpha=1.0)
    pure_risk = compute_route(g, (51.500, -0.10), (51.500, -0.08), risk, alpha=0.0)

    assert pure_time.total_time_s < pure_risk.total_time_s
    assert pure_time.total_risk > pure_risk.total_risk


def test_route_geometry_starts_and_ends_at_endpoints():
    g, risk = _toy_graph()
    res = compute_fastest(g, (51.500, -0.10), (51.500, -0.08), risk)
    assert res.geometry[0] == pytest.approx((-0.10, 51.500))
    assert res.geometry[-1] == pytest.approx((-0.08, 51.500))


def test_temporal_multiplier_amplifies_risk_dimension():
    g, risk = _toy_graph()

    # at alpha=0.5 with mult=1.0: short = 0.5*40 + 0.5*90 = 65 per edge → 130
    #                            safe  = 0.5*120 + 0     = 60 per edge → 120; safe wins
    base = compute_route(g, (51.500, -0.10), (51.500, -0.08), risk, alpha=0.5, temporal_multiplier=1.0)

    # at alpha=0.5 with mult=0.1: short = 0.5*40 + 0.5*90*0.1 = 24.5/edge → 49
    #                            safe  = 0.5*120              = 60/edge   → 120; short wins
    low = compute_route(g, (51.500, -0.10), (51.500, -0.08), risk, alpha=0.5, temporal_multiplier=0.1)

    assert [u for u, v, k in base.edges] == [1, 2]
    assert [u for u, v, k in low.edges] == [1, 4]


def test_to_routing_scores_maps_min_to_zero_and_max_to_hundred():
    raw = {(1, 2, 0): 0.001, (3, 4, 0): 0.5, (5, 6, 0): 100.0, (7, 8, 0): 0.01}
    out = to_routing_scores(raw)
    sorted_pcts = sorted(out.values())
    assert sorted_pcts[0] == 0.0
    assert sorted_pcts[-1] == 100.0
    # the sorted ranks should be evenly spaced
    assert sorted_pcts == pytest.approx([0.0, 100 / 3, 200 / 3, 100.0])


def test_to_routing_scores_is_empty_for_empty_input():
    assert to_routing_scores({}) == {}


def test_to_routing_scores_preserves_relative_order():
    raw = {(1, 2, 0): 5.0, (3, 4, 0): 1.0, (5, 6, 0): 9.0}
    out = to_routing_scores(raw)
    assert out[(3, 4, 0)] < out[(1, 2, 0)] < out[(5, 6, 0)]


def test_picks_lowest_weight_among_parallel_edges():
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=1.0, y=0.0)
    # two parallel edges between 1 and 2 with different travel times
    g.add_edge(1, 2, key=0, length=1000, travel_time=200)  # slower
    g.add_edge(1, 2, key=1, length=800, travel_time=80)    # faster
    risk = {(1, 2, 0): 0.0, (1, 2, 1): 0.0}

    res = compute_fastest(g, (0.0, 0.0), (0.0, 1.0), risk)
    assert res.edges == [(1, 2, 1)]
    assert res.total_time_s == pytest.approx(80.0)
