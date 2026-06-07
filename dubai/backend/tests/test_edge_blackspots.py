from app.models.edge_blackspots import aggregate_edge_blackspots, route_blackspots
from app.models.graph import DUBAI_EAST, DUBAI_NORTH, DUBAI_SOUTH, DUBAI_WEST


def test_aggregate_counts_and_flags():
    edges = [(1, 2, 0)] * 10 + [(3, 4, 0)] * 2 + [(5, 6, 0)] * 1
    severe = [True] * 3 + [False] * 7 + [False] * 2 + [True] * 1
    per, thr = aggregate_edge_blackspots(edges, severe, blackspot_pct=0.2)
    assert per[(1, 2, 0)] == {"count": 10, "severe": 3, "wsum": 13, "blackspot": True}
    assert per[(3, 4, 0)]["wsum"] == 2 and per[(3, 4, 0)]["blackspot"] is False
    assert per[(5, 6, 0)] == {"count": 1, "severe": 1, "wsum": 2, "blackspot": False}


def test_aggregate_empty():
    per, thr = aggregate_edge_blackspots([], [], blackspot_pct=0.05)
    assert per == {} and thr == 0.0


def test_route_blackspots_is_direction_agnostic():
    # collision data stored on 1->2 and on 6->5 (reverse of travel direction)
    edge_index = {
        "1_2_0": {"count": 50, "severe": 10, "wsum": 60, "blackspot": True},
        "3_4_0": {"count": 2, "severe": 0, "wsum": 2, "blackspot": False},
        "6_5_0": {"count": 40, "severe": 5, "wsum": 45, "blackspot": True},
    }
    # route travels 1->2->4->5->6 ; the 5->6 hop must match reverse key 6_5_0
    result = route_blackspots([1, 2, 4, 5, 6], edge_index)
    assert result["n_blackspots"] == 2
    assert result["risk_exposure"] == 105.0  # 60 + 45 (2->4 and 4->5 have no data)
    crossed = {(b["u"], b["v"]) for b in result["blackspots"]}
    assert (1, 2) in crossed and (5, 6) in crossed


def test_route_blackspots_no_data_hops():
    result = route_blackspots([10, 11, 12], {})
    assert result["n_blackspots"] == 0 and result["risk_exposure"] == 0.0


def test_graph_bbox_sane():
    assert DUBAI_NORTH > DUBAI_SOUTH
    assert DUBAI_EAST > DUBAI_WEST
    assert 24.7 < DUBAI_SOUTH < DUBAI_NORTH < 25.6
    assert 54.8 < DUBAI_WEST < DUBAI_EAST < 55.8
