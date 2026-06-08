import networkx as nx
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def setup_function():
    # reset app state before each test (lifespan isn't run by the test client)
    app.state.stats = None
    app.state.blackspots = None
    app.state.edge_index = None
    app.state.graph = None
    app.state.severity = None
    app.state.df = None


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_stats_503_when_missing():
    assert client.get("/api/stats").status_code == 503


def test_stats_returns_when_loaded():
    app.state.stats = {"summary": {"total": 7}}
    r = client.get("/api/stats")
    assert r.status_code == 200 and r.json()["summary"]["total"] == 7


def test_blackspots_severe_only_filter():
    app.state.blackspots = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [55.30, 25.20]},
             "properties": {"count": 10, "severe": 0}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [55.31, 25.21]},
             "properties": {"count": 8, "severe": 3}},
        ],
    }
    assert len(client.get("/api/blackspots").json()["features"]) == 2
    assert len(client.get("/api/blackspots?severe_only=true").json()["features"]) == 1


def test_route_503_when_no_graph():
    assert client.get("/api/route-blackspots?origin=25.2,55.3&dest=25.21,55.31").status_code == 503


def _tiny_graph():
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(1, x=55.300, y=25.200)
    g.add_node(2, x=55.310, y=25.210)
    g.add_node(3, x=55.320, y=25.220)
    g.add_edge(1, 2, 0, length=100)
    g.add_edge(2, 3, 0, length=100)
    return g


def test_route_422_bad_coords():
    app.state.graph = _tiny_graph()
    app.state.edge_index = {}
    assert client.get("/api/route-blackspots?origin=bad&dest=25.2,55.3").status_code == 422


def test_route_happy_path():
    app.state.graph = _tiny_graph()
    app.state.edge_index = {
        "1_2_0": {"count": 60, "severe": 10, "wsum": 70, "blackspot": True},
        "2_3_0": {"count": 2, "severe": 0, "wsum": 2, "blackspot": False},
    }
    r = client.get("/api/route-blackspots?origin=25.200,55.300&dest=25.220,55.320")
    assert r.status_code == 200
    j = r.json()
    assert j["n_blackspots"] == 1
    assert len(j["geometry"]) == 3
    b = j["blackspots"][0]
    assert "lat" in b and "lng" in b
