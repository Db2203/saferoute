from dataclasses import dataclass

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes.routing import get_graph, get_risk_scores, get_temporal_artifact


@dataclass
class _StubArtifact:
    pass


def _graph_with_edge(maxspeed="40 mph"):
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(1, x=-0.10, y=51.50)
    g.add_node(2, x=-0.09, y=51.50)
    g.add_edge(1, 2, key=0, maxspeed=maxspeed, length=100.0)
    return g


@pytest.fixture
def client():
    app.dependency_overrides[get_graph] = _graph_with_edge
    app.dependency_overrides[get_risk_scores] = lambda: {(1, 2, 0): 42.0}
    app.dependency_overrides[get_temporal_artifact] = lambda: _StubArtifact()
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------- /api/risk

def test_risk_returns_score_and_adjusted(client, monkeypatch):
    monkeypatch.setattr("app.routes.risk.predict_risk_multiplier", lambda art, feats: 1.5)
    resp = client.get("/api/risk", params={"u": 1, "v": 2, "key": 0, "when": "2026-05-22T17:30"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_score"] == 42.0
    assert data["speed_limit"] == 40  # parsed from "40 mph"
    assert data["temporal_multiplier"] == 1.5
    assert data["adjusted_risk"] == pytest.approx(63.0)  # 42 * 1.5
    assert data["context"]["hour"] == 17
    # 2026-05-22 is a Friday; STATS19 code for Friday is 6
    assert data["context"]["day_of_week"] == 6
    assert data["context"]["month"] == 5


def test_risk_unknown_edge_returns_404(client, monkeypatch):
    monkeypatch.setattr("app.routes.risk.predict_risk_multiplier", lambda art, feats: 1.0)
    resp = client.get("/api/risk", params={"u": 99, "v": 98, "key": 0})
    assert resp.status_code == 404


def test_risk_speed_limit_fallback_when_maxspeed_missing(monkeypatch):
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(1, x=-0.1, y=51.5)
    g.add_node(2, x=-0.09, y=51.5)
    g.add_edge(1, 2, key=0, length=100.0)  # no maxspeed
    app.dependency_overrides[get_graph] = lambda: g
    app.dependency_overrides[get_risk_scores] = lambda: {(1, 2, 0): 10.0}
    app.dependency_overrides[get_temporal_artifact] = lambda: _StubArtifact()
    monkeypatch.setattr("app.routes.risk.predict_risk_multiplier", lambda art, feats: 1.0)
    try:
        resp = client_get = TestClient(app).get("/api/risk", params={"u": 1, "v": 2, "key": 0})
        assert resp.status_code == 200
        assert resp.json()["speed_limit"] == 30  # London default
    finally:
        app.dependency_overrides.clear()


# ------------------------------------------------------------ /api/temporal

def test_temporal_returns_24_hour_profile(client, monkeypatch):
    monkeypatch.setattr("app.routes.risk.predict_risk_multiplier", lambda art, feats: 1.0 + feats["hour"] / 100)
    monkeypatch.setattr("app.routes.risk.ox.distance.nearest_edges", lambda graph, X, Y: (1, 2, 0))
    resp = client.get("/api/temporal", params={"lat": 51.50, "lng": -0.095, "when": "2026-05-22T00:00"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["profile"]) == 24
    assert [p["hour"] for p in data["profile"]] == list(range(24))
    assert data["matched_edge"] == [1, 2, 0]
    assert data["speed_limit"] == 40
    assert data["day_of_week"] == 6  # Friday
    # multiplier varies with hour per our stub
    assert data["profile"][0]["temporal_multiplier"] == pytest.approx(1.0)
    assert data["profile"][23]["temporal_multiplier"] == pytest.approx(1.23)


def test_temporal_out_of_range_returns_400(client):
    resp = client.get("/api/temporal", params={"lat": 200.0, "lng": 500.0})
    assert resp.status_code == 400
