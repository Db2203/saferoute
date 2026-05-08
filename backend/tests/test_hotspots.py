from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes.hotspots import get_engine


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.executed_with: dict | None = None

    def execute(self, query, params=None):
        self.executed_with = dict(params) if params else None
        return SimpleNamespace(fetchall=lambda: self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def __init__(self, rows):
        self.conn = _FakeConn(rows)

    def connect(self):
        return self.conn


@pytest.fixture
def client_factory():
    def make(rows):
        fake = _FakeEngine(rows)
        app.dependency_overrides[get_engine] = lambda: fake
        client = TestClient(app)
        return client, fake
    yield make
    app.dependency_overrides.clear()


def test_returns_hotspots_for_valid_bounds(client_factory):
    rows = [
        SimpleNamespace(cluster_id=1, lat=51.52, lng=-0.07, accident_count=1899, avg_severity_weight=1.15),
        SimpleNamespace(cluster_id=2, lat=51.49, lng=-0.10, accident_count=1248, avg_severity_weight=1.18),
    ]
    client, fake = client_factory(rows)

    resp = client.get("/api/hotspots", params={"bounds": "51.28,-0.51,51.69,0.33"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["hotspots"]) == 2
    assert data["hotspots"][0]["cluster_id"] == 1
    assert data["hotspots"][0]["accident_count"] == 1899
    # bounds are passed through to the SQL params correctly
    assert fake.conn.executed_with == {"south": 51.28, "west": -0.51, "north": 51.69, "east": 0.33}


def test_returns_empty_list_when_no_rows(client_factory):
    client, _ = client_factory([])
    resp = client.get("/api/hotspots", params={"bounds": "51.0,-0.5,51.1,-0.4"})
    assert resp.status_code == 200
    assert resp.json() == {"hotspots": []}


def test_rejects_non_numeric_bounds(client_factory):
    client, _ = client_factory([])
    resp = client.get("/api/hotspots", params={"bounds": "not,a,real,bbox"})
    assert resp.status_code == 400
    assert "four comma-separated floats" in resp.json()["detail"]


def test_rejects_wrong_bounds_count(client_factory):
    client, _ = client_factory([])
    resp = client.get("/api/hotspots", params={"bounds": "1,2,3"})
    assert resp.status_code == 400


def test_rejects_inverted_bounds(client_factory):
    client, _ = client_factory([])
    # north < south
    resp = client.get("/api/hotspots", params={"bounds": "51.7,-0.5,51.3,0.0"})
    assert resp.status_code == 400
    assert "south < north" in resp.json()["detail"]


def test_missing_bounds_param_returns_422(client_factory):
    client, _ = client_factory([])
    resp = client.get("/api/hotspots")
    assert resp.status_code == 422
