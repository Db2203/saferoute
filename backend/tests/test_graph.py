import pickle

import networkx as nx

from app.models.graph import build_london_graph, load_london_graph


def _tiny_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node(1, x=-0.10, y=51.50)
    g.add_node(2, x=-0.11, y=51.51)
    g.add_edge(1, 2, length=100, highway="primary")
    return g


def _write_pickle(path, graph):
    with path.open("wb") as f:
        pickle.dump(graph, f)


def test_load_returns_pickled_graph(tmp_path):
    cache = tmp_path / "g.pkl"
    _write_pickle(cache, _tiny_graph())

    loaded = load_london_graph(cache)
    assert loaded.number_of_nodes() == 2
    assert loaded.number_of_edges() == 1


def test_build_skips_when_cache_exists(tmp_path, monkeypatch):
    cache = tmp_path / "g.pkl"
    _write_pickle(cache, _tiny_graph())

    calls = []
    monkeypatch.setattr("osmnx.graph_from_bbox", lambda *a, **kw: calls.append(kw) or nx.MultiDiGraph())

    graph = build_london_graph(cache_path=cache, force=False)
    assert calls == []
    assert graph.number_of_nodes() == 2


def test_build_rebuilds_when_force(tmp_path, monkeypatch):
    cache = tmp_path / "g.pkl"
    _write_pickle(cache, _tiny_graph())

    fresh = nx.MultiDiGraph()
    fresh.add_node(99)
    calls = []
    monkeypatch.setattr("osmnx.graph_from_bbox", lambda *a, **kw: (calls.append(kw), fresh)[1])

    graph = build_london_graph(cache_path=cache, force=True)
    assert len(calls) == 1
    assert list(graph.nodes()) == [99]


def test_build_creates_cache_when_missing(tmp_path, monkeypatch):
    cache = tmp_path / "subdir" / "g.pkl"
    monkeypatch.setattr("osmnx.graph_from_bbox", lambda *a, **kw: _tiny_graph())

    graph = build_london_graph(cache_path=cache, force=False)
    assert cache.exists()
    assert graph.number_of_nodes() == 2
