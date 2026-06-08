import networkx as nx
import osmnx as ox
from fastapi import APIRouter, HTTPException, Request

from app.models.edge_blackspots import route_blackspots

router = APIRouter(prefix="/api", tags=["route"])


def _parse_latlng(s: str) -> tuple[float, float]:
    lat, lng = s.split(",")
    return float(lat), float(lng)


def _edge_geom_coords(graph, u, v) -> list[list[float]]:
    """[[lng, lat], ...] tracing the actual road for edge u->v (oriented u->v).

    OSMnx edges carry a `geometry` LineString following the real road curve;
    falling back to a straight u->v line only where it's absent."""
    data = graph.get_edge_data(u, v)
    key = min(data, key=lambda k: data[k].get("length", 1.0))
    geom = data[key].get("geometry")
    if geom is not None:
        seg = [[float(x), float(y)] for x, y in geom.coords]
    else:
        seg = [
            [graph.nodes[u]["x"], graph.nodes[u]["y"]],
            [graph.nodes[v]["x"], graph.nodes[v]["y"]],
        ]
    ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
    if seg and abs(seg[0][0] - ux) + abs(seg[0][1] - uy) > abs(seg[-1][0] - ux) + abs(seg[-1][1] - uy):
        seg.reverse()  # stored v->u; flip to u->v
    return seg


def _route_geometry(graph, path) -> list[list[float]]:
    coords: list[list[float]] = []
    for u, v in zip(path[:-1], path[1:]):
        seg = _edge_geom_coords(graph, u, v)
        if coords and seg and coords[-1] == seg[0]:
            coords.extend(seg[1:])  # drop the shared junction point
        else:
            coords.extend(seg)
    return coords


@router.get("/route-blackspots")
def get_route_blackspots(request: Request, origin: str, dest: str):
    graph = request.app.state.graph
    edge_index = request.app.state.edge_index
    if graph is None or edge_index is None:
        raise HTTPException(status_code=503, detail="routing not available — build the graph")
    try:
        olat, olng = _parse_latlng(origin)
        dlat, dlng = _parse_latlng(dest)
    except ValueError:
        raise HTTPException(status_code=422, detail="origin and dest must be 'lat,lng'")

    o_node = ox.distance.nearest_nodes(graph, X=olng, Y=olat)
    d_node = ox.distance.nearest_nodes(graph, X=dlng, Y=dlat)
    try:
        path = nx.shortest_path(graph, o_node, d_node, weight="length")
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="no route found between those points")

    result = route_blackspots(path, edge_index)
    # place each blackspot marker on the road (midpoint of the edge geometry)
    for b in result["blackspots"]:
        seg = _edge_geom_coords(graph, b["u"], b["v"])
        mid = seg[len(seg) // 2]
        b["lng"], b["lat"] = round(mid[0], 6), round(mid[1], 6)
    result["geometry"] = _route_geometry(graph, path)
    return result
