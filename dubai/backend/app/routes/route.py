import networkx as nx
import osmnx as ox
from fastapi import APIRouter, HTTPException, Request

from app.models.edge_blackspots import route_blackspots

router = APIRouter(prefix="/api", tags=["route"])


def _parse_latlng(s: str) -> tuple[float, float]:
    lat, lng = s.split(",")
    return float(lat), float(lng)


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
    # attach a lat/lng to each blackspot (segment midpoint) for the map
    for b in result["blackspots"]:
        u, v = b["u"], b["v"]
        b["lat"] = round((graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2, 6)
        b["lng"] = round((graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2, 6)
    result["geometry"] = [[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in path]
    return result
