"""Load precomputed artifacts at startup. Every loader returns None if its file
is missing, so the API boots even on a fresh clone (endpoints then 503 instead
of crashing). Regenerate the files with the scripts in scripts/."""
from __future__ import annotations

import json
from pathlib import Path

from app.data.aggregates import BLACKSPOTS_GEOJSON, STATS_JSON
from app.models import severity_model
from app.models.edge_blackspots import EDGE_BLACKSPOTS_JSON
from app.models.graph import CACHE_PATH, load_dubai_graph


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_stats():
    return _read_json(STATS_JSON)


def load_blackspots():
    return _read_json(BLACKSPOTS_GEOJSON)


def load_edge_index():
    data = _read_json(EDGE_BLACKSPOTS_JSON)
    return data["edges"] if data else None


def load_graph():
    return load_dubai_graph() if CACHE_PATH.exists() else None


def load_severity():
    return severity_model.load_artifact() if severity_model.MODEL_PKL.exists() else None
