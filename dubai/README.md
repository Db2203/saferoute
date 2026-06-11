# SafeRoute Dubai

A road-safety **intelligence dashboard** for Dubai, built on 8 years of Dubai Police
traffic-incident data. It maps where crashes concentrate, models what makes a crash
severe, shows when the roads are most dangerous, and flags the blackspots along any
route.

> **Same architecture as the London SafeRoute, adapted to the data.** The London
> version does safety-aware *routing*. For Dubai, I tested that idea on the real
> data first and found it **doesn't work** — collisions blanket ~48% of road
> segments and the highway grid has no low-risk alternatives, so the "fastest" and
> "safest" routes come out essentially identical (0–1% difference). Rather than ship
> a feature the data can't support, SafeRoute Dubai focuses on what the data *does*
> support: **where, when, and what kind** of crashes are dangerous — plus a route
> check that shows the specific blackspots you'll cross.

## Features

- **Blackspot map** — 2,113 accident blackspots over Dubai (toggle all / severe-only; the severe view surfaces *different* spots — only 14 of the top-50 busiest cells are also top-50 by severe crashes).
- **"What's dangerous"** — severe-rate by collision type (pedestrian 46% / motorcycle 44% / rollover 43% → hitting a wall <1%).
- **"When"** — collisions by hour, an hour × day-of-week heatmap, monthly seasonality, and the yearly severe-rate trend (rose ~7% → 15% since 2019).
- **Cross-filtered** — click any type, hour, day, or year and every panel plus the map re-filter together.
- **Route check** — enter origin → destination; it draws the route along the actual roads and lists the blackspots it crosses.

## Key numbers

| | |
|---|---|
| Raw incidents | 720,155 (Aug 2018 – Feb 2026) |
| Usable collisions (cleaned) | 386,796 |
| Severe share | ~12% |
| Severity model | RandomForest, **ROC-AUC 0.869**, severe-recall 0.83 |
| Blackspots | 2,113 cells ≈ **half of all collisions**; top 1% of road segments = 19% of crashes |

## The data

[Dubai Pulse `dp_traffic_incidents-open`](https://www.dubaipulse.gov.ae) (Dubai Police
open data). Per-incident rows: id, timestamp, an **Arabic** description, and
coordinates. Notable quirks handled in the pipeline:

- The coordinate columns are **mislabeled** — `acci_x` is latitude, `acci_y` is longitude.
- **Type and severity live inside the Arabic description** (`… - بسيط` minor / `… - متوسط` moderate / `… - بليغ` severe), parsed out as a controlled vocabulary + a **3-level** severity.
- The controlled vocabulary **changed around 2021** (older phrasing carries a حادث "accident" prefix and says "car"; newer drops it and says "vehicle"). The pipeline merges the old/new names so each collision type is one category spanning 2018–2026 — **~22 canonical types**.
- Rollovers are coded **`تدهور`**, not `انقلاب` (which never appears); catching this recovered **~15k collisions (incl. ~6.6k severe)** that a naive keyword filter silently dropped.
- Filtered to actual collisions **within 150 m of a road** inside the Dubai bbox with valid timestamps; exact-duplicate records dropped.

The snapshot is frozen for reproducibility (data as of 2026-06-01; spans Aug 2018 – Feb 2026).

## Methodology (and why)

- **Blackspots:** a ~220 m grid counting severity-weighted collisions per cell. (DBSCAN
  was tested but chains Dubai's dense core into one giant cluster; a grid is robust and
  explainable.)
- **Severity model:** binary minor/severe RandomForest on incident type + hour + day +
  month + location, `class_weight="balanced"`, **ROC-AUC 0.869**. It's **descriptive, not
  a live predictor**: incident type dominates (~74% of importance) and is only known
  *after* a crash. Strip it out and predict from only what you'd know beforehand
  (location + time) and it collapses to **AUC ~0.62** — barely better than chance. So
  severe crashes aren't predictable in advance, which is *why* this is intelligence
  rather than prediction. (We lead with AUC / severe-recall, not raw accuracy — a lazy
  "always-minor" model scores 89% and is useless.)
- **Route check:** the route is computed on an OpenStreetMap graph (OSMnx); collisions
  are snapped to road edges (direction-agnostic), and the route's blackspot crossings
  are returned. No traffic-volume normalisation exists for Dubai, so road risk is
  honest severity-weighted collision density.

## Architecture

```
Dubai Pulse CSV ─► preprocess ─► road-proximity filter ─► collisions.parquet
                                   ├─► aggregates  ─► stats.json, blackspots.geojson
                                   ├─► severity model ─► severity_model.pkl
                                   └─► OSM graph + snap ─► dubai_graph.pkl, edge_blackspots.json
                                                              │
        FastAPI  (/api/analytics, /api/blackspots, /api/route-blackspots)  ◄┘
                                   │
        Next.js + Leaflet + MapLibre + recharts  (map, dashboard, route check)
```

- **Backend:** FastAPI, DB-light — precomputed artifacts *and* the full collision table held in memory at startup, so the dashboard filters on the fly (~35 ms/query) with no database.
- **Frontend:** Next.js (App Router) + Leaflet + MapLibre GL (vector basemap) + recharts.
- **Basemap:** MapTiler Streets when a key is set, else a **self-hosted offline Protomaps
  PMTiles** file — no tile server, no key (see `frontend/README.md` to regenerate).
- **Stack:** pandas, scikit-learn, OSMnx, NetworkX.

## Run it

**Backend** (from `dubai/backend`, with deps from `requirements.txt`):

```bash
# one-time pipeline (needs data/raw/dp_traffic_incidents.csv)
python -m app.data.preprocessing        # -> collisions.parquet
python -m scripts.filter_to_roads       # drop collisions >150m from any road
python -m app.data.aggregates           # -> stats.json, blackspots.geojson
python -m scripts.train_severity        # -> trained_models/severity_model.pkl
python -m scripts.build_graph           # -> data/cache/dubai_graph.pkl (downloads OSM)
python -m scripts.build_edge_blackspots # -> edge_blackspots.json

uvicorn app.main:app --port 8001        # 8001 so it runs alongside the London app (8000)
pytest                                  # 46 tests
```

**Frontend** (from `dubai/frontend`):

```bash
npm install
npm run dev    # http://localhost:3001  (expects backend on :8001; see .env.example)
```

## Limitations / future work

- Road risk isn't normalised by traffic volume (none is openly available for Dubai) —
  it's collision density, framed as blackspot avoidance.
- Some minor streets fall back to Arabic labels where OpenStreetMap has no `name:en`.
- The source is a continuously-updated feed; this snapshot is frozen (data as of
  2026-06-01) for reproducibility.
