# SafeRoute

![CI](https://github.com/Db2203/saferoute/actions/workflows/ci.yml/badge.svg)

Safety-aware navigation for London. Uses UK STATS19 collision data (2020–2024) and ML to recommend driving routes that trade off speed against historical risk, side-by-side with the fastest route.

## Stack

- **Backend**: Python 3.10+, FastAPI, SQLAlchemy + GeoAlchemy2
- **Database**: PostgreSQL 16/18 + PostGIS (spatial queries)
- **ML / data**: scikit-learn (DBSCAN, Random Forest), NetworkX, OSMnx, pandas
- **Frontend**: Next.js (App Router), React, Leaflet, Tailwind
- **Data sources**: STATS19 collisions + DfT AADT traffic counts, scoped to a London bounding box

## What's working today

Backend pipeline is end-to-end:

- 503k UK collisions → cleaned and London-filtered to **123,576** rows in Parquet
- 46k DfT AADT traffic counts → London-filtered to **4,416** count points
- London driveable road graph (165k nodes, 381k edges) cached locally
- DBSCAN clusters → **1,863 hotspots** persisted to Postgres
- Risk score per OSM edge (count × severity ÷ AADT, normalized 0–100) on **64,652 scored edges** with 13% fallback rate
- Random Forest temporal predictor (severity-by-context) trained, ~48% accuracy, baseline-normalized risk multiplier exposed for routing

Routing API and frontend map UI are next.

## Setup (developer)

1. **Postgres + PostGIS** running on `localhost:5432` (native install or `docker compose up`). Database `saferoute` with PostGIS extension; user `saferoute` / password `saferoute`.
2. **Backend**:
   ```
   cd backend
   python -m venv .venv
   .venv/Scripts/pip install -r requirements.txt   # Windows
   # source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
   ```
3. **Data**:
   ```
   .venv/Scripts/python ../scripts/download_stats19.py
   .venv/Scripts/python ../scripts/download_aadt.py
   .venv/Scripts/python -m app.data.preprocessing
   .venv/Scripts/python -m scripts.init_db
   .venv/Scripts/python -m scripts.load_to_db
   .venv/Scripts/python -m scripts.build_graph
   .venv/Scripts/python -m scripts.build_hotspots
   .venv/Scripts/python -m scripts.build_risk_scores
   .venv/Scripts/python -m scripts.train_temporal
   ```
4. **API**: `.venv/Scripts/python -m uvicorn app.main:app --reload` then hit `localhost:8000/health`.
5. **Frontend**: `cd frontend && npm install && npm run dev` then open `localhost:3000`.

## Tests

```
cd backend
.venv/Scripts/python -m pytest
```

40 unit tests covering loaders, preprocessing, clustering, graph caching, risk scoring, and the temporal model.

## Status

Active development. Roadmap is staged (scaffold → data → ML → API → UI → polish). See commit history for the build log.
