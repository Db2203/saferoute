# SafeRoute

Safety-aware navigation. Uses UK STATS19 accident data and a few ML models to recommend routes that trade off speed against historical risk, side-by-side with the fastest route.

## Stack

- Backend: Python, FastAPI, scikit-learn, NetworkX, OSMnx
- Database: PostgreSQL + PostGIS (via Docker)
- Frontend: Next.js, React, Leaflet, Tailwind
- Data: STATS19 (2019-2023) + DfT AADT traffic counts, scoped to London

## Setup

Setup instructions land here once the data pipeline and services are wired up. For now this is a fresh scaffold.

## Status

Early stages. Tracking progress through a staged build (scaffold → data → ML → API → UI).
