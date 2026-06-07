# SafeRoute Dubai — frontend

Next.js + Leaflet map UI for the Dubai road-safety dashboard.

## Dev

```bash
npm install
npm run dev
```

Open http://localhost:3000. Expects the backend running on `http://localhost:8000`
(override with `NEXT_PUBLIC_API_BASE_URL`).

## Basemap

The basemap is a self-hosted [Protomaps](https://protomaps.com) vector tileset —
a single `.pmtiles` file served from `public/`, so there's no tile-server
dependency or API key. The file is gitignored (regeneratable). To recreate it,
download the [`pmtiles` CLI](https://github.com/protomaps/go-pmtiles/releases)
and extract the Dubai bounding box from a Protomaps daily planet build:

```bash
pmtiles extract https://build.protomaps.com/<YYYYMMDD>.pmtiles public/dubai.pmtiles \
  --bbox=54.8,24.7,55.8,25.5 --maxzoom=15
```

Labels are forced to English (`name:en`, falling back to the local name) in
`src/components/Map.tsx`. Fonts/sprites load from Protomaps' asset CDN.
