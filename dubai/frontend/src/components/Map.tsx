"use client";

import { useEffect } from "react";
import { MapContainer, useMap } from "react-leaflet";
import L from "leaflet";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import "@maplibre/maplibre-gl-leaflet";
import { Protocol } from "pmtiles";
import { layers, namedFlavor } from "@protomaps/basemaps";

import BlackspotLayer from "./BlackspotLayer";
import RouteLayer from "./RouteLayer";
import { RouteResult } from "@/lib/api";

const DUBAI_CENTER: [number, number] = [25.2, 55.27];
// Self-hosted Dubai basemap (single file in public/), no tile server, no key.
const PMTILES_URL = "pmtiles:///dubai.pmtiles";
const ASSETS = "https://protomaps.github.io/basemaps-assets";

// Register the pmtiles protocol once (Fast Refresh re-runs this module).
const g = globalThis as unknown as { __pmtilesAdded?: boolean };
if (!g.__pmtilesAdded) {
  maplibregl.addProtocol("pmtiles", new Protocol().tile);
  g.__pmtilesAdded = true;
}

// Protomaps' default labels are bilingual (English + Arabic) in Dubai. Force
// English, falling back to the local name only where there's no name:en. The
// Protomaps fontstack includes Arabic glyphs, so the fallback renders cleanly.
const ENGLISH_NAME: unknown = ["coalesce", ["get", "name:en"], ["get", "name"]];

maplibregl
  .setRTLTextPlugin(
    "https://unpkg.com/@mapbox/mapbox-gl-rtl-text@0.3.0/dist/mapbox-gl-rtl-text.js",
    false,
  )
  .catch(() => {
    /* already registered */
  });

function buildStyle() {
  const base = layers("protomaps", namedFlavor("light"), { lang: "en" }) as unknown as Array<
    Record<string, unknown>
  >;
  const englishLayers = base.map((l) => {
    const layout = l.layout as Record<string, unknown> | undefined;
    const tf = layout?.["text-field"];
    // Only rewrite NAME labels; leave ref/shield layers untouched.
    if (tf !== undefined && JSON.stringify(tf).includes("name")) {
      return { ...l, layout: { ...layout, "text-field": ENGLISH_NAME } };
    }
    return l;
  });
  return {
    version: 8,
    glyphs: `${ASSETS}/fonts/{fontstack}/{range}.pbf`,
    sprite: `${ASSETS}/sprites/v4/light`,
    sources: {
      protomaps: {
        type: "vector",
        url: PMTILES_URL,
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      },
    },
    layers: englishLayers,
  };
}

function VectorBasemap({ onError }: { onError: (failed: boolean) => void }) {
  const map = useMap();

  useEffect(() => {
    if (typeof window !== "undefined") {
      (window as unknown as { maplibregl: typeof maplibregl }).maplibregl = maplibregl;
    }
    // With a MapTiler key, use their polished Streets style (already
    // English-first). Without one, fall back to the self-hosted offline PMTiles.
    const key = process.env.NEXT_PUBLIC_MAPTILER_KEY;
    const style = key
      ? `https://api.maptiler.com/maps/streets-v2/style.json?key=${key}`
      : buildStyle();
    const attribution = key
      ? '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
    const gl = (
      L as unknown as {
        maplibreGL: (o: { style: unknown; attribution?: string }) => L.Layer;
      }
    ).maplibreGL({ style, attribution });
    gl.addTo(map);

    // Graceful degradation: blackspots are a separate Leaflet layer and render
    // regardless, so the data still shows even if the basemap fails to load.
    const glMap = (gl as unknown as { getMaplibreMap: () => maplibregl.Map }).getMaplibreMap();
    let errors = 0;
    const onGlError = () => {
      errors += 1;
      if (errors >= 4) onError(true);
    };
    glMap.on("error", onGlError);

    return () => {
      glMap.off("error", onGlError);
      map.removeLayer(gl);
    };
  }, [map, onError]);

  return null;
}

export default function Map({
  severeOnly,
  onBasemapError,
  route,
}: {
  severeOnly: boolean;
  onBasemapError: (failed: boolean) => void;
  route: RouteResult | null;
}) {
  return (
    <MapContainer
      center={DUBAI_CENTER}
      zoom={11}
      minZoom={9}
      preferCanvas
      style={{ height: "100%", width: "100%", backgroundColor: "#e5e7eb" }}
    >
      <VectorBasemap onError={onBasemapError} />
      <BlackspotLayer severeOnly={severeOnly} />
      <RouteLayer route={route} />
    </MapContainer>
  );
}
