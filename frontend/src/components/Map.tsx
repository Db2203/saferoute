"use client";

import L from "leaflet";
import { useEffect } from "react";
import {
  CircleMarker,
  MapContainer,
  Polyline,
  TileLayer,
  Tooltip,
  useMap,
} from "react-leaflet";

import type { RouteState } from "@/lib/api";
import HotspotLayer from "./HotspotLayer";

const LONDON_CENTER: [number, number] = [51.5074, -0.1278];
const LONDON_BOUNDS = L.latLngBounds([51.28, -0.51], [51.69, 0.33]);

// API geometry is [lng, lat][]; Leaflet wants [lat, lng][].
function toLatLng(geometry: [number, number][]): [number, number][] {
  return geometry.map(([lng, lat]) => [lat, lng]);
}

// Pans/zooms the map to the fastest route whenever a new route arrives.
function FitToRoute({ route }: { route: RouteState | null }) {
  const map = useMap();
  useEffect(() => {
    if (!route) return;
    const points = toLatLng(route.response.fastest.geometry);
    if (points.length > 0) {
      map.fitBounds(points, { padding: [48, 48] });
    }
  }, [route, map]);
  return null;
}

export default function Map({
  route,
  showHotspots,
  onHotspotError,
}: {
  route: RouteState | null;
  showHotspots: boolean;
  onHotspotError?: (message: string | null) => void;
}) {
  return (
    <MapContainer
      center={LONDON_CENTER}
      zoom={11}
      minZoom={9}
      maxBounds={LONDON_BOUNDS}
      className="h-full w-full"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {showHotspots && <HotspotLayer onError={onHotspotError} />}

      {route && (
        <>
          {/* draw safest first so the red fastest line sits on top where they overlap */}
          <Polyline
            positions={toLatLng(route.response.safest.geometry)}
            pathOptions={{ color: "#059669", weight: 5, opacity: 0.85 }}
          />
          <Polyline
            positions={toLatLng(route.response.fastest.geometry)}
            pathOptions={{ color: "#dc2626", weight: 4, opacity: 0.85 }}
          />
          <CircleMarker
            center={route.origin}
            radius={7}
            pathOptions={{ color: "#1f2937", fillColor: "#ffffff", fillOpacity: 1, weight: 2 }}
          >
            <Tooltip>Start</Tooltip>
          </CircleMarker>
          <CircleMarker
            center={route.dest}
            radius={7}
            pathOptions={{ color: "#1f2937", fillColor: "#ffffff", fillOpacity: 1, weight: 2 }}
          >
            <Tooltip>Destination</Tooltip>
          </CircleMarker>
        </>
      )}

      <FitToRoute route={route} />
    </MapContainer>
  );
}
