"use client";

import { CircleMarker, Polyline, Tooltip } from "react-leaflet";

import { RouteResult } from "@/lib/api";

export default function RouteLayer({ route }: { route: RouteResult | null }) {
  if (!route) return null;

  // backend geometry is [lng, lat]; Leaflet wants [lat, lng]
  const line = route.geometry.map(([lng, lat]) => [lat, lng] as [number, number]);

  return (
    <>
      <Polyline positions={line} pathOptions={{ color: "#2563eb", weight: 5, opacity: 0.85 }} />
      {route.blackspots.map((b, i) => (
        <CircleMarker
          key={i}
          center={[b.lat, b.lng]}
          radius={6}
          pathOptions={{ color: "#ffffff", weight: 2, fillColor: "#dc2626", fillOpacity: 0.95 }}
        >
          <Tooltip>
            blackspot on route: {b.count} crashes ({b.severe} severe)
          </Tooltip>
        </CircleMarker>
      ))}
    </>
  );
}
