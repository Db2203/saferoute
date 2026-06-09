"use client";

import { CircleMarker, Tooltip } from "react-leaflet";

import { BlackspotFeature } from "@/lib/api";

// Renders the (already filtered) blackspot cells handed down from the page.
export default function BlackspotLayer({ features }: { features: BlackspotFeature[] }) {
  return (
    <>
      {features.map((f) => {
        const [lng, lat] = f.geometry.coordinates;
        const radius = Math.max(4, Math.sqrt(f.properties.count) * 1.1);
        return (
          <CircleMarker
            key={`${lat}-${lng}`}
            center={[lat, lng]}
            radius={radius}
            pathOptions={{ weight: 0.5, color: "#ffffff", fillColor: "#dc2626", fillOpacity: 0.55 }}
          >
            <Tooltip>
              {f.properties.dominant_type ?? "collisions"}: {f.properties.count} crashes (
              {f.properties.severe} severe)
            </Tooltip>
          </CircleMarker>
        );
      })}
    </>
  );
}
