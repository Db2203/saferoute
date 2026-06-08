"use client";

import { useEffect, useState } from "react";
import { CircleMarker, Tooltip } from "react-leaflet";

import { BlackspotFeature, getBlackspots } from "@/lib/api";

export default function BlackspotLayer({ severeOnly }: { severeOnly: boolean }) {
  const [features, setFeatures] = useState<BlackspotFeature[]>([]);

  useEffect(() => {
    getBlackspots()
      .then((d) => setFeatures(d.features))
      .catch(() => console.warn("failed to load blackspots — is the backend running?"));
  }, []);

  const shown = severeOnly ? features.filter((f) => f.properties.severe > 0) : features;

  return (
    <>
      {shown.map((f, i) => {
        const [lng, lat] = f.geometry.coordinates;
        const metric = severeOnly ? f.properties.severe : f.properties.count;
        const radius = Math.max(4, Math.sqrt(metric) * 1.1);
        return (
          <CircleMarker
            key={`${lat}-${lng}-${i}`}
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
