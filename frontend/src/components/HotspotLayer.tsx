"use client";

import type { LatLngBounds } from "leaflet";
import { useEffect, useState } from "react";
import { CircleMarker, Tooltip, useMapEvents } from "react-leaflet";

import { getHotspots, type Hotspot } from "@/lib/api";

export default function HotspotLayer() {
  const [hotspots, setHotspots] = useState<Hotspot[]>([]);

  function refresh(b: LatLngBounds) {
    getHotspots({
      south: b.getSouth(),
      west: b.getWest(),
      north: b.getNorth(),
      east: b.getEast(),
    })
      .then((res) => setHotspots(res.hotspots))
      .catch((err) => {
        console.warn("hotspots fetch failed", err);
      });
  }

  const map = useMapEvents({
    moveend: () => refresh(map.getBounds()),
  });

  useEffect(() => {
    refresh(map.getBounds());
    // map is stable from useMapEvents; safe to omit from deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      {hotspots.map((h) => (
        <CircleMarker
          key={h.cluster_id}
          center={[h.lat, h.lng]}
          radius={Math.max(4, Math.sqrt(h.accident_count) * 0.6)}
          pathOptions={{
            color: "#dc2626",
            fillColor: "#dc2626",
            fillOpacity: 0.4,
            weight: 1,
          }}
        >
          <Tooltip>
            {h.accident_count} accidents · severity{" "}
            {h.avg_severity_weight.toFixed(2)}
          </Tooltip>
        </CircleMarker>
      ))}
    </>
  );
}
