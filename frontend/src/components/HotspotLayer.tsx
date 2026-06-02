"use client";

import axios from "axios";
import type { LatLngBounds } from "leaflet";
import { useEffect, useRef, useState } from "react";
import { CircleMarker, Tooltip, useMapEvents } from "react-leaflet";

import { getHotspots, type Hotspot } from "@/lib/api";

interface Props {
  onError?: (message: string | null) => void;
}

export default function HotspotLayer({ onError }: Props) {
  const [hotspots, setHotspots] = useState<Hotspot[]>([]);
  // Track the in-flight request so a new pan can cancel the previous one —
  // otherwise a slow earlier response can land after a newer one and overwrite
  // fresh data with stale.
  const inFlight = useRef<AbortController | null>(null);

  function refresh(b: LatLngBounds) {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    getHotspots(
      {
        south: b.getSouth(),
        west: b.getWest(),
        north: b.getNorth(),
        east: b.getEast(),
      },
      controller.signal,
    )
      .then((res) => {
        setHotspots(res.hotspots);
        onError?.(null);
      })
      .catch((err) => {
        // a cancelled request isn't a real failure — ignore it
        if (axios.isCancel(err)) return;
        console.warn("hotspots fetch failed", err);
        onError?.("Couldn't reach the server — is the backend running?");
      });
  }

  const map = useMapEvents({
    moveend: () => refresh(map.getBounds()),
  });

  useEffect(() => {
    refresh(map.getBounds());
    return () => inFlight.current?.abort();
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
