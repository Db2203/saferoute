"use client";

import L from "leaflet";
import { MapContainer, TileLayer } from "react-leaflet";

import HotspotLayer from "./HotspotLayer";

const LONDON_CENTER: [number, number] = [51.5074, -0.1278];
const LONDON_BOUNDS = L.latLngBounds([51.28, -0.51], [51.69, 0.33]);

export default function Map() {
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
      <HotspotLayer />
    </MapContainer>
  );
}
