import axios from "axios";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface BlackspotFeature {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] }; // [lng, lat]
  properties: {
    count: number;
    severe: number;
    wsum: number;
    dominant_type: string | null;
  };
}

export interface BlackspotsResponse {
  type: "FeatureCollection";
  features: BlackspotFeature[];
}

export async function getBlackspots(): Promise<BlackspotsResponse> {
  const { data } = await axios.get<BlackspotsResponse>(`${BASE_URL}/api/blackspots`);
  return data;
}
