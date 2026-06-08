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

export interface Stats {
  summary: {
    total: number;
    severe: number;
    minor: number;
    untagged: number;
    severe_rate_pct: number;
    date_from: string;
    date_to: string;
    n_types: number;
    n_blackspots: number;
  };
  severe_rate_by_type: { type_ar: string; type_en: string | null; n: number; severe_rate_pct: number }[];
  severe_pct_by_hour: { hour: number; severe_rate_pct: number }[];
  counts_by_hour: { hour: number; count: number }[];
  counts_by_dow: { dow: string; count: number }[];
  yearly: { year: number; count: number; severe_rate_pct: number | null }[];
}

export async function getStats(): Promise<Stats> {
  const { data } = await axios.get<Stats>(`${BASE_URL}/api/stats`);
  return data;
}
