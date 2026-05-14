import axios from "axios";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface Hotspot {
  cluster_id: number;
  lat: number;
  lng: number;
  accident_count: number;
  avg_severity_weight: number;
}

export interface HotspotsResponse {
  hotspots: Hotspot[];
}

export interface RouteSegment {
  geometry: [number, number][]; // [lng, lat][]
  total_time_s: number;
  total_distance_m: number;
  total_risk: number;
  edge_count: number;
}

export interface RouteComparison {
  extra_time_s: number;
  extra_distance_m: number;
  risk_reduction: number;
  risk_reduction_pct: number;
}

export interface RouteContext {
  hour: number;
  day_of_week: number;
  month: number;
  weather: number;
  temporal_multiplier: number;
}

export interface RouteResponse {
  fastest: RouteSegment;
  safest: RouteSegment;
  comparison: RouteComparison;
  context: RouteContext;
}

export interface Bounds {
  south: number;
  west: number;
  north: number;
  east: number;
}

export interface GetRouteParams {
  origin: [number, number]; // [lat, lng]
  dest: [number, number];
  when?: string;
  weather?: number;
}

const client = axios.create({ baseURL: BASE_URL });

export async function getHotspots(bounds: Bounds): Promise<HotspotsResponse> {
  const { south, west, north, east } = bounds;
  const res = await client.get<HotspotsResponse>("/api/hotspots", {
    params: { bounds: `${south},${west},${north},${east}` },
  });
  return res.data;
}

export async function getRoute(params: GetRouteParams): Promise<RouteResponse> {
  const { origin, dest, when, weather } = params;
  const res = await client.get<RouteResponse>("/api/route", {
    params: {
      origin: `${origin[0]},${origin[1]}`,
      dest: `${dest[0]},${dest[1]}`,
      when,
      weather,
    },
  });
  return res.data;
}
