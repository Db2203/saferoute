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

export interface GeocodeResult {
  lat: number;
  lng: number;
  label: string; // full Nominatim display_name
  shortLabel: string; // first couple of parts, for compact display
}

// Shared between the sidebar form and the map so both render the same route.
export interface RouteState {
  response: RouteResponse;
  origin: [number, number]; // [lat, lng]
  dest: [number, number];
  originLabel: string;
  destLabel: string;
}

const client = axios.create({ baseURL: BASE_URL });

export async function getHotspots(
  bounds: Bounds,
  signal?: AbortSignal,
): Promise<HotspotsResponse> {
  const { south, west, north, east } = bounds;
  const res = await client.get<HotspotsResponse>("/api/hotspots", {
    params: { bounds: `${south},${west},${north},${east}` },
    signal,
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

// Greater London viewbox (left,top,right,bottom) so "King's Cross" resolves to
// London, not Sydney. Nominatim is OpenStreetMap's free geocoder — fine for a
// demo; production would need a key'd provider with a usage agreement.
const LONDON_VIEWBOX = "-0.51,51.69,0.33,51.28";

// Trim Nominatim's long "name, road, area, borough, city, postcode, country"
// down to the first two parts — enough to identify the place at a glance.
function shorten(displayName: string): string {
  const parts = displayName
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);
  return parts.slice(0, 2).join(", ");
}

export async function geocode(query: string): Promise<GeocodeResult | null> {
  const res = await axios.get<Array<{ lat: string; lon: string; display_name: string }>>(
    "https://nominatim.openstreetmap.org/search",
    {
      params: {
        q: query,
        format: "json",
        limit: 1,
        viewbox: LONDON_VIEWBOX,
        bounded: 1,
      },
    },
  );
  const hit = res.data?.[0];
  if (!hit) return null;
  return {
    lat: parseFloat(hit.lat),
    lng: parseFloat(hit.lon),
    label: hit.display_name,
    shortLabel: shorten(hit.display_name),
  };
}

// Up to 5 London matches for type-ahead suggestions. Accepts an AbortSignal so
// the caller can cancel an in-flight request when the user keeps typing.
export async function suggestPlaces(
  query: string,
  signal?: AbortSignal,
): Promise<GeocodeResult[]> {
  const res = await axios.get<Array<{ lat: string; lon: string; display_name: string }>>(
    "https://nominatim.openstreetmap.org/search",
    {
      params: {
        q: query,
        format: "json",
        limit: 5,
        viewbox: LONDON_VIEWBOX,
        bounded: 1,
      },
      signal,
    },
  );
  return (res.data ?? []).map((hit) => ({
    lat: parseFloat(hit.lat),
    lng: parseFloat(hit.lon),
    label: hit.display_name,
    shortLabel: shorten(hit.display_name),
  }));
}
