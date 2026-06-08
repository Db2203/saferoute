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

export interface RouteBlackspot {
  u: number;
  v: number;
  count: number;
  severe: number;
  wsum: number;
  lat: number;
  lng: number;
}

export interface RouteResult {
  n_blackspots: number;
  risk_exposure: number;
  blackspots: RouteBlackspot[];
  geometry: [number, number][]; // [lng, lat]
}

export async function getRouteBlackspots(
  origin: [number, number], // [lat, lng]
  dest: [number, number],
): Promise<RouteResult> {
  const { data } = await axios.get<RouteResult>(`${BASE_URL}/api/route-blackspots`, {
    params: { origin: `${origin[0]},${origin[1]}`, dest: `${dest[0]},${dest[1]}` },
  });
  return data;
}

const NOMINATIM = "https://nominatim.openstreetmap.org/search";
const DUBAI_VIEWBOX = "54.9,25.4,55.6,24.9";

export interface PlaceResult {
  label: string;
  lat: number;
  lng: number;
}

// Autocomplete: up to 5 Dubai place matches for a partial query.
export async function searchPlaces(query: string): Promise<PlaceResult[]> {
  const { data } = await axios.get<{ display_name: string; lat: string; lon: string }[]>(NOMINATIM, {
    params: {
      q: query,
      format: "json",
      limit: 5,
      countrycodes: "ae",
      viewbox: DUBAI_VIEWBOX,
      bounded: 1,
    },
  });
  return data.map((d) => ({ label: d.display_name, lat: parseFloat(d.lat), lng: parseFloat(d.lon) }));
}

// Geocode a place name to [lat, lng] (first Dubai match), via OpenStreetMap Nominatim.
export async function geocode(query: string): Promise<[number, number] | null> {
  const r = await searchPlaces(query);
  return r.length ? [r[0].lat, r[0].lng] : null;
}
