"use client";

import { useState } from "react";

import {
  geocode,
  getRoute,
  type GeocodeResult,
  type RouteResponse,
  type RouteState,
} from "@/lib/api";
import PlaceInput from "./PlaceInput";

function minutes(seconds: number): string {
  return `${Math.round(seconds / 60)} min`;
}

function km(meters: number): string {
  return `${(meters / 1000).toFixed(1)} km`;
}

interface Props {
  onRoute: (route: RouteState | null) => void;
}

export default function RouteComparison({ onRoute }: Props) {
  const [originText, setOriginText] = useState("King's Cross");
  const [destText, setDestText] = useState("Heathrow Airport");
  // Coordinates locked in when the user picks a suggestion; null means
  // "they typed freely, geocode the text on submit".
  const [originPlace, setOriginPlace] = useState<GeocodeResult | null>(null);
  const [destPlace, setDestPlace] = useState<GeocodeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RouteResponse | null>(null);

  async function resolve(text: string, place: GeocodeResult | null): Promise<GeocodeResult | null> {
    return place ?? (await geocode(text));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!originText.trim() || !destText.trim()) {
      setError("Enter both a start and a destination.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    onRoute(null);
    try {
      const [o, d] = await Promise.all([
        resolve(originText, originPlace),
        resolve(destText, destPlace),
      ]);
      if (!o) {
        setError(`Couldn't find "${originText}" in London.`);
        return;
      }
      if (!d) {
        setError(`Couldn't find "${destText}" in London.`);
        return;
      }
      const response = await getRoute({ origin: [o.lat, o.lng], dest: [d.lat, d.lng] });
      setResult(response);
      onRoute({
        response,
        origin: [o.lat, o.lng],
        dest: [d.lat, d.lng],
        originLabel: o.label,
        destLabel: d.label,
      });
    } catch {
      setError("Could not compute a route between those points.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <PlaceInput
          label="Start"
          placeholder="e.g. King's Cross"
          value={originText}
          onChange={(text) => {
            setOriginText(text);
            setOriginPlace(null);
          }}
          onSelect={setOriginPlace}
        />
        <PlaceInput
          label="Destination"
          placeholder="e.g. Heathrow Airport"
          value={destText}
          onChange={(text) => {
            setDestText(text);
            setDestPlace(null);
          }}
          onSelect={setDestPlace}
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-zinc-700 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
        >
          {loading ? "Finding routes…" : "Compare routes"}
        </button>
      </form>

      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          {error}
        </div>
      )}

      {result && (
        <div className="flex flex-col gap-3">
          <div className="rounded-md border-l-4 border-red-500 bg-zinc-50 p-3 dark:bg-zinc-950">
            <div className="text-xs font-semibold uppercase tracking-wide text-red-600">
              Fastest
            </div>
            <div className="mt-1 text-sm text-zinc-700 dark:text-zinc-300">
              {minutes(result.fastest.total_time_s)} · {km(result.fastest.total_distance_m)} ·
              risk {Math.round(result.fastest.total_risk)}
            </div>
          </div>

          <div className="rounded-md border-l-4 border-green-600 bg-zinc-50 p-3 dark:bg-zinc-950">
            <div className="text-xs font-semibold uppercase tracking-wide text-green-700">
              Safest
            </div>
            <div className="mt-1 text-sm text-zinc-700 dark:text-zinc-300">
              {minutes(result.safest.total_time_s)} · {km(result.safest.total_distance_m)} ·
              risk {Math.round(result.safest.total_risk)}
            </div>
          </div>

          <div className="rounded-md bg-amber-50 p-3 text-sm dark:bg-amber-950/40">
            <span className="font-semibold text-amber-700 dark:text-amber-400">
              {result.comparison.risk_reduction_pct.toFixed(0)}% less risk
            </span>{" "}
            <span className="text-zinc-600 dark:text-zinc-400">
              for {minutes(result.comparison.extra_time_s)} extra
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
