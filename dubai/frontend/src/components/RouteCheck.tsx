"use client";

import { useState } from "react";

import { geocode, getRouteBlackspots, RouteResult } from "@/lib/api";
import PlaceInput from "./PlaceInput";

export default function RouteCheck({ onRoute }: { onRoute: (r: RouteResult | null) => void }) {
  const [originText, setOriginText] = useState("");
  const [destText, setDestText] = useState("");
  const [originCoord, setOriginCoord] = useState<[number, number] | null>(null);
  const [destCoord, setDestCoord] = useState<[number, number] | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RouteResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function check(e: React.FormEvent) {
    e.preventDefault();
    if (!originText.trim() || !destText.trim()) {
      setError("Enter both a start and a destination.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    onRoute(null);
    try {
      // use the picked suggestion's coords; fall back to geocoding typed text
      const o = originCoord ?? (await geocode(originText));
      const d = destCoord ?? (await geocode(destText));
      if (!o || !d) {
        setError("Couldn't find one of those places in Dubai.");
      } else {
        const r = await getRouteBlackspots(o, d);
        setResult(r);
        onRoute(r);
      }
    } catch {
      setError("Routing failed — is the backend running?");
    }
    setLoading(false);
  }

  return (
    <div>
      <h2 className="text-sm font-semibold text-zinc-100">Check a route</h2>
      <p className="mb-2 text-[11px] text-zinc-400">blackspots you&apos;ll cross, origin → destination</p>
      <form onSubmit={check} className="flex flex-col gap-2">
        <PlaceInput
          placeholder="From (e.g. Dubai Marina)"
          value={originText}
          onChange={(t) => {
            setOriginText(t);
            setOriginCoord(null);
          }}
          onSelect={(p) => {
            setOriginText(p.label);
            setOriginCoord([p.lat, p.lng]);
          }}
        />
        <PlaceInput
          placeholder="To (e.g. Deira)"
          value={destText}
          onChange={(t) => {
            setDestText(t);
            setDestCoord(null);
          }}
          onSelect={(p) => {
            setDestText(p.label);
            setDestCoord([p.lat, p.lng]);
          }}
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-500 disabled:opacity-50"
        >
          {loading ? "Checking…" : "Check route"}
        </button>
      </form>

      {error && <p className="mt-2 text-xs text-amber-400">{error}</p>}

      {result && (
        <div className="mt-3 rounded-lg bg-zinc-800 px-3 py-2 text-sm text-zinc-200">
          Your route crosses{" "}
          <span className="font-semibold text-red-400">{result.n_blackspots}</span> of Dubai&apos;s
          worst blackspots.
          <p className="mt-1 text-[11px] text-zinc-400">
            (highlighted on the map; severity-weighted exposure {result.risk_exposure.toLocaleString()})
          </p>
          <button
            onClick={() => {
              setResult(null);
              onRoute(null);
            }}
            className="mt-2 text-[11px] text-zinc-400 underline hover:text-zinc-200"
          >
            clear route
          </button>
        </div>
      )}
    </div>
  );
}
