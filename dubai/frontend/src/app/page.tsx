"use client";

import dynamic from "next/dynamic";
import { useState } from "react";

import Dashboard from "@/components/Dashboard";
import RouteCheck from "@/components/RouteCheck";
import { RouteResult } from "@/lib/api";

const Map = dynamic(() => import("@/components/Map"), { ssr: false });

export default function Home() {
  const [severeOnly, setSevereOnly] = useState(false);
  const [basemapError, setBasemapError] = useState(false);
  const [route, setRoute] = useState<RouteResult | null>(null);

  return (
    <div className="flex h-screen w-screen">
      <aside className="flex w-[360px] shrink-0 flex-col overflow-y-auto border-r border-zinc-800 bg-zinc-900 p-4 text-zinc-100">
        <h1 className="text-lg font-semibold tracking-tight text-white">SafeRoute Dubai</h1>
        <p className="text-xs text-zinc-400">
          Road-safety intelligence · Dubai Police data, 2018–2026
        </p>
        <label className="mt-3 flex items-center gap-2 text-sm text-zinc-200">
          <input
            type="checkbox"
            checked={severeOnly}
            onChange={(e) => setSevereOnly(e.target.checked)}
            className="accent-red-500"
          />
          Show severe crashes only
        </label>

        <RouteCheck onRoute={setRoute} />
        <Dashboard />
      </aside>

      <main className="relative flex-1">
        <Map severeOnly={severeOnly} onBasemapError={setBasemapError} route={route} />
        {basemapError && (
          <div className="absolute bottom-4 left-4 z-[1000] max-w-xs rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800 shadow">
            Basemap couldn&apos;t load — showing accident data only.
          </div>
        )}
      </main>
    </div>
  );
}
