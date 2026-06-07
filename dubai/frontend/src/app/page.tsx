"use client";

import dynamic from "next/dynamic";
import { useState } from "react";

const Map = dynamic(() => import("@/components/Map"), { ssr: false });

export default function Home() {
  const [severeOnly, setSevereOnly] = useState(false);
  const [basemapError, setBasemapError] = useState(false);

  return (
    <div className="relative h-screen w-screen">
      <Map severeOnly={severeOnly} onBasemapError={setBasemapError} />

      <div className="absolute left-4 top-4 z-[1000] rounded-lg bg-white/95 px-4 py-3 shadow-lg">
        <h1 className="text-lg font-semibold tracking-tight">SafeRoute Dubai</h1>
        <p className="mb-2 text-xs text-zinc-500">Accident blackspots</p>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={severeOnly}
            onChange={(e) => setSevereOnly(e.target.checked)}
          />
          Severe crashes only
        </label>
      </div>

      {basemapError && (
        <div className="absolute bottom-4 left-4 z-[1000] max-w-xs rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800 shadow">
          Basemap couldn&apos;t load — showing accident data only.
        </div>
      )}
    </div>
  );
}
