"use client";

import dynamic from "next/dynamic";
import { useState } from "react";

import RouteComparison from "@/components/RouteComparison";
import type { RouteState } from "@/lib/api";

const Map = dynamic(() => import("@/components/Map"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-zinc-400">
      loading map…
    </div>
  ),
});

export default function Home() {
  const [route, setRoute] = useState<RouteState | null>(null);
  const [showHotspots, setShowHotspots] = useState(true);
  const [hotspotError, setHotspotError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-zinc-50 dark:bg-zinc-950">
      {/* Map: full-screen on mobile, leaves room for the sidebar on desktop */}
      <div className="absolute inset-0 md:left-96">
        {hotspotError && (
          <div className="absolute left-1/2 top-4 z-[1000] -translate-x-1/2 rounded-md border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-700 shadow-md dark:border-red-900 dark:bg-red-950 dark:text-red-300">
            {hotspotError}
          </div>
        )}
        <Map route={route} showHotspots={showHotspots} onHotspotError={setHotspotError} />
      </div>

      {/* Backdrop behind the drawer on mobile */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="absolute inset-0 z-[1050] bg-black/30 md:hidden"
        />
      )}

      {/* Sidebar: slide-in drawer on mobile, fixed column on desktop */}
      <aside
        className={`absolute inset-y-0 left-0 z-[1100] flex w-96 max-w-[85vw] flex-col gap-6 overflow-y-auto border-r border-zinc-200 bg-white p-6 transition-transform dark:border-zinc-800 dark:bg-zinc-900 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        } md:translate-x-0`}
      >
        <header>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            SafeRoute
          </h1>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Safety-aware navigation for London
          </p>
        </header>
        <RouteComparison onRoute={setRoute} />
        <label className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2 text-sm text-zinc-700 dark:border-zinc-800 dark:text-zinc-300">
          <span>Show accident hotspots</span>
          <input
            type="checkbox"
            checked={showHotspots}
            onChange={(e) => setShowHotspots(e.target.checked)}
            className="h-4 w-4 accent-zinc-900 dark:accent-zinc-100"
          />
        </label>
      </aside>

      {/* Mobile-only toggle for the drawer */}
      <button
        type="button"
        onClick={() => setSidebarOpen((open) => !open)}
        className="absolute right-4 top-4 z-[1200] rounded-md bg-white px-3 py-2 text-sm font-medium text-zinc-900 shadow-md md:hidden dark:bg-zinc-800 dark:text-zinc-100"
      >
        {sidebarOpen ? "Close" : "☰ Route"}
      </button>
    </main>
  );
}
