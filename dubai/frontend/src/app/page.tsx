"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

import {
  Analytics,
  BlackspotFeature,
  Filters,
  getAnalytics,
  getBlackspots,
  RouteResult,
} from "@/lib/api";
import { DowBars, Heatmap, HourBars, Seasonality, TypeBars, YearTrend } from "@/components/Charts";
import FilterChips from "@/components/FilterChips";
import KpiStrip from "@/components/KpiStrip";
import Panel from "@/components/Panel";
import RouteCheck from "@/components/RouteCheck";

const Map = dynamic(() => import("@/components/Map"), { ssr: false });

export default function Home() {
  const [filters, setFilters] = useState<Filters>({});
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [blackspots, setBlackspots] = useState<BlackspotFeature[]>([]);
  const [error, setError] = useState(false);
  const [route, setRoute] = useState<RouteResult | null>(null);
  const [routeOpen, setRouteOpen] = useState(false);
  const [basemapError, setBasemapError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getAnalytics(filters), getBlackspots(filters)])
      .then(([a, b]) => {
        if (cancelled) return;
        setAnalytics(a);
        setBlackspots(b.features);
        setError(false);
      })
      .catch(() => !cancelled && setError(true));
    return () => {
      cancelled = true;
    };
  }, [filters]);

  // click a value to filter by it; click the active value again to clear it
  const toggle = (key: keyof Filters, value: string | number) =>
    setFilters((f) => ({ ...f, [key]: f[key] === value ? undefined : value }));
  const clearKey = (key: keyof Filters) =>
    setFilters((f) => {
      const next = { ...f };
      delete next[key];
      return next;
    });
  const toggleCell = (hour: number, dow: string) =>
    setFilters((f) =>
      f.hour === hour && f.dow === dow
        ? { ...f, hour: undefined, dow: undefined }
        : { ...f, hour, dow },
    );

  return (
    <div className="flex h-screen w-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
        <div>
          <h1 className="text-base font-semibold tracking-tight text-white">SafeRoute Dubai</h1>
          <p className="text-[11px] text-zinc-500">
            Road-safety intelligence · Dubai Police collision data, 2018–2026
          </p>
        </div>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-xs text-zinc-300">
            <input
              type="checkbox"
              checked={filters.severity === "severe"}
              onChange={() => toggle("severity", "severe")}
              className="accent-red-500"
            />
            Severe only
          </label>
          <button
            onClick={() => setRouteOpen((o) => !o)}
            className={`rounded-md border px-3 py-1 text-xs font-medium ${
              routeOpen
                ? "border-red-500 bg-red-600 text-white"
                : "border-zinc-700 text-zinc-200 hover:bg-zinc-800"
            }`}
          >
            Route check
          </button>
        </div>
      </header>

      {error && (
        <div className="border-b border-amber-500/30 bg-amber-500/10 px-4 py-1.5 text-xs text-amber-300">
          Couldn&apos;t reach the backend — start it on :8000 and reload.
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col gap-2 p-2">
        {analytics && <KpiStrip a={analytics} nBlackspots={blackspots.length} />}
        <FilterChips filters={filters} onClear={clearKey} onClearAll={() => setFilters({})} />

        <div className="flex min-h-0 flex-1 flex-col gap-2">
          {/* top: map + the headline "what's dangerous" chart */}
          <div className="flex min-h-0 flex-[1.4] gap-2">
            <Panel
              title="Where crashes concentrate"
              sub="blackspot cells — size = crash count (filtered to your selection)"
              className="relative flex-[3]"
            >
              <div className="relative h-full overflow-hidden rounded-md">
                <Map blackspots={blackspots} route={route} onBasemapError={setBasemapError} />
                {basemapError && (
                  <div className="absolute bottom-2 left-2 z-[1000] max-w-xs rounded-md border border-amber-300 bg-amber-50 px-2 py-1 text-[11px] text-amber-800 shadow">
                    Basemap couldn&apos;t load — showing accident data only.
                  </div>
                )}
                {routeOpen && (
                  <div className="absolute right-2 top-2 z-[1000] w-72 rounded-lg border border-zinc-700 bg-zinc-900/95 p-3 shadow-xl backdrop-blur">
                    <RouteCheck onRoute={setRoute} />
                  </div>
                )}
              </div>
            </Panel>
            <Panel
              title="What's most dangerous"
              sub="% of each collision type that is severe — click to filter"
              className="flex-[2] overflow-y-auto"
            >
              {analytics && (
                <TypeBars
                  data={analytics.by_type}
                  active={filters.type}
                  onSelect={(t) => toggle("type", t)}
                />
              )}
            </Panel>
          </div>

          {/* bottom: supporting temporal panels */}
          <div className="flex h-[38%] min-h-0 gap-2">
            <Panel title="By hour of day" sub="crash count — click an hour" className="flex-1">
              {analytics && (
                <HourBars data={analytics.by_hour} active={filters.hour} onSelect={(h) => toggle("hour", h)} />
              )}
            </Panel>
            <Panel title="When, by hour × day" sub="darker = more crashes" className="flex-[1.5]">
              {analytics && (
                <Heatmap
                  data={analytics.hour_dow}
                  active={{ hour: filters.hour, dow: filters.dow }}
                  onSelect={toggleCell}
                />
              )}
            </Panel>
            <Panel title="By day of week" sub="click a day" className="flex-1">
              {analytics && (
                <DowBars data={analytics.by_dow} active={filters.dow} onSelect={(d) => toggle("dow", d)} />
              )}
            </Panel>
            <Panel title="Severe rate by year" sub="click a year" className="flex-1">
              {analytics && (
                <YearTrend data={analytics.by_year} active={filters.year} onSelect={(y) => toggle("year", y)} />
              )}
            </Panel>
            <Panel title="Seasonality" sub="crashes by month" className="flex-1">
              {analytics && <Seasonality data={analytics.by_month} />}
            </Panel>
          </div>
        </div>
      </div>
    </div>
  );
}
