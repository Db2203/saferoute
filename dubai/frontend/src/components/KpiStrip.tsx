"use client";

import { Analytics } from "@/lib/api";

function Kpi({ value, label, accent }: { value: string; label: string; accent?: string }) {
  return (
    <div className="flex-1 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2">
      <div className={`text-lg font-semibold tabular-nums ${accent ?? "text-white"}`}>{value}</div>
      <div className="text-[10px] leading-tight text-zinc-500">{label}</div>
    </div>
  );
}

export default function KpiStrip({ a, nBlackspots }: { a: Analytics; nBlackspots: number }) {
  const s = a.summary;
  const busiest = a.by_hour.reduce((b, h) => (h.count > b.count ? h : b), a.by_hour[0]);
  const deadliest = a.by_type[0]; // by_type is sorted by severe-rate desc
  const span =
    s.date_from && s.date_to ? `${s.date_from.slice(0, 4)}–${s.date_to.slice(0, 4)}` : "—";

  return (
    <div className="flex gap-2">
      <Kpi value={s.total.toLocaleString()} label="collisions" />
      <Kpi
        value={s.severe_rate_pct === null ? "—" : `${s.severe_rate_pct}%`}
        label={`severe (${s.severe.toLocaleString()})`}
        accent="text-red-400"
      />
      <Kpi value={nBlackspots.toLocaleString()} label="blackspots shown" />
      <Kpi value={busiest ? `${busiest.hour}:00` : "—"} label="busiest hour" />
      <Kpi
        value={deadliest ? `${deadliest.severe_rate_pct ?? "—"}%` : "—"}
        label={deadliest ? `deadliest: ${deadliest.type_en}` : "deadliest type"}
        accent="text-orange-400"
      />
      <Kpi value={span} label="date range" />
    </div>
  );
}
