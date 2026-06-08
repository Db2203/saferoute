"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { getStats, Stats } from "@/lib/api";

function rateColor(rate: number): string {
  if (rate >= 40) return "#f87171";
  if (rate >= 25) return "#fb923c";
  if (rate >= 15) return "#facc15";
  if (rate >= 5) return "#a3e635";
  return "#4ade80";
}

const AXIS_TICK = { fontSize: 10, fill: "#a1a1aa" } as const;
const TOOLTIP = {
  contentStyle: {
    backgroundColor: "#18181b",
    border: "1px solid #3f3f46",
    borderRadius: 6,
    fontSize: 11,
  },
  itemStyle: { color: "#fafafa" },
  labelStyle: { color: "#a1a1aa" },
} as const;

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-lg bg-zinc-800 px-3 py-2">
      <div className="text-xl font-semibold tabular-nums text-white">{value}</div>
      <div className="text-[11px] leading-tight text-zinc-400">{label}</div>
    </div>
  );
}

function Section({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="mt-5">
      <h2 className="text-sm font-semibold text-zinc-100">{title}</h2>
      {sub && <p className="mb-1 text-[11px] text-zinc-400">{sub}</p>}
      {children}
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getStats().then(setStats).catch(() => setError(true));
  }, []);

  if (error) {
    return <p className="mt-4 text-xs text-amber-400">Couldn&apos;t load stats — is the backend running?</p>;
  }
  if (!stats) {
    return <p className="mt-4 text-xs text-zinc-500">Loading stats…</p>;
  }

  const s = stats.summary;
  const years = `${s.date_from.slice(0, 4)}–${s.date_to.slice(0, 4)}`;
  const topTypes = stats.severe_rate_by_type.filter((t) => t.type_en).slice(0, 10);
  const yearly = stats.yearly.filter((y) => y.severe_rate_pct !== null);

  return (
    <div>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <Stat value={s.total.toLocaleString()} label="collisions" />
        <Stat value={`${s.severe_rate_pct}%`} label="severe" />
        <Stat value={s.n_blackspots.toLocaleString()} label="blackspots" />
        <Stat value={years} label="years of data" />
      </div>

      <Section title="What's most dangerous" sub="% of each collision type that is severe">
        <ResponsiveContainer width="100%" height={Math.max(180, topTypes.length * 26)}>
          <BarChart data={topTypes} layout="vertical" margin={{ left: 8, right: 30 }}>
            <XAxis type="number" domain={[0, 60]} tick={AXIS_TICK} unit="%" stroke="#3f3f46" />
            <YAxis type="category" dataKey="type_en" width={120} tick={AXIS_TICK} interval={0} stroke="#3f3f46" />
            <Tooltip {...TOOLTIP} formatter={(v) => [`${v}% severe`, ""]} labelFormatter={() => ""} cursor={{ fill: "#27272a" }} />
            <Bar dataKey="severe_rate_pct" radius={[0, 3, 3, 0]} label={{ position: "right", fontSize: 10, fill: "#d4d4d8", formatter: (v) => `${v}%` }}>
              {topTypes.map((t, i) => (
                <Cell key={i} fill={rateColor(t.severe_rate_pct)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Section>

      <Section title="When are crashes most severe?" sub="severe % by hour — note the late-night rise">
        <ResponsiveContainer width="100%" height={150}>
          <BarChart data={stats.severe_pct_by_hour} margin={{ left: -16, right: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
            <XAxis dataKey="hour" tick={AXIS_TICK} interval={2} stroke="#3f3f46" />
            <YAxis tick={AXIS_TICK} unit="%" stroke="#3f3f46" />
            <Tooltip {...TOOLTIP} formatter={(v) => [`${v}% severe`, ""]} labelFormatter={(h) => `${h}:00`} cursor={{ fill: "#27272a" }} />
            <Bar dataKey="severe_rate_pct" radius={[2, 2, 0, 0]}>
              {stats.severe_pct_by_hour.map((h, i) => (
                <Cell key={i} fill={h.hour <= 5 || h.hour >= 22 ? "#fca5a5" : "#ef4444"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Section>

      <Section title="Are roads getting safer?" sub="severe rate by year">
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={yearly} margin={{ left: -16, right: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
            <XAxis dataKey="year" tick={AXIS_TICK} stroke="#3f3f46" />
            <YAxis tick={AXIS_TICK} unit="%" stroke="#3f3f46" />
            <Tooltip {...TOOLTIP} formatter={(v) => [`${v}% severe`, ""]} cursor={{ stroke: "#3f3f46" }} />
            <Line type="monotone" dataKey="severe_rate_pct" stroke="#f87171" strokeWidth={2} dot={{ r: 2, fill: "#f87171" }} />
          </LineChart>
        </ResponsiveContainer>
      </Section>
    </div>
  );
}
