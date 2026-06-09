"use client";

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

import { Analytics } from "@/lib/api";

export function rateColor(rate: number | null): string {
  if (rate === null) return "#52525b";
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

// recharts passes a state object on click; we only need activeLabel.
type ChartClick = { activeLabel?: string | number };
const labelOf = (s: unknown) =>
  s && typeof s === "object" ? (s as ChartClick).activeLabel : undefined;

const SELECTED = "#3b82f6"; // a clicked bar highlights blue
const num = (n: number) => n.toLocaleString();

export function TypeBars({
  data,
  active,
  onSelect,
}: {
  data: Analytics["by_type"];
  active?: string;
  onSelect: (type: string) => void;
}) {
  const rows = data; // backend already caps the list; show all, panel scrolls
  return (
    <ResponsiveContainer width="100%" height={Math.max(200, rows.length * 24)}>
      <BarChart
        data={rows}
        layout="vertical"
        margin={{ left: 8, right: 34 }}
        onClick={(s) => { const l = labelOf(s); if (l != null) onSelect(String(l)); }}
      >
        <XAxis type="number" domain={[0, 100]} tick={AXIS_TICK} unit="%" stroke="#3f3f46" />
        <YAxis type="category" dataKey="type_en" width={118} tick={AXIS_TICK} interval={0} stroke="#3f3f46" />
        <Tooltip {...TOOLTIP} formatter={(v) => [`${v}% severe`, ""]} labelFormatter={(l) => String(l)} cursor={{ fill: "#27272a" }} />
        <Bar dataKey="severe_rate_pct" radius={[0, 3, 3, 0]} className="cursor-pointer" label={{ position: "right", fontSize: 10, fill: "#d4d4d8", formatter: (v) => `${v}%` }}>
          {rows.map((t) => (
            <Cell key={t.type_en} fill={active === t.type_en ? SELECTED : rateColor(t.severe_rate_pct)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function HourBars({
  data,
  active,
  onSelect,
}: {
  data: Analytics["by_hour"];
  active?: number;
  onSelect: (hour: number) => void;
}) {
  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart
        data={data}
        margin={{ left: -16, right: 8 }}
        onClick={(s) => { const l = labelOf(s); if (l != null) onSelect(Number(l)); }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
        <XAxis dataKey="hour" tick={AXIS_TICK} interval={2} stroke="#3f3f46" />
        <YAxis tick={AXIS_TICK} stroke="#3f3f46" />
        <Tooltip {...TOOLTIP} formatter={(v) => [`${num(v as number)} crashes`, ""]} labelFormatter={(h) => `${h}:00`} cursor={{ fill: "#27272a" }} />
        <Bar dataKey="count" radius={[2, 2, 0, 0]} className="cursor-pointer">
          {data.map((h) => (
            <Cell key={h.hour} fill={active === h.hour ? SELECTED : "#ef4444"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DowBars({
  data,
  active,
  onSelect,
}: {
  data: Analytics["by_dow"];
  active?: string;
  onSelect: (dow: string) => void;
}) {
  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart
        data={data}
        margin={{ left: -16, right: 8 }}
        onClick={(s) => { const l = labelOf(s); if (l != null) onSelect(String(l)); }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
        <XAxis dataKey="dow" tick={AXIS_TICK} stroke="#3f3f46" />
        <YAxis tick={AXIS_TICK} stroke="#3f3f46" />
        <Tooltip {...TOOLTIP} formatter={(v) => [`${num(v as number)} crashes`, ""]} cursor={{ fill: "#27272a" }} />
        <Bar dataKey="count" radius={[2, 2, 0, 0]} className="cursor-pointer">
          {data.map((d) => (
            <Cell key={d.dow} fill={active === d.dow ? SELECTED : "#fb923c"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function Seasonality({ data }: { data: Analytics["by_month"] }) {
  const rows = data.map((m) => ({ ...m, name: MONTHS[m.month - 1] }));
  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart data={rows} margin={{ left: -16, right: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
        <XAxis dataKey="name" tick={AXIS_TICK} interval={0} stroke="#3f3f46" />
        <YAxis tick={AXIS_TICK} stroke="#3f3f46" />
        <Tooltip {...TOOLTIP} formatter={(v) => [`${num(v as number)} crashes`, ""]} cursor={{ fill: "#27272a" }} />
        <Bar dataKey="count" radius={[2, 2, 0, 0]} fill="#a78bfa" />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function YearTrend({
  data,
  active,
  onSelect,
}: {
  data: Analytics["by_year"];
  active?: number;
  onSelect: (year: number) => void;
}) {
  return (
    <ResponsiveContainer width="100%" height={150}>
      <LineChart
        data={data}
        margin={{ left: -16, right: 12 }}
        onClick={(s) => { const l = labelOf(s); if (l != null) onSelect(Number(l)); }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
        <XAxis dataKey="year" tick={AXIS_TICK} stroke="#3f3f46" />
        <YAxis tick={AXIS_TICK} unit="%" stroke="#3f3f46" />
        <Tooltip {...TOOLTIP} formatter={(v) => [`${v}% severe`, ""]} cursor={{ stroke: "#3f3f46" }} />
        <Line
          type="monotone"
          dataKey="severe_rate_pct"
          stroke="#f87171"
          strokeWidth={2}
          className="cursor-pointer"
          dot={(props: { cx?: number; cy?: number; payload?: { year: number } }) => {
            const on = props.payload?.year === active;
            return (
              <circle
                key={props.payload?.year}
                cx={props.cx}
                cy={props.cy}
                r={on ? 4 : 2}
                fill={on ? SELECTED : "#f87171"}
              />
            );
          }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// Hour (cols) x day-of-week (rows) collision-count grid. Click a cell to filter
// to that hour+day. recharts has no heatmap, so this is a plain CSS grid.
const DOWS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export function Heatmap({
  data,
  active,
  onSelect,
}: {
  data: Analytics["hour_dow"];
  active: { hour?: number; dow?: string };
  onSelect: (hour: number, dow: string) => void;
}) {
  const max = data.reduce((m, c) => Math.max(m, c.count), 1);
  const byKey = new Map(data.map((c) => [`${c.dow}-${c.hour}`, c.count]));
  return (
    <div className="flex h-full flex-col justify-center text-[9px] text-zinc-500">
      {DOWS.map((dow) => (
        <div key={dow} className="flex items-center gap-1">
          <span className="w-6 shrink-0 text-right">{dow}</span>
          <div className="grid flex-1 gap-px" style={{ gridTemplateColumns: "repeat(24, 1fr)" }}>
            {Array.from({ length: 24 }, (_, hour) => {
              const c = byKey.get(`${dow}-${hour}`) ?? 0;
              const on = active.hour === hour && active.dow === dow;
              return (
                <button
                  key={hour}
                  title={`${dow} ${hour}:00 — ${num(c)} crashes`}
                  onClick={() => onSelect(hour, dow)}
                  className={`aspect-square rounded-[1px] ${on ? "ring-1 ring-blue-400" : ""}`}
                  style={{ backgroundColor: `rgba(239,68,68,${0.08 + (c / max) * 0.92})` }}
                />
              );
            })}
          </div>
        </div>
      ))}
      <div className="ml-7 mt-1 flex justify-between text-[8px] text-zinc-600">
        <span>0:00</span>
        <span>6:00</span>
        <span>12:00</span>
        <span>18:00</span>
        <span>23:00</span>
      </div>
    </div>
  );
}
