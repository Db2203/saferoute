"use client";

import { Filters } from "@/lib/api";

const LABELS: { key: keyof Filters; text: (v: string | number) => string }[] = [
  { key: "type", text: (v) => `Type: ${v}` },
  { key: "year", text: (v) => `Year: ${v}` },
  { key: "hour", text: (v) => `Hour: ${v}:00` },
  { key: "dow", text: (v) => `Day: ${v}` },
  { key: "severity", text: () => "Severe only" },
];

export default function FilterChips({
  filters,
  onClear,
  onClearAll,
}: {
  filters: Filters;
  onClear: (key: keyof Filters) => void;
  onClearAll: () => void;
}) {
  const active = LABELS.filter(({ key }) => filters[key] !== undefined);
  if (active.length === 0) {
    return <span className="text-[11px] text-zinc-500">Showing all collisions — click any chart to filter.</span>;
  }
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-[11px] text-zinc-500">Filtered:</span>
      {active.map(({ key, text }) => (
        <button
          key={key}
          onClick={() => onClear(key)}
          className="flex items-center gap-1 rounded-full border border-blue-500/40 bg-blue-500/10 px-2 py-0.5 text-[11px] text-blue-300 hover:bg-blue-500/20"
        >
          {text(filters[key] as string | number)}
          <span className="text-blue-400">×</span>
        </button>
      ))}
      <button onClick={onClearAll} className="ml-1 text-[11px] text-zinc-400 underline hover:text-zinc-200">
        clear all
      </button>
    </div>
  );
}
