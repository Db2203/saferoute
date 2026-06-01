"use client";

import { useEffect, useRef, useState } from "react";

import { suggestPlaces, type GeocodeResult } from "@/lib/api";

interface Props {
  label: string;
  placeholder?: string;
  value: string;
  onChange: (text: string) => void;
  onSelect: (place: GeocodeResult) => void;
}

export default function PlaceInput({ label, placeholder, value, onChange, onSelect }: Props) {
  const [suggestions, setSuggestions] = useState<GeocodeResult[]>([]);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(-1);
  // Skip the next search after a programmatic select (we just set the text).
  const justSelected = useRef(false);

  useEffect(() => {
    if (justSelected.current) {
      justSelected.current = false;
      return;
    }
    const q = value.trim();
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      if (q.length < 3) {
        setSuggestions([]);
        setOpen(false);
        return;
      }
      try {
        const results = await suggestPlaces(q, controller.signal);
        setSuggestions(results);
        setOpen(results.length > 0);
        setHighlight(-1);
      } catch {
        // aborted (user kept typing) or network error — ignore
      }
    }, 350);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [value]);

  function choose(place: GeocodeResult) {
    justSelected.current = true;
    onChange(place.shortLabel);
    onSelect(place);
    setOpen(false);
    setSuggestions([]);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (!open || suggestions.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlight((h) => Math.min(h + 1, suggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlight((h) => Math.max(h - 1, 0));
    } else if (e.key === "Enter") {
      if (highlight >= 0) {
        e.preventDefault();
        choose(suggestions[highlight]);
      }
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <label className="relative flex flex-col gap-1 text-sm">
      <span className="text-zinc-600 dark:text-zinc-400">{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => suggestions.length > 0 && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
        placeholder={placeholder}
        autoComplete="off"
        className="rounded-md border border-zinc-300 bg-white px-3 py-2 text-zinc-900 outline-none focus:border-zinc-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
      />
      {open && (
        <ul className="absolute top-full z-[1000] mt-1 max-h-60 w-full overflow-auto rounded-md border border-zinc-300 bg-white shadow-lg dark:border-zinc-700 dark:bg-zinc-900">
          {suggestions.map((s, i) => (
            <li key={`${s.lat},${s.lng}`}>
              <button
                type="button"
                // mousedown fires before blur — prevent default so the click registers
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => choose(s)}
                className={`block w-full truncate px-3 py-2 text-left text-sm text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800 ${
                  i === highlight ? "bg-zinc-100 dark:bg-zinc-800" : ""
                }`}
                title={s.label}
              >
                {s.shortLabel}
              </button>
            </li>
          ))}
        </ul>
      )}
    </label>
  );
}
