"use client";

import { useEffect, useRef, useState } from "react";

import { PlaceResult, searchPlaces } from "@/lib/api";

export default function PlaceInput({
  placeholder,
  value,
  onChange,
  onSelect,
}: {
  placeholder: string;
  value: string;
  onChange: (text: string) => void;
  onSelect: (place: PlaceResult) => void;
}) {
  const [suggestions, setSuggestions] = useState<PlaceResult[]>([]);
  const [open, setOpen] = useState(false);
  const justSelected = useRef(false);

  useEffect(() => {
    if (justSelected.current) {
      justSelected.current = false;
      return;
    }
    if (value.trim().length < 3) {
      setSuggestions([]);
      return;
    }
    // debounce — respect Nominatim's ~1 req/sec fair-use limit
    const t = setTimeout(async () => {
      try {
        const r = await searchPlaces(value);
        setSuggestions(r);
        setOpen(true);
      } catch {
        setSuggestions([]);
      }
    }, 400);
    return () => clearTimeout(t);
  }, [value]);

  const inputCls =
    "w-full rounded-md border border-zinc-700 bg-zinc-800 px-2 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-red-500 focus:outline-none";

  return (
    <div className="relative">
      <input
        className={inputCls}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => suggestions.length > 0 && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)} // let click land first
      />
      {open && suggestions.length > 0 && (
        <ul className="absolute z-[2000] mt-1 max-h-52 w-full overflow-auto rounded-md border border-zinc-700 bg-zinc-800 shadow-lg">
          {suggestions.map((s, i) => (
            <li key={i}>
              <button
                type="button"
                className="block w-full truncate px-2 py-1.5 text-left text-xs text-zinc-200 hover:bg-zinc-700"
                title={s.label}
                onClick={() => {
                  justSelected.current = true;
                  onChange(s.label);
                  onSelect(s);
                  setOpen(false);
                }}
              >
                {s.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
