"use client";

import dynamic from "next/dynamic";

const Map = dynamic(() => import("@/components/Map"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-zinc-400">
      loading map…
    </div>
  ),
});

export default function Home() {
  return (
    <main className="grid h-screen grid-cols-[24rem_1fr] bg-zinc-50 dark:bg-zinc-950">
      <aside className="flex flex-col gap-6 border-r border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            SafeRoute
          </h1>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Safety-aware navigation for London
          </p>
        </header>
        <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4 text-sm text-zinc-600 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400">
          Pan or zoom the map to load accident hotspots in view. Route
          comparison coming soon.
        </div>
      </aside>
      <section className="relative">
        <Map />
      </section>
    </main>
  );
}
