export default function Panel({
  title,
  sub,
  className = "",
  children,
}: {
  title: string;
  sub?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      className={`flex min-h-0 flex-col rounded-lg border border-zinc-800 bg-zinc-900 p-3 ${className}`}
    >
      <header className="mb-1">
        <h2 className="text-xs font-semibold tracking-tight text-zinc-100">{title}</h2>
        {sub && <p className="text-[10px] leading-tight text-zinc-500">{sub}</p>}
      </header>
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  );
}
