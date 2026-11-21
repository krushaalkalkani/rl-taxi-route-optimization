"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import type { Network, PolicyReplay, Replay as ReplayData } from "@/lib/types";
import { clock, faresUntil, statusText, taxiAt } from "@/lib/replay";

const CityMap = dynamic(() => import("./CityMap"), {
  ssr: false,
  loading: () => <div className="h-full w-full animate-pulse bg-[var(--pavement-2)]" />,
});

const SCENARIOS = [
  { id: "pm-rush", name: "Evening rush", note: "Starts 17:00 — heavy traffic, airport runs are a trap" },
  { id: "am-rush", name: "Morning commute", note: "Starts 07:00 — riders pour out of Penn & Grand Central" },
  { id: "night", name: "Late night", note: "Starts 22:00 — clear roads, theatre crowds, surge spikes" },
];
const SPEEDS = [60, 180, 480];

const DRIVERS = [
  { key: "ppo" as const, name: "RL agent", sub: "MaskablePPO", color: "var(--medallion)", hex: "#e8a900" },
  { key: "nearest" as const, name: "Nearest-rider rule", sub: "Greedy baseline", color: "var(--signal)", hex: "#2e5aa6" },
];

function Panel({
  network,
  replay,
  t,
  driver,
  ended,
}: {
  network: Network;
  replay: PolicyReplay;
  t: number;
  driver: (typeof DRIVERS)[number];
  ended: boolean;
}) {
  const s = taxiAt(replay, t);
  const { fares, served } = faresUntil(replay, t);
  return (
    <figure className="flex min-w-0 flex-col">
      <figcaption className="flex items-end justify-between gap-3 pb-2">
        <div className="flex items-center gap-2">
          <span
            aria-hidden
            className="inline-block h-3.5 w-3.5 rounded-full border-2 border-[var(--ink)]"
            style={{ background: driver.color }}
          />
          <span className="font-display text-xl font-semibold leading-none">{driver.name}</span>
          <span className="text-sm text-[var(--ink-2)]">{driver.sub}</span>
        </div>
        <div className="text-right tabular-nums">
          <span className="font-display text-2xl font-semibold leading-none">${fares.toFixed(0)}</span>
          <span className="ml-1 text-sm text-[var(--ink-2)]">fares, {served} {served === 1 ? "ride" : "rides"}</span>
        </div>
      </figcaption>
      <div className="relative h-[340px] overflow-hidden rounded-md border border-[var(--rule)] sm:h-[440px]">
        <CityMap network={network} replay={replay} t={t} color={driver.hex} label={driver.name} />
        <div className="pointer-events-none absolute left-2 top-2 z-[500] rounded bg-[var(--ink)] px-2 py-1 text-sm text-white">
          {ended ? `Shift over: $${replay.stats.profit.toFixed(0)} profit` : statusText(s)}
        </div>
      </div>
    </figure>
  );
}

export default function Replay() {
  const [network, setNetwork] = useState<Network | null>(null);
  useEffect(() => {
    fetch("/data/network.json")
      .then((r) => r.json())
      .then(setNetwork);
  }, []);
  const [scenario, setScenario] = useState(SCENARIOS[0].id);
  const [data, setData] = useState<ReplayData | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(SPEEDS[1]);
  const raf = useRef<number>(0);
  const last = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;
    setData(null);
    fetch(`/data/replay-${scenario}.json`)
      .then((r) => r.json())
      .then((d: ReplayData) => {
        if (!alive) return;
        setData(d);
        setT(0);
        setPlaying(true);
      });
    return () => {
      alive = false;
    };
  }, [scenario]);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) setPlaying(false);
  }, []);

  const end = data ? Math.max(data.ppo.shift_s, data.ppo.end_t, data.nearest.end_t) : 1;

  const tick = useCallback(
    (now: number) => {
      if (last.current != null) {
        const dt = ((now - last.current) / 1000) * speed;
        setT((x) => {
          const nx = x + dt;
          if (nx >= end) {
            setPlaying(false);
            return end;
          }
          return nx;
        });
      }
      last.current = now;
      raf.current = requestAnimationFrame(tick);
    },
    [speed, end],
  );

  useEffect(() => {
    if (!playing || !data) return;
    last.current = null;
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, data, tick]);

  const ended = t >= end - 1;
  const sc = SCENARIOS.find((s) => s.id === scenario)!;

  return (
    <section aria-label="Shift replay" className="mt-8">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-y border-[var(--rule)] py-3">
        <div role="tablist" aria-label="Scenario" className="flex gap-1">
          {SCENARIOS.map((s) => (
            <button
              key={s.id}
              role="tab"
              aria-selected={s.id === scenario}
              onClick={() => setScenario(s.id)}
              className={`rounded px-3 py-1.5 font-display text-lg font-medium transition-colors ${
                s.id === scenario ? "bg-[var(--ink)] text-white" : "hover:bg-[var(--pavement-2)]"
              }`}
            >
              {s.name}
            </button>
          ))}
        </div>
        <p className="text-sm text-[var(--ink-2)]">{sc.note}</p>
      </div>

      <div className="flex flex-wrap items-center gap-4 py-3">
        <button
          onClick={() => {
            if (ended) setT(0);
            setPlaying((p) => !p || ended);
          }}
          className="min-w-24 rounded bg-[var(--medallion)] px-4 py-1.5 font-display text-lg font-semibold text-[var(--ink)] hover:brightness-95"
        >
          {ended ? "Replay" : playing ? "Pause" : "Play"}
        </button>
        <div className="font-display text-3xl font-semibold tabular-nums" aria-live="off">
          {data ? clock(data.ppo.start_hour, t) : "--:--"}
        </div>
        <input
          type="range"
          min={0}
          max={end}
          step={10}
          value={t}
          onChange={(e) => {
            setT(Number(e.target.value));
          }}
          aria-label="Shift time"
          className="h-1 min-w-40 flex-1 accent-[var(--ink)]"
        />
        <label className="flex items-center gap-2 text-sm text-[var(--ink-2)]">
          Speed
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="rounded border border-[var(--rule)] bg-white px-2 py-1 text-[var(--ink)]"
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>
      </div>

      {data && network ? (
        <div className="grid gap-6 lg:grid-cols-2">
          {DRIVERS.map((d) => (
            <Panel key={d.key} network={network} replay={data[d.key]} t={t} driver={d} ended={ended} />
          ))}
        </div>
      ) : (
        <div className="grid h-[480px] place-items-center text-[var(--ink-2)]">Loading shift…</div>
      )}

      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-sm text-[var(--ink-2)]">
        <Legend swatch={<span className="inline-block h-1 w-6 rounded bg-[#96968e]" />} text="Free-flowing street" />
        <Legend swatch={<span className="inline-block h-1 w-6 rounded bg-[#d69628]" />} text="Slowed by traffic" />
        <Legend swatch={<span className="inline-block h-1 w-6 rounded bg-[#c2362b]" />} text="Heavy congestion" />
        <Legend swatch={<span className="inline-block h-3 w-3 rounded-full border-2 border-[var(--ink)] bg-white" />} text="Rider waiting" />
        <Legend swatch={<span className="inline-block h-3 w-3 rounded-full border-2 border-[var(--signal)] bg-white" />} text="Rider for LaGuardia" />
        <Legend swatch={<span className="inline-block h-3 w-3 rounded-full bg-[#c2362b]/30" />} text="Surge zone" />
        <Legend swatch={<span className="inline-block h-0 w-6 border-t-2 border-dashed border-[var(--ink-2)]" />} text="Driving empty" />
      </div>

      {data && ended && <Outcome data={data} />}
    </section>
  );
}

function Legend({ swatch, text }: { swatch: React.ReactNode; text: string }) {
  return (
    <span className="flex items-center gap-2">
      {swatch}
      {text}
    </span>
  );
}

function Outcome({ data }: { data: ReplayData }) {
  const a = data.ppo.stats;
  const b = data.nearest.stats;
  const rows: [string, (s: typeof a) => string][] = [
    ["Profit per hour", (s) => `$${s.profit_per_hour.toFixed(2)}`],
    ["Rides completed", (s) => `${s.served}`],
    ["Airport runs", (s) => `${s.airport_trips}`],
    ["Surge-priced rides", (s) => `${s.surge_trips}`],
    ["Share of km driven empty", (s) => `${Math.round(s.empty_ratio * 100)}%`],
  ];
  return (
    <table className="mt-6 w-full max-w-xl text-left tabular-nums">
      <caption className="pb-2 text-left font-display text-xl font-semibold">How this shift ended</caption>
      <thead>
        <tr className="border-b border-[var(--ink)] text-sm text-[var(--ink-2)]">
          <th className="py-1 font-normal" />
          <th className="py-1 font-normal">RL agent</th>
          <th className="py-1 font-normal">Nearest rule</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([k, f]) => (
          <tr key={k} className="border-b border-[var(--rule)]">
            <td className="py-1.5">{k}</td>
            <td className="py-1.5 font-medium">{f(a)}</td>
            <td className="py-1.5">{f(b)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
