import type { LatLon, PolicyReplay } from "./types";

export const SURGE_HALF_LIFE = 1200;

export type TaxiState =
  | { kind: "driving"; pos: LatLon; drive: PolicyReplay["drives"][number] }
  | { kind: "waiting"; pos: LatLon }
  | { kind: "offmap"; label: string; pos: LatLon | null };

const OFFMAP_LABEL: Record<string, string> = {
  to_airport: "Driving a rider to LaGuardia",
  airport_queue: "In the LaGuardia taxi line",
  return_with_rider: "Bringing a rider back from LaGuardia",
  return_empty: "Driving back from LaGuardia, empty",
};

function interp(nodes: LatLon[], frac: number[], f: number): LatLon {
  if (f <= 0) return nodes[0];
  if (f >= 1) return nodes[nodes.length - 1];
  let i = 1;
  while (i < frac.length && frac[i] < f) i++;
  const a = nodes[i - 1];
  const b = nodes[Math.min(i, nodes.length - 1)];
  const span = frac[i] - frac[i - 1] || 1;
  const w = (f - frac[i - 1]) / span;
  return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w];
}

/** Where is the taxi, and what is it doing, at sim time t? */
export function taxiAt(r: PolicyReplay, t: number): TaxiState {
  for (const o of r.offmap) {
    if (t >= o.t0 && t < o.t1) return { kind: "offmap", label: OFFMAP_LABEL[o.kind] ?? o.kind, pos: null };
  }
  let last: LatLon | null = null;
  let lastT = -Infinity;
  for (const d of r.drives) {
    if (t >= d.t0 && t < d.t1) {
      const f = (t - d.t0) / Math.max(d.t1 - d.t0, 1e-6);
      return { kind: "driving", pos: interp(d.nodes, d.frac, f), drive: d };
    }
    if (d.t1 <= t && d.t1 > lastT) {
      lastT = d.t1;
      last = d.nodes[d.nodes.length - 1];
    }
  }
  for (const w of r.waits) {
    if (t >= w.t0 && t < w.t1) return { kind: "waiting", pos: w.at };
    if (w.t1 <= t && w.t1 > lastT) {
      lastT = w.t1;
      last = w.at;
    }
  }
  const first = r.drives[0]?.nodes[0] ?? r.waits[0]?.at ?? null;
  return { kind: "waiting", pos: last ?? first ?? [40.754, -73.985] };
}

export function statusText(s: TaxiState): string {
  if (s.kind === "offmap") return s.label;
  if (s.kind === "waiting") return "Waiting for a rider";
  switch (s.drive.kind) {
    case "to_pickup":
      return "Heading to a pickup";
    case "passenger":
      return "Rider on board";
    default:
      return "Repositioning to busier streets";
  }
}

export function faresUntil(r: PolicyReplay, t: number): { fares: number; served: number } {
  let fares = 0;
  let served = 0;
  for (const e of r.events) {
    if (e.ev === "dropoff" && e.t <= t) {
      fares += e.fare ?? 0;
      served += 1;
    }
  }
  return { fares, served };
}

export function surgeAt(r: PolicyReplay, t: number): { at: LatLon; value: number }[] {
  const byZone = new Map<number, { at: LatLon; value: number }>();
  for (const s of r.surges) {
    if (t < s.t || t > s.t + 3600) continue;
    const v = (s.peak - 1) * Math.pow(0.5, (t - s.t) / SURGE_HALF_LIFE);
    const cur = byZone.get(s.zone) ?? { at: s.at, value: 0 };
    cur.value += v;
    byZone.set(s.zone, cur);
  }
  return [...byZone.values()].filter((s) => s.value > 0.05);
}

export function clock(startHour: number, t: number): string {
  const mins = Math.floor(startHour * 60 + t / 60);
  const h = Math.floor(mins / 60) % 24;
  const m = ((mins % 60) + 60) % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}
