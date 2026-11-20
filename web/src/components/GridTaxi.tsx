"use client";

import { useEffect, useState } from "react";
import type { TaxiGrid } from "@/lib/types";

// Gymnasium Taxi-v4 map: walls sit to the right of these (row, col) cells.
const WALLS_RIGHT = [
  [0, 1],
  [1, 1],
  [3, 0],
  [4, 0],
  [3, 2],
  [4, 2],
];
const LOCS = [
  { r: 0, c: 0, name: "R" },
  { r: 0, c: 4, name: "G" },
  { r: 4, c: 0, name: "Y" },
  { r: 4, c: 3, name: "B" },
];
const ACTIONS = ["south", "north", "east", "west", "pickup", "dropoff"];
const CELL = 44;

export default function GridTaxi({ data }: { data: TaxiGrid }) {
  const [i, setI] = useState(0);
  const frames = data.episode;
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const id = setInterval(() => setI((x) => (x + 1) % (frames.length + 4)), 450);
    return () => clearInterval(id);
  }, [frames.length]);
  const f = frames[Math.min(i, frames.length - 1)];
  const dest = LOCS[f.dest];
  const inTaxi = f.passenger === 4;
  const pas = inTaxi ? null : LOCS[f.passenger];
  const step = Math.min(i, frames.length - 1);
  return (
    <div className="flex flex-col items-start gap-2">
      <svg
        viewBox={`-2 -2 ${CELL * 5 + 4} ${CELL * 5 + 4}`}
        width={CELL * 5 + 4}
        role="img"
        aria-label="Q-learning agent solving Gymnasium Taxi-v4"
      >
        <rect x={0} y={0} width={CELL * 5} height={CELL * 5} fill="#fff" stroke="#1c2228" strokeWidth={2} />
        {Array.from({ length: 4 }, (_, k) => (
          <g key={k} stroke="#d9d7d0">
            <line x1={(k + 1) * CELL} y1={0} x2={(k + 1) * CELL} y2={CELL * 5} />
            <line x1={0} y1={(k + 1) * CELL} x2={CELL * 5} y2={(k + 1) * CELL} />
          </g>
        ))}
        {WALLS_RIGHT.map(([r, c]) => (
          <line key={`${r}${c}`} x1={(c + 1) * CELL} y1={r * CELL} x2={(c + 1) * CELL} y2={(r + 1) * CELL} stroke="#1c2228" strokeWidth={4} />
        ))}
        {LOCS.map((l) => (
          <text key={l.name} x={l.c * CELL + CELL / 2} y={l.r * CELL + CELL / 2 + 6} textAnchor="middle" fontSize={18} fill="#a8a69e" fontWeight={600}>
            {l.name}
          </text>
        ))}
        <rect x={dest.c * CELL + 4} y={dest.r * CELL + 4} width={CELL - 8} height={CELL - 8} fill="none" stroke="#0b6b4b" strokeWidth={3} rx={3} />
        {pas && <circle cx={pas.c * CELL + CELL / 2} cy={pas.r * CELL + CELL / 2} r={8} fill="#2e5aa6" />}
        <rect
          x={f.col * CELL + 7}
          y={f.row * CELL + 11}
          width={CELL - 14}
          height={CELL - 22}
          rx={4}
          fill="#e8a900"
          stroke="#1c2228"
          strokeWidth={2}
          style={{ transition: "x 300ms, y 300ms" }}
        />
        {inTaxi && <circle cx={f.col * CELL + CELL / 2} cy={f.row * CELL + CELL / 2} r={5} fill="#2e5aa6" />}
      </svg>
      <p className="text-sm tabular-nums text-[var(--ink-2)]">
        Step {step + 1} of {frames.length}: {f.action >= 0 ? ACTIONS[f.action] : "delivered"}
      </p>
    </div>
  );
}
