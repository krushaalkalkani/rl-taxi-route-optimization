"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";
import type { Evaluation, LearningCurve, TaxiGrid } from "@/lib/types";

const INK = "#1c2228";
const INK2 = "#5a6169";
const RULE = "#c9c7c0";
const tick = { fill: INK2, fontSize: 13 };

export function LearningCurveChart({ data }: { data: LearningCurve }) {
  // Skip the untrained starting point: it sits near $10/h and would flatten the interesting range.
  const rows = data.curve
    .filter((p) => p.timesteps >= 100_000)
    .map((p) => ({ steps: p.timesteps / 1000, pph: p.profit_per_hour }));
  // Round to whole millions; PPO's final rollout overshoots the target by a few thousand steps.
  const maxK = Math.max(1000, Math.round(Math.max(...rows.map((r) => r.steps)) / 1000) * 1000);
  const plotted = rows.map((r) => ({ ...r, steps: Math.min(r.steps, maxK) }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={plotted} margin={{ top: 10, right: 16, bottom: 20, left: 0 }}>
        <CartesianGrid stroke={RULE} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey="steps"
          type="number"
          domain={[0, maxK]}
          ticks={Array.from({ length: maxK / 1000 + 1 }, (_, i) => i * 1000)}
          allowDataOverflow
          tick={tick}
          tickFormatter={(v) => (v === 0 ? "0" : `${v / 1000}M`)}
          label={{ value: "Training steps (decisions)", position: "insideBottom", offset: -12, fill: INK2, fontSize: 13 }}
        />
        <YAxis tick={tick} tickFormatter={(v) => `$${v}`} width={52} domain={[66, 75]} ticks={[66, 68, 70, 72, 74]} />
        <Tooltip
          formatter={(v) => [`$${Number(v).toFixed(2)} / hour`, "RL agent"]}
          labelFormatter={(v) => `${Number(v).toLocaleString()}k steps`}
        />
        <ReferenceLine
          y={data.nearest_baseline}
          stroke="#2e5aa6"
          strokeDasharray="6 4"
          label={{ value: `Nearest rule $${data.nearest_baseline.toFixed(0)}/h`, fill: "#2e5aa6", fontSize: 13, position: "insideBottomRight" }}
        />
        <Line type="monotone" dataKey="pph" stroke="#c28d00" strokeWidth={2.5} dot={{ r: 2.5, fill: "#e8a900" }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

const POLICY_STYLE: Record<string, { name: string; color: string; dash?: string }> = {
  ppo: { name: "RL agent", color: "#c28d00" },
  smart: { name: "Hand-tuned rules", color: INK, dash: "2 3" },
  nearest: { name: "Nearest rider", color: "#2e5aa6", dash: "6 4" },
};

export function ByHourChart({ data }: { data: Evaluation }) {
  const policies = Object.keys(POLICY_STYLE).filter((p) => data.per_start_hour[p]);
  const rows = Array.from({ length: 24 }, (_, h) => {
    const r: Record<string, number> = { hour: h };
    for (const p of policies) r[p] = data.per_start_hour[p][String(h)];
    return r;
  });
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={rows} margin={{ top: 10, right: 16, bottom: 20, left: 0 }}>
        <CartesianGrid stroke={RULE} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey="hour"
          tick={tick}
          ticks={[0, 3, 6, 9, 12, 15, 18, 21]}
          tickFormatter={(h) => `${String(h).padStart(2, "0")}:00`}
          label={{ value: "Shift start time", position: "insideBottom", offset: -12, fill: INK2, fontSize: 13 }}
        />
        <YAxis tick={tick} tickFormatter={(v) => `$${v}`} width={52} domain={[50, 95]} ticks={[50, 60, 70, 80, 90]} />
        <Tooltip formatter={(v, k) => [`$${Number(v).toFixed(2)} / hour`, POLICY_STYLE[String(k)]?.name ?? k]} labelFormatter={(h) => `Shift starting ${String(h).padStart(2, "0")}:00`} />
        <Legend formatter={(k) => POLICY_STYLE[k]?.name ?? k} wrapperStyle={{ fontSize: 14, paddingTop: 16 }} verticalAlign="top" align="left" />
        {policies.map((p) => (
          <Line
            key={p}
            dataKey={p}
            type="monotone"
            stroke={POLICY_STYLE[p].color}
            strokeWidth={p === "ppo" ? 2.75 : 1.75}
            strokeDasharray={POLICY_STYLE[p].dash}
            dot={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function GridCurveChart({ data }: { data: TaxiGrid }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data.curve} margin={{ top: 10, right: 16, bottom: 20, left: 0 }}>
        <CartesianGrid stroke={RULE} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey="episode"
          tick={tick}
          label={{ value: "Training episodes", position: "insideBottom", offset: -12, fill: INK2, fontSize: 13 }}
        />
        <YAxis tick={tick} width={52} />
        <Tooltip formatter={(v) => [Number(v).toFixed(1), "Mean return"]} labelFormatter={(v) => `Episode ${v}`} />
        <ReferenceLine y={0} stroke={INK2} />
        <Line type="monotone" dataKey="mean_return" stroke={INK} strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
