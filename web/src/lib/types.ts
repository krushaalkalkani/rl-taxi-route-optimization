export type LatLon = [number, number];

export interface Network {
  name: string;
  n_nodes: number;
  n_edges: number;
  bounds: [LatLon, LatLon];
  edges: LatLon[][];
  edge_zone: number[];
  zones: { id: number; lat: number; lon: number }[];
  hotspots: { name: string; lat: number; lon: number; kind: string }[];
  congestion: number[][]; // [zone][hour]
  traffic_source: string;
  demand_by_hour: number[];
}

export interface Drive {
  kind: "to_pickup" | "passenger" | "reposition";
  rid: number | null;
  t0: number;
  t1: number;
  nodes: LatLon[];
  frac: number[];
  geom: LatLon[];
}

export interface Rider {
  rid: number;
  t: number;
  expires: number;
  pickup: LatLon;
  dropoff: LatLon;
  status: "missed" | "picked" | "served" | "cancelled";
  airport: boolean;
  t_pickup?: number;
  t_dropoff?: number;
  fare?: number;
}

export interface EpisodeStats {
  revenue: number;
  profit: number;
  profit_per_hour: number;
  served: number;
  cancelled: number;
  requests_seen: number;
  km_total: number;
  km_empty: number;
  empty_ratio: number;
  utilization: number;
  airport_trips: number;
  surge_trips: number;
  hours: number;
}

export interface PolicyReplay {
  seed: number;
  start_hour: number;
  shift_s: number;
  end_t: number;
  stats: EpisodeStats;
  drives: Drive[];
  waits: { t0: number; t1: number; at: LatLon }[];
  offmap: { kind: string; t0: number; t1: number }[];
  surges: { zone: number; t: number; peak: number; at: LatLon }[];
  requests: Rider[];
  events: { ev: string; rid: number | null; t: number; fare?: number }[];
}

export interface Replay {
  scenario: string;
  ppo: PolicyReplay;
  nearest: PolicyReplay;
}

export interface Metric {
  mean: number;
  ci95: number;
}

export interface Evaluation {
  episodes: number;
  summary: Record<string, Record<string, Metric>>;
  per_start_hour: Record<string, Record<string, number>>;
  traffic_source: string;
}

export interface LearningCurve {
  curve: { timesteps: number; profit_per_hour: number }[];
  nearest_baseline: number;
  best: number;
  timesteps: number;
  n_envs: number;
  train_seconds: number;
}

export interface TaxiGrid {
  curve: { episode: number; mean_return: number; epsilon: number }[];
  q_learning: { mean_return: number; mean_steps: number; success_rate: number };
  random: { mean_return: number; mean_steps: number; success_rate: number };
  episode: { row: number; col: number; passenger: number; dest: number; action: number }[];
}

export interface GmapsValidation {
  hour: number;
  n: number;
  traffic_source: string;
  mape: number;
  median_ape: number;
  pearson_r: number;
}
