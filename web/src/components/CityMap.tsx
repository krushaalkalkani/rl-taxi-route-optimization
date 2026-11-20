"use client";

import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { Network, PolicyReplay } from "@/lib/types";
import { surgeAt, taxiAt } from "@/lib/replay";

interface Props {
  network: Network;
  replay: PolicyReplay;
  t: number;
  color: string;
  label: string;
}

function congestionColor(m: number): string {
  // 1.0 → quiet pavement grey, 1.4 → amber, 1.9+ → brake red
  const x = Math.max(0, Math.min(1, (m - 1.0) / 0.9));
  const stops = [
    [150, 150, 142],
    [214, 150, 40],
    [194, 54, 43],
  ];
  const seg = x < 0.5 ? 0 : 1;
  const w = x < 0.5 ? x / 0.5 : (x - 0.5) / 0.5;
  const c = stops[seg].map((a, i) => Math.round(a + (stops[seg + 1][i] - a) * w));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export default function CityMap({ network, replay, t, color, label }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const roads = useRef<L.Polyline[]>([]);
  const lastHour = useRef<number>(-1);
  const layers = useRef<{
    history: L.LayerGroup;
    trail: L.Polyline;
    riders: L.LayerGroup;
    surges: L.LayerGroup;
    taxi: L.CircleMarker;
    historyCount: number;
  } | null>(null);

  // Build map once per network
  useEffect(() => {
    if (!el.current) return;
    const renderer = L.canvas({ padding: 0.3 });
    const m = L.map(el.current, {
      zoomControl: false,
      attributionControl: true,
      preferCanvas: true,
      scrollWheelZoom: false,
      zoomSnap: 0.1,
      renderer,
    });
    // No basemap: the street graph the agent drives on *is* the map.
    m.attributionControl.addAttribution('Streets &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors');
    L.control.zoom({ position: "bottomright" }).addTo(m);
    const fit = () => m.fitBounds(network.bounds as L.LatLngBoundsExpression, { padding: [10, 10] });
    fit();
    const ro = new ResizeObserver(() => {
      m.invalidateSize();
      fit();
    });
    ro.observe(el.current);

    roads.current = network.edges.map((g) =>
      L.polyline(g as L.LatLngExpression[], { weight: 2.5, color: "#96968e", opacity: 0.9, renderer, interactive: false }).addTo(m),
    );
    for (const h of network.hotspots) {
      L.circleMarker([h.lat, h.lon], {
        radius: 3,
        color: "#1c2228",
        weight: 1,
        fillColor: "#1c2228",
        fillOpacity: 1,
        renderer,
      })
        .bindTooltip(h.name, { direction: "top", offset: [0, -4] })
        .addTo(m);
    }
    const surges = L.layerGroup().addTo(m);
    const history = L.layerGroup().addTo(m);
    const trail = L.polyline([], { weight: 4, color, opacity: 0.95, renderer }).addTo(m);
    const riders = L.layerGroup().addTo(m);
    const taxi = L.circleMarker([0, 0], {
      radius: 8,
      color: "#1c2228",
      weight: 2,
      fillColor: color,
      fillOpacity: 1,
      renderer,
    }).addTo(m);
    layers.current = { history, trail, riders, surges, taxi, historyCount: -1 };
    map.current = m;
    lastHour.current = -1;
    return () => {
      ro.disconnect();
      m.remove();
      map.current = null;
    };
  }, [network, color]);

  // Per-frame update
  useEffect(() => {
    const ly = layers.current;
    if (!ly || !map.current) return;

    // Traffic colouring, once per simulated hour
    const hour = Math.floor(replay.start_hour + t / 3600) % 24;
    if (hour !== lastHour.current) {
      lastHour.current = hour;
      roads.current.forEach((p, i) => {
        const z = network.edge_zone[i];
        p.setStyle({ color: congestionColor(network.congestion[z][hour]) });
      });
    }

    // Faint history of completed drives
    const done = replay.drives.filter((d) => d.t1 <= t);
    if (done.length !== ly.historyCount) {
      ly.history.clearLayers();
      for (const d of done) {
        L.polyline(d.geom as L.LatLngExpression[], {
          weight: 3,
          color,
          opacity: d.kind === "passenger" ? 0.35 : 0.18,
          dashArray: d.kind === "passenger" ? undefined : "4 5",
        }).addTo(ly.history);
      }
      ly.historyCount = done.length;
    }

    // Current taxi
    const s = taxiAt(replay, t);
    if (s.pos) {
      ly.taxi.setLatLng(s.pos);
      ly.taxi.setStyle({ opacity: 1, fillOpacity: 1 });
    } else {
      ly.taxi.setStyle({ opacity: 0, fillOpacity: 0 });
    }
    if (s.kind === "driving") {
      ly.trail.setLatLngs(s.drive.geom as L.LatLngExpression[]);
      ly.trail.setStyle({ dashArray: s.drive.kind === "passenger" ? undefined : "6 6", opacity: 0.9 });
    } else {
      ly.trail.setLatLngs([]);
    }

    // Waiting riders (this taxi's universe)
    ly.riders.clearLayers();
    for (const r of replay.requests) {
      const gone = Math.min(r.expires, r.t_pickup ?? Infinity);
      if (r.t <= t && t < gone) {
        L.circleMarker(r.pickup, {
          radius: r.airport ? 6 : 4,
          color: r.airport ? "#2e5aa6" : "#1c2228",
          weight: 2,
          fillColor: "#ffffff",
          fillOpacity: 1,
        })
          .bindTooltip(r.airport ? "Rider going to LaGuardia" : "Rider waiting")
          .addTo(ly.riders);
      }
    }

    // Surge zones
    ly.surges.clearLayers();
    for (const z of surgeAt(replay, t)) {
      L.circle(z.at, {
        radius: 220 + 260 * Math.min(z.value, 1.2),
        color: "#c2362b",
        weight: 1,
        fillColor: "#c2362b",
        fillOpacity: 0.08 + 0.18 * Math.min(z.value, 1),
      })
        .bindTooltip(`Surge ×${(1 + z.value).toFixed(1)}`)
        .addTo(ly.surges);
    }
  }, [t, replay, network, color]);

  return <div ref={el} className="h-full w-full" role="img" aria-label={`Map of ${label}'s shift in Midtown Manhattan`} />;
}
