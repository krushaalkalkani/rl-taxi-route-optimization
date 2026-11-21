import Replay from "@/components/Replay";
import GridTaxi from "@/components/GridTaxi";
import { ByHourChart, GridCurveChart, LearningCurveChart } from "@/components/Charts";
import { readJson } from "@/lib/data";
import type { Evaluation, GmapsValidation, LearningCurve, Network, TaxiGrid } from "@/lib/types";

const REPO = "https://github.com/krushaalkalkani/rl-taxi-route-optimization";

const POLICY_ROWS: { key: string; name: string; what: string }[] = [
  { key: "ppo", name: "RL agent (MaskablePPO)", what: "Learned from 4M simulated decisions" },
  { key: "smart", name: "Hand-tuned rules", what: "Fare per minute, skips airport runs in traffic, chases surge" },
  { key: "nearest", name: "Nearest rider", what: "Always takes the closest rider who can be reached in time" },
  { key: "fare_rate", name: "Best fare rate", what: "Highest fare per minute of driving, airport included" },
  { key: "random", name: "Random", what: "Any valid action, for scale" },
];

function pct(a: number, b: number) {
  const v = ((a - b) / b) * 100;
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}%`;
}

export default function Home() {
  const network = readJson<Network>("network.json");
  const evaluation = readJson<Evaluation>("evaluation.json");
  const curve = readJson<LearningCurve>("learning_curve.json");
  const grid = readJson<TaxiGrid>("taxi_v4.json");
  const gmaps = readJson<GmapsValidation>("gmaps_validation.json");
  const s = evaluation?.summary;
  const ppo = s?.ppo?.profit_per_hour.mean;
  const nearest = s?.nearest?.profit_per_hour.mean;
  const smart = s?.smart?.profit_per_hour.mean;
  const paired = evaluation?.paired_vs_nearest?.ppo;
  const googleCalibrated = network?.traffic_source === "google-routes-api";

  return (
    <div className="mx-auto max-w-[1240px] px-4 pb-20 sm:px-8">
      <header className="flex items-center justify-between py-5">
        <span className="whitespace-nowrap font-display text-xl font-bold">Midtown Cab RL</span>
        <nav className="flex gap-5 whitespace-nowrap text-[15px]">
          <a href="#results" className="hidden hover:underline sm:inline">Results</a>
          <a href="#how" className="hidden hover:underline sm:inline">How it works</a>
          <a href={REPO} className="font-medium underline">Code on GitHub</a>
        </nav>
      </header>

      <main>
        <section className="pt-6">
          <h1 className="font-display text-[44px] font-bold leading-[1.02] sm:text-[68px]">
            A taxi that learned to work Midtown
          </h1>
          <p className="prose-measure mt-4 text-lg text-[var(--ink-2)]">
            Two cabs work the same shift on Manhattan&apos;s real streets, with the same riders and the same
            traffic. The yellow one is a reinforcement-learning agent. It picks which rider to take, when to turn
            down a LaGuardia run, and where to wait. The blue one always takes the nearest rider.
          </p>
        </section>

        <Replay />
        <p className="prose-measure mt-4 text-[15px] text-[var(--ink-2)]">
          Each replay is one four-hour shift on a seed fixed before training finished, not picked afterwards.
          Single shifts are noisy: the agent loses some of them.
          {paired ? ` Across ${evaluation?.episodes} test shifts it comes out ahead on ${Math.round(paired.win_rate * 100)}%.` : ""}
        </p>

        <section id="results" className="mt-20 scroll-mt-6">
          <h2 className="font-display text-4xl font-bold">Results over {evaluation?.episodes ?? 0} shifts</h2>
          {s && ppo && nearest && smart ? (
            <>
              <p className="prose-measure mt-3 text-lg">
                Across {evaluation.episodes} held-out four-hour shifts, spread evenly over every start hour,
                the agent earns <strong>${ppo.toFixed(2)}</strong> per hour after fuel, compared with $
                {nearest.toFixed(2)} for the nearest-rider rule ({pct(ppo, nearest)}) and ${smart.toFixed(2)} for
                rules I tuned by hand ({pct(ppo, smart)}).
                {paired && (
                  <>
                    {" "}Because every policy faces the same riders on each seed, the per-shift difference is
                    measured directly: <strong>{paired.mean_diff >= 0 ? "+" : "−"}${Math.abs(paired.mean_diff).toFixed(2)}/h
                    ± {paired.ci95.toFixed(2)}</strong> (95% CI), and the agent comes out ahead on{" "}
                    {Math.round(paired.win_rate * 100)}% of shifts.
                  </>
                )}
              </p>
              <div className="mt-8 overflow-x-auto">
                <table className="w-full min-w-[640px] text-left tabular-nums">
                  <thead>
                    <tr className="border-b-2 border-[var(--ink)] text-sm text-[var(--ink-2)]">
                      <th className="py-2 pr-4 font-normal">Policy</th>
                      <th className="py-2 pr-4 font-normal">Profit / hour</th>
                      <th className="py-2 pr-4 font-normal">Rides per shift</th>
                      <th className="py-2 pr-4 font-normal">Time with rider</th>
                      <th className="py-2 font-normal">Empty km</th>
                    </tr>
                  </thead>
                  <tbody>
                    {POLICY_ROWS.filter((p) => s[p.key]).map((p) => {
                      const m = s[p.key];
                      return (
                        <tr key={p.key} className={`border-b border-[var(--rule)] ${p.key === "ppo" ? "bg-[var(--medallion)]/15" : ""}`}>
                          <td className="py-3 pr-4">
                            <div className="font-medium">{p.name}</div>
                            <div className="text-sm text-[var(--ink-2)]">{p.what}</div>
                          </td>
                          <td className="py-3 pr-4">
                            <span className="font-display text-2xl font-semibold">${m.profit_per_hour.mean.toFixed(2)}</span>
                            <span className="ml-1.5 text-sm text-[var(--ink-2)]">±{m.profit_per_hour.ci95.toFixed(2)}</span>
                          </td>
                          <td className="py-3 pr-4">{m.served.mean.toFixed(1)}</td>
                          <td className="py-3 pr-4">{Math.round(m.utilization.mean * 100)}%</td>
                          <td className="py-3">{Math.round(m.empty_ratio.mean * 100)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="mt-2 text-sm text-[var(--ink-2)]">± is a 95% confidence interval over shifts.</p>
              </div>

              <div className="mt-14 grid gap-12 lg:grid-cols-2">
                <div>
                  <h3 className="font-display text-2xl font-semibold">Profit by time of day</h3>
                  <p className="mt-1 text-[var(--ink-2)]">
                    Most of the gain comes in the evening rush and late at night, when choosing well
                    between airport runs, surge zones and short hops matters most.
                  </p>
                  <div className="mt-4">
                    <ByHourChart data={evaluation} />
                  </div>
                </div>
                {curve && (
                  <div>
                    <h3 className="font-display text-2xl font-semibold">Learning curve</h3>
                    <p className="mt-1 text-[var(--ink-2)]">
                      Profit per hour on 96 validation shifts during training: {(curve.timesteps / 1e6).toFixed(0)}M
                      decisions on {curve.n_envs} parallel simulators, on a laptop CPU.
                    </p>
                    <div className="mt-4">
                      <LearningCurveChart data={curve} />
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <p className="mt-3 text-[var(--ink-2)]">Run <code>python -m taxi_rl.evaluate</code> to populate results.</p>
          )}
        </section>

        <section id="how" className="mt-24 scroll-mt-6">
          <h2 className="font-display text-4xl font-bold">How it works</h2>
          <p className="prose-measure mt-3 text-lg">
            The whole pipeline is open source and runs on a laptop. Each stage feeds the next.
          </p>
          <ol className="mt-8 grid gap-x-12 gap-y-10 md:grid-cols-2">
            <Step n={1} title="Real street network">
              Midtown Manhattan from 9th Ave to 1st Ave and 14th to 57th St, pulled from OpenStreetMap
              with OSMnx: {network?.n_nodes.toLocaleString("en-US") ?? "—"} intersections and{" "}
              {network?.n_edges.toLocaleString("en-US") ?? "—"} directed road
              segments, one-way streets included. Routing runs SciPy&apos;s C Dijkstra across all pairs of
              intersections.
            </Step>
            <Step n={2} title="Traffic that changes by the hour">
              Every street&apos;s travel time is scaled by a congestion factor for its zone and the hour. The
              shortest route is recomputed under each hour&apos;s traffic, so the 6 PM route differs from the
              3 AM route. A Google Maps Routes API client calibrates these factors against Google&apos;s
              traffic-aware ETAs.
            </Step>
            <Step n={3} title="Riders, surge and the airport">
              Riders appear at 11 real hotspots (Penn Station, Grand Central, Times Square and others), each
              with its own daily rhythm. Surge events briefly raise demand and fares in one zone. Some riders
              want LaGuardia: a big fare, but the cab leaves Midtown and may come back empty.
            </Step>
            <Step n={4} title="A Gymnasium environment">
              <code>CityTaxi-v0</code> is a semi-Markov decision process. At each decision the agent can take
              one of the 5 closest reachable riders, drive to one of 8 zones, or wait. It observes 87 features
              and gets action masks for invalid moves. Reward is fare minus driving cost. It passes
              Gymnasium&apos;s <code>check_env</code>.
            </Step>
            <Step n={5} title="Training the agent">
              Stable-Baselines3 MaskablePPO with a custom permutation-equivariant policy: one shared network
              scores every candidate rider, another scores every zone, so a lesson learned about one rider
              applies to all of them. Each action is charged its duration at a reference earning rate
              (average-reward shaping for the semi-MDP). A plain MLP plateaued at +1% over greedy. The scoring
              policy reached +6%.
            </Step>
            <Step n={6} title="Fair evaluation">
              Rider demand is drawn up front from its own random stream, so every policy faces identical
              riders on each seed. The model was chosen on validation seeds, then tested once on 480 unseen
              shifts covering all 24 start hours equally, with 95%
              confidence intervals on every metric.
            </Step>
          </ol>
        </section>

        {grid && (
          <section className="mt-24">
            <h2 className="font-display text-4xl font-bold">Warm-up: the Gymnasium Taxi</h2>
            <div className="mt-6 grid items-start gap-12 lg:grid-cols-[auto_1fr]">
              <GridTaxi data={grid} />
              <div>
                <p className="prose-measure text-lg">
                  Before the city, the classic <code>Taxi-v4</code> from the Gymnasium docs: a 5×5 grid, 500
                  states, 6 actions. Tabular Q-learning (α = 0.1, γ = 0.99, ε decaying from 1.0) solves{" "}
                  <strong>{Math.round(grid.q_learning.success_rate * 100)}%</strong> of held-out episodes in{" "}
                  {grid.q_learning.mean_steps} steps on average. A random policy succeeds{" "}
                  {Math.round(grid.random.success_rate * 100)}% of the time. A table of values works here. It
                  can&apos;t scale to a real city, where the state includes continuous time, traffic and a
                  changing set of riders. That&apos;s why the city agent uses a neural policy.
                </p>
                <div className="mt-4 max-w-xl">
                  <GridCurveChart data={grid} />
                </div>
              </div>
            </div>
          </section>
        )}

        <div className="mt-24 grid gap-x-16 gap-y-20 lg:grid-cols-2">
        <section>
          <h2 className="font-display text-4xl font-bold">Google Maps in the loop</h2>
          <div className="prose-measure mt-3 space-y-4 text-lg">
            <p>
              <code>taxi_rl.gmaps</code> calls the Routes API <code>computeRouteMatrix</code> endpoint in
              traffic-aware mode. It samples origin and destination intersections in each zone at several
              hours of a typical weekday and compares Google&apos;s traffic ETA with its no-traffic ETA. From
              that it fits the simulator&apos;s congestion table and a correction for OpenStreetMap speed limits.
              A <code>validate</code> command then reports the error between simulator and Google ETAs on random
              trips.
            </p>
            {gmaps ? (
              <p>
                Against Google&apos;s ETAs for {gmaps.n} random trips at {gmaps.hour}:00, the simulator&apos;s
                mean absolute percentage error is <strong>{(gmaps.mape * 100).toFixed(1)}%</strong> (r ={" "}
                {gmaps.pearson_r.toFixed(2)}).
              </p>
            ) : (
              <p className="text-[var(--ink-2)]">
                {googleCalibrated
                  ? "This deployment uses a Google-calibrated traffic table."
                  : "The results on this page use the built-in rush-hour traffic profile. Add a Routes API key and run the calibration to swap in Google-measured congestion."}
              </p>
            )}
          </div>
        </section>

        <section>
          <h2 className="font-display text-4xl font-bold">What this doesn&apos;t model</h2>
          <ul className="prose-measure mt-4 list-disc space-y-2 pl-5 text-lg">
            <li>Rider demand is synthetic, shaped around real hotspots, not replayed from NYC TLC trip records.</li>
            <li>There&apos;s one cab. Other drivers appear only as riders who give up waiting.</li>
            <li>Traffic varies by zone and hour, not by individual street or incident.</li>
            <li>Gains over a well-tuned heuristic are modest. Most of the room to improve is in a few hard calls.</li>
          </ul>
        </section>
        </div>
      </main>

      <footer className="mt-24 flex flex-wrap justify-between gap-4 border-t border-[var(--rule)] pt-6 text-[15px] text-[var(--ink-2)]">
        <span>Built by Krushal Kalkani with Gymnasium, Stable-Baselines3, OSMnx and the Google Maps Routes API.</span>
        <a href={REPO} className="underline">Source and training code</a>
      </footer>
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li className="grid grid-cols-[44px_1fr] gap-3">
      <span
        aria-hidden
        className="grid h-9 w-9 place-items-center rounded-full border-2 border-[var(--ink)] font-display text-lg font-bold"
      >
        {n}
      </span>
      <div>
        <h3 className="font-display text-2xl font-semibold leading-tight">{title}</h3>
        <p className="mt-1.5 text-[var(--ink-2)]">{children}</p>
      </div>
    </li>
  );
}
