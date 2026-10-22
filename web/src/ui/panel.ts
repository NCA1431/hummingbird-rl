/**
 * The page's controls and the head-up display, in plain HTML.
 *
 * Choosing an episode goes task -> controller -> episode. The HUD shows the
 * conditions of the episode (wind, mass, damage, the sudden failure if any),
 * the live state of the bird, and the result once the episode is over.
 */

import type { Episode, IndexEntry } from "../modes/watch";

export const SPEEDS: [number, string][] = [
  [1 / 64, "1/64"], [1 / 16, "1/16"], [1 / 4, "1/4"], [1, "1"], [2, "2"], [4, "4"],
];
export type CameraMode = "follow" | "overview" | "side" | "top" | "behind";

const TASK_NAMES: Record<string, string> = {
  hover: "Hover & recover", goto: "Go-to-point", obstacles: "Obstacles", navigation: "Navigation",
};

export interface PanelHandlers {
  select(entry: IndexEntry): void;
  playPause(): void;
  restart(): void;
  speed(s: number): void;
  seek(t: number): void;
  camera(mode: CameraMode): void;
  scale(s: number): void;
  wind(show: boolean): void;
  model(on: boolean): void;
}

export interface LiveValues {
  time: number;
  duration: number;
  speed: number;
  playing: boolean;
  altitude: number; // m
  velocity: number; // m/s
  distance: number; // m
  thrustRatio: number; // thrust / weight
  hold: number; // 0..1
  failed: boolean; // the sudden failure has happened
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export class Panel {
  private entries: IndexEntry[] = [];
  private readonly task = $<HTMLSelectElement>("task");
  private readonly policy = $<HTMLSelectElement>("policy");
  private readonly episode = $<HTMLSelectElement>("episode");
  private readonly play = $<HTMLButtonElement>("play");
  private readonly slider = $<HTMLInputElement>("time");
  private readonly info = $<HTMLDivElement>("info");
  private readonly live = $<HTMLDivElement>("live");
  private speedButtons: HTMLButtonElement[] = [];
  private seeking = false;

  constructor(private readonly h: PanelHandlers) {
    this.task.onchange = () => this.fillPolicies();
    this.policy.onchange = () => this.fillEpisodes();
    this.episode.onchange = () => this.choose();
    this.play.onclick = () => h.playPause();
    $<HTMLButtonElement>("restart").onclick = () => h.restart();
    this.slider.oninput = () => {
      this.seeking = true;
      h.seek(Number(this.slider.value));
    };
    this.slider.onchange = () => (this.seeking = false);

    const speeds = $<HTMLDivElement>("speeds");
    for (const [s, label] of SPEEDS) {
      const b = document.createElement("button");
      b.textContent = `×${label}`;
      b.onclick = () => h.speed(s);
      speeds.append(b);
      this.speedButtons.push(b);
    }
    document.querySelectorAll<HTMLButtonElement>("[data-camera]").forEach((b) => {
      b.onclick = () => h.camera(b.dataset.camera as CameraMode);
    });
    const scale = $<HTMLButtonElement>("scale");
    scale.onclick = () => {
      const big = scale.dataset.big !== "1";
      scale.dataset.big = big ? "1" : "0";
      scale.textContent = big ? "Bird ×3" : "Bird ×1";
      h.scale(big ? 3 : 1);
    };
    const wind = $<HTMLInputElement>("wind");
    wind.onchange = () => h.wind(wind.checked);
    const model = $<HTMLButtonElement>("model");
    model.onclick = () => {
      const on = model.dataset.on !== "1";
      model.dataset.on = on ? "1" : "0";
      model.textContent = on ? "3D model" : "Drawn bird";
      h.model(on);
    };
    const credits = $<HTMLDivElement>("credits");
    $<HTMLButtonElement>("credits-button").onclick = () => credits.classList.toggle("open");

    window.addEventListener("keydown", (e) => {
      if (e.target instanceof HTMLSelectElement) return;
      if (e.code === "Space") {
        e.preventDefault();
        h.playPause();
      } else if (e.key === "r") h.restart();
    });
  }

  setIndex(entries: IndexEntry[]): void {
    this.entries = entries;
    const tasks = [...new Set(entries.map((e) => e.task))].sort(
      (a, b) => Object.keys(TASK_NAMES).indexOf(a) - Object.keys(TASK_NAMES).indexOf(b),
    );
    this.task.replaceChildren(...tasks.map((t) => new Option(TASK_NAMES[t] ?? t, t)));
    this.fillPolicies();
  }

  private fillPolicies(): void {
    const policies = [...new Set(this.entries.filter((e) => e.task === this.task.value).map((e) => e.policy))];
    // PID and LQR first, then the agents in alphabetical order.
    const rank = (p: string) => (p === "pid" ? 0 : p === "lqr" ? 1 : 2);
    policies.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
    this.policy.replaceChildren(...policies.map((p) => new Option(["pid", "lqr"].includes(p) ? p.toUpperCase() : p, p)));
    this.fillEpisodes();
  }

  private fillEpisodes(): void {
    const list = this.matching();
    this.episode.replaceChildren(
      ...list.map((e, i) => new Option(`${e.suite} #${e.seed % 1000} · ${e.success ? "success" : e.reason.replace(/_/g, " ")}`, String(i))),
    );
    this.choose();
  }

  /** The episodes of the chosen task and controller: test episodes first, then by seed. */
  private matching(): IndexEntry[] {
    return this.entries
      .filter((e) => e.task === this.task.value && e.policy === this.policy.value)
      .sort((a, b) => Number(a.suite !== "test") - Number(b.suite !== "test") || a.seed - b.seed);
  }

  private choose(): void {
    const entry = this.matching()[Number(this.episode.value)];
    if (entry) this.h.select(entry);
  }

  showEpisode(ep: Episode, weight: number): void {
    const c = ep.conditions;
    const wind = Math.hypot(...c.mean_wind) / weight;
    const pct = (x: number) => `${Math.round(100 * x)}%`;
    const failure =
      c.failure_time !== null
        ? ` · <b>sudden failure</b>: ${c.failure_wing === 0 ? "left" : "right"} wing −${pct(c.failure_amount)} at ${c.failure_time.toFixed(2)} s`
        : "";
    this.info.innerHTML =
      `<b>${TASK_NAMES[ep.task] ?? ep.task}</b> · ${ep.policy ?? "?"} · ${ep.suite ?? ""} episode ${ep.seed}<br>` +
      `wind ${pct(wind)} of the weight, gusts ${pct(c.gust_sigma / weight)} · mass ×${c.mass_factor.toFixed(2)} · ` +
      `wing damage L ${pct(c.damage[0])} R ${pct(c.damage[1])}${failure}`;
    this.slider.max = String(ep.frames.t[ep.frames.t.length - 1]);
    this.slider.step = String(ep.dt);
  }

  update(ep: Episode, v: LiveValues): void {
    this.play.textContent = v.playing ? "Pause" : "Play";
    this.speedButtons.forEach((b, i) => b.classList.toggle("on", SPEEDS[i][0] === v.speed));
    if (!this.seeking) this.slider.value = String(v.time);
    const done = v.time >= v.duration - 1e-9;
    const result = done
      ? ` · <b>${ep.result.success ? "success" : ep.result.reason.replace(/_/g, " ")}</b>`
      : v.failed
        ? " · <b style='color:#c2410c'>wing failure</b>"
        : "";
    this.live.innerHTML =
      `t = ${v.time.toFixed(2)} / ${v.duration.toFixed(2)} s${result}<br>` +
      `altitude ${(100 * v.altitude).toFixed(1)} cm · speed ${(100 * v.velocity).toFixed(0)} cm/s · ` +
      `distance ${(100 * v.distance).toFixed(1)} cm · thrust ${v.thrustRatio.toFixed(2)} × weight` +
      (v.hold > 0 ? ` · holding ${Math.round(100 * v.hold)}%` : "");
  }

  /** Enables the model button once the 3D model has loaded (or says it could not be). */
  modelReady(ok = true): void {
    const model = $<HTMLButtonElement>("model");
    model.disabled = !ok;
    model.dataset.on = ok ? "1" : "0";
    model.textContent = ok ? "3D model" : "Drawn bird";
  }

  message(html: string): void {
    this.info.innerHTML = html;
  }
}
