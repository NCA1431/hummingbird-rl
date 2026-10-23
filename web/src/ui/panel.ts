/**
 * The page's controls and the head-up display, in plain HTML.
 *
 * Two modes share the same controls and the same HUD:
 *   Recorded  task -> controller -> episode: the test episodes exported from
 *             Python, replayed exactly as they were measured;
 *   Live      task -> controller -> conditions, and "New flight": a new random
 *             situation, flown by the controller in the browser. Changing the
 *             controller or the conditions keeps the same seed, so two
 *             controllers can be compared on exactly the same scene.
 * The HUD shows the conditions of the episode (wind, mass, damage, the sudden
 * failure if any), the state of the bird, and the result once it is over.
 */

import type { PolicyEntry } from "../control/mlp";
import { newSeed } from "../modes/live";
import type { Episode, IndexEntry } from "../modes/watch";
import type { RegimeName } from "../sim/disturbances";
import { TASKS } from "../sim/tasks";

export const SPEEDS: [number, string][] = [
  [1 / 64, "1/64"], [1 / 16, "1/16"], [1 / 4, "1/4"], [1, "1"], [2, "2"], [4, "4"],
];
export type CameraMode = "follow" | "overview" | "side" | "top" | "behind" | "eye";
export type Mode = "recorded" | "live";

const TASK_NAMES: Record<string, string> = {
  hover: "Hover & recover", goto: "Go-to-point", obstacles: "Obstacles", navigation: "Navigation",
};

const ABOUT: Record<Mode, string> = {
  recorded: "Hummingbird RL · recorded test episodes, replayed exactly as they were measured.",
  live: "Hummingbird RL · live: new random situations, flown in your browser.",
};

/** What the live mode should fly. */
export interface LiveChoice {
  task: string;
  controller: string; // "pid", "lqr" or the name of an exported network
  file: string | null; // the network's file in public/policies/ (null for PID and LQR)
  regime: RegimeName;
  seed: number;
}

export interface PanelHandlers {
  select(entry: IndexEntry): void;
  live(choice: LiveChoice): void;
  playPause(): void;
  restart(): void;
  speed(s: number): void;
  seek(t: number): void;
  camera(mode: CameraMode): void;
  scale(s: number): void;
  wind(show: boolean): void;
  model(on: boolean): void;
}

/** What the HUD shows at the current time. */
export interface Readout {
  time: number;
  duration: number;
  speed: number;
  playing: boolean;
  ended: boolean; // the episode itself is over (its result is known)
  after: boolean; // what is shown now comes after the end (live mode, after a loss of control)
  altitude: number; // m
  velocity: number; // m/s
  distance: number; // m
  thrustRatio: number; // thrust / weight
  wings: [number, number, number]; // commanded half-stroke amplitude of the left and right wing (rad), frequency (Hz)
  hold: number; // 0..1
  failed: boolean; // the sudden failure has happened
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** PID and LQR first, then the agents in alphabetical order. */
const rank = (p: string) => (p === "pid" ? 0 : p === "lqr" ? 1 : 2);
const byRank = (a: string, b: string) => rank(a) - rank(b) || a.localeCompare(b);
/**
 * How an episode ended, in plain words: short for the menus, long for the HUD. The limits are
 * those of the simulator: ground at 2 cm, arena 1.5 m wide and 1.2 m high, 80° of pitch or roll.
 */
function ending(task: string, success: boolean, reason: string, long: boolean): string {
  if (success) return "success";
  const words: Record<string, [string, string]> = {
    time_limit: [
      "time limit",
      task === "hover"
        ? "time limit: more than 3 cm from the target on average over the last 2 s"
        : "time limit: the target was not held in time",
    ],
    loss_of_control: ["lost control", "lost control: tilted past 80°"],
    collision: ["hit an obstacle", "hit an obstacle"],
    crash: ["hit the ground", "crashed into the ground"],
    out_of_bounds: ["left the arena", "left the arena"],
  };
  const w = words[reason];
  return w ? w[long ? 1 : 0] : reason.replace(/_/g, " ");
}

/** "PID", "LQR", or a run name made readable: ppo_hover_randomised_s1 -> "PPO · randomised · #1". */
function label(p: string): string {
  if (p === "pid" || p === "lqr") return p.toUpperCase();
  const m = /^(ppo|sac|td3)_(hover|goto|obstacles|navigation)_(.+?)_s(\d+)(_best)?$/.exec(p);
  if (!m) return p;
  return [m[1].toUpperCase(), m[3].replace(/_/g, " "), `#${m[4]}`, m[5] ? "best" : ""].filter(Boolean).join(" · ");
}

export class Panel {
  mode: Mode = "recorded";
  private entries: IndexEntry[] = [];
  private networks: PolicyEntry[] = [];
  private seed = newSeed();
  private readonly task = $<HTMLSelectElement>("task");
  private readonly policy = $<HTMLSelectElement>("policy");
  private readonly episode = $<HTMLSelectElement>("episode");
  private readonly regime = $<HTMLSelectElement>("regime");
  private readonly newFlight = $<HTMLButtonElement>("new-flight");
  private readonly modeButtons = { recorded: $<HTMLButtonElement>("mode-recorded"), live: $<HTMLButtonElement>("mode-live") };
  private readonly play = $<HTMLButtonElement>("play");
  private readonly slider = $<HTMLInputElement>("time");
  private readonly info = $<HTMLDivElement>("info");
  private readonly readout = $<HTMLDivElement>("readout");
  private readonly clock = $<HTMLSpanElement>("clock");
  private readonly markFailure = $<HTMLSpanElement>("mark-failure");
  private readonly afterBand = $<HTMLSpanElement>("after-band");
  private speedButtons: HTMLButtonElement[] = [];
  private seeking = false;

  constructor(private readonly h: PanelHandlers) {
    this.task.onchange = () => this.fillPolicies(true);
    this.policy.onchange = () => (this.mode === "live" ? this.chooseLive() : this.fillEpisodes());
    this.episode.onchange = () => this.choose();
    this.regime.onchange = () => this.chooseLive();
    this.newFlight.onclick = () => this.chooseLive(true);
    this.modeButtons.recorded.onclick = () => this.setMode("recorded");
    this.modeButtons.live.onclick = () => this.setMode("live");
    this.play.onclick = () => h.playPause();
    $<HTMLButtonElement>("restart").onclick = () => h.restart();
    this.slider.oninput = () => {
      this.seeking = true;
      h.seek(Number(this.slider.value));
    };
    this.slider.onchange = () => (this.seeking = false);

    const speeds = $<HTMLDivElement>("speeds");
    for (const [s, text] of SPEEDS) {
      const b = document.createElement("button");
      b.textContent = `×${text}`;
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
    // Fold buttons. On a phone the HUD starts folded: the two panels would hide half the scene.
    for (const [buttonId, panelId] of [["fold-hud", "hud"], ["fold-controls", "controls"]]) {
      const button = $<HTMLButtonElement>(buttonId);
      const box = $<HTMLDivElement>(panelId);
      const fold = (folded: boolean) => {
        box.classList.toggle("folded", folded);
        button.textContent = folded ? "+" : "−";
        button.title = folded ? "Show everything" : "Fold";
      };
      button.onclick = () => fold(!box.classList.contains("folded"));
      if (panelId === "hud" && window.innerWidth < 600) fold(true);
    }

    // Two small panels, one at a time, just above the controls (whose height depends on the
    // screen width): what the scene shows, and the credits.
    const controls = $<HTMLDivElement>("controls");
    const popups = [$<HTMLDivElement>("legend"), $<HTMLDivElement>("credits")];
    ["legend-button", "credits-button"].forEach((id, i) => {
      $<HTMLButtonElement>(id).onclick = () => {
        popups[1 - i].classList.remove("open");
        popups[i].style.bottom = `${controls.offsetHeight + 20}px`;
        popups[i].classList.toggle("open");
      };
    });

    window.addEventListener("keydown", (e) => {
      if (e.target instanceof HTMLSelectElement) return;
      if (e.code === "Space") {
        e.preventDefault();
        h.playPause();
      } else if (e.key === "r") h.restart();
      else if (e.key === "n" && this.mode === "live") this.chooseLive(true);
    });
  }

  /** The recorded episodes (public/episodes/index.json). Without any, the page starts in live mode. */
  setIndex(entries: IndexEntry[]): void {
    this.entries = entries;
    this.modeButtons.recorded.disabled = entries.length === 0;
    this.setMode(entries.length ? this.mode : "live");
  }

  /** The networks exported for the live mode (public/policies/index.json). */
  setPolicies(networks: PolicyEntry[]): void {
    this.networks = networks;
    if (this.mode === "live") this.fillPolicies(false, false); // the current flight stays on screen
  }

  setMode(mode: Mode): void {
    this.mode = mode;
    const live = mode === "live";
    this.modeButtons.recorded.classList.toggle("on", !live);
    this.modeButtons.live.classList.toggle("on", live);
    this.episode.hidden = live;
    this.regime.hidden = !live;
    this.newFlight.hidden = !live;
    $<HTMLDivElement>("about").textContent = ABOUT[mode];
    this.fillTasks();
  }

  private fillTasks(): void {
    const tasks = this.mode === "live" ? [...TASKS] : [...new Set(this.entries.map((e) => e.task))];
    tasks.sort((a, b) => Object.keys(TASK_NAMES).indexOf(a) - Object.keys(TASK_NAMES).indexOf(b));
    keep(this.task, tasks.map((t) => new Option(TASK_NAMES[t] ?? t, t)));
    this.fillPolicies(true);
  }

  /** The controllers available for the chosen task; a new task gets a new live seed. */
  private fillPolicies(newTask: boolean, fly = true): void {
    const task = this.task.value;
    const names =
      this.mode === "live"
        ? ["pid", "lqr", ...this.networks.filter((n) => n.task === task).map((n) => n.name)]
        : [...new Set(this.entries.filter((e) => e.task === task).map((e) => e.policy))];
    keep(this.policy, names.sort(byRank).map((p) => new Option(label(p), p)));
    if (!fly) return;
    if (this.mode === "live") this.chooseLive(newTask);
    else this.fillEpisodes();
  }

  private fillEpisodes(): void {
    const list = this.matching();
    this.episode.replaceChildren(
      ...list.map((e, i) => new Option(`${e.suite} #${e.seed % 1000} · ${ending(e.task, e.success, e.reason, false)}`, String(i))),
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

  private chooseLive(newFlight = false): void {
    if (newFlight) this.seed = newSeed();
    const name = this.policy.value;
    const network = this.networks.find((n) => n.name === name);
    this.h.live({
      task: this.task.value,
      controller: name,
      file: network ? network.file : null,
      regime: this.regime.value as RegimeName,
      seed: this.seed,
    });
  }

  showEpisode(ep: Episode, weight: number): void {
    const c = ep.conditions;
    const wind = Math.hypot(...c.mean_wind) / weight;
    const pct = (x: number) => `${Math.round(100 * x)}%`;
    const total = ep.frames.t[ep.frames.t.length - 1];
    const wing = c.failure_wing === 0 ? "left" : "right";
    const failure =
      c.failure_time === null
        ? ""
        : c.failure_time < total
          ? ` · <b>sudden failure</b>: ${wing} wing −${pct(c.failure_amount)} at ${c.failure_time.toFixed(2)} s`
          : ` · sudden failure planned at ${c.failure_time.toFixed(2)} s, not reached (the flight ended before)`;
    const which = ep.suite === "live" ? `live flight, seed ${ep.seed}` : `${ep.suite ?? ""} episode ${ep.seed}`;
    // Only what differs from the nominal bird is listed: a mass other than 1 and a wing weakened
    // from the start appear in randomised conditions, the sudden failure in its own regime.
    // In % of the normal bird's weight, the unit in which the conditions are drawn (up to 30%
    // in randomised conditions), whatever the mass of this bird.
    const air = wind === 0 && c.gust_sigma === 0 ? "calm air" : `wind ${pct(wind)} of the weight, gusts ${pct(c.gust_sigma / weight)}`;
    const mass = c.mass_factor !== 1 ? ` · mass ×${c.mass_factor.toFixed(2)}` : "";
    const weak = [0, 1]
      .filter((i) => c.damage[i] > 0)
      .map((i) => ` · ${i === 0 ? "left" : "right"} wing weakened by ${pct(c.damage[i])} from the start`)
      .join("");
    this.info.innerHTML =
      `<b>${TASK_NAMES[ep.task] ?? ep.task}</b> · ${label(ep.policy ?? "?")} · ${which}` +
      `<span class="more"><br>${air}${mass}${weak}${failure}</span>`;
    this.slider.max = String(total);
    this.slider.step = "any"; // a flight can end between two decisions: the end must be reachable
    place(this.markFailure, c.failure_time !== null && c.failure_time < total ? c.failure_time / total : null);
    place(this.afterBand, ep.endFrame !== undefined ? ep.frames.t[ep.endFrame] / total : null);
  }

  update(ep: Episode, v: Readout): void {
    this.play.textContent = v.playing ? "Pause" : "Play";
    this.speedButtons.forEach((b, i) => b.classList.toggle("on", SPEEDS[i][0] === v.speed));
    if (!this.seeking) this.slider.value = String(v.time);
    this.clock.textContent = `${v.time.toFixed(2)} / ${v.duration.toFixed(2)} s`;
    const failure = v.failed ? " · <b style='color:#c2410c'>wing failure</b>" : "";
    const status = v.ended
      ? `<b>${ending(ep.task, ep.result.success, ep.result.reason, true)}</b>` +
        (v.after ? " · <i>after the end of the episode: shown, not counted</i>" : "")
      : `flying${failure}`;
    this.readout.innerHTML =
      `${status}<span class="more"><br>` +
      `altitude ${(100 * v.altitude).toFixed(1)} cm · speed ${(100 * v.velocity).toFixed(0)} cm/s · ` +
      `distance ${(100 * v.distance).toFixed(1)} cm · thrust ${v.thrustRatio.toFixed(2)} × weight` +
      (v.hold > 0 ? ` · holding ${Math.round(100 * v.hold)}%` : "") +
      `<br>wings: left ${deg(v.wings[0])}°, right ${deg(v.wings[1])}° of half-stroke, ${v.wings[2].toFixed(0)} Hz</span>`;
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

const deg = (rad: number) => ((rad * 180) / Math.PI).toFixed(0);

/** Puts a timeline mark (or the start of the band) at fraction `f` of the slider's length, or hides it (null). */
function place(mark: HTMLElement, f: number | null): void {
  mark.hidden = f === null;
  if (f !== null) mark.style.left = `calc(8px + (100% - 16px) * ${f.toFixed(4)})`;
}

/** Replaces a select's options, keeping the current choice when it is still offered. */
function keep(select: HTMLSelectElement, options: HTMLOptionElement[]): void {
  const before = select.value;
  select.replaceChildren(...options);
  if (options.some((o) => o.value === before)) select.value = before;
}
