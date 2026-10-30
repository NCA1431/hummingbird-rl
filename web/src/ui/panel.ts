/**
 * The page's controls and the head-up display, in plain HTML.
 *
 * Three modes share the same controls and the same HUD:
 *   Recorded  task -> controller -> episode: the test episodes exported from
 *             Python, replayed exactly as they were measured;
 *   Live      task -> controller -> conditions, and "New flight": a new random
 *             situation, flown by the controller in the browser. Changing the
 *             controller or the conditions keeps the same seed, so two
 *             controllers can be compared on exactly the same scene;
 *   Sandbox   the visitor builds the scene and the conditions (sandboxPanel.ts),
 *             and flies up to four controllers in the same situation;
 *   Play      the birds fly without end, in real time, after a target the visitor
 *             moves, among obstacles the visitor moves (modes/play.ts). Any
 *             controller of any task can fly; there is no timeline.
 * The HUD shows the conditions of the episode (wind, mass, damage, the sudden
 * failure if any), the state of the bird, and the result once it is over. With
 * several birds, it shows the state of each, and the numbers of the one followed.
 */

import type { PolicyEntry } from "../control/mlp";
import { newSeed } from "../modes/live";
import type { Tool } from "../modes/sandbox";
import type { Episode, IndexEntry } from "../modes/watch";
import type { Conditions, RegimeName } from "../sim/disturbances";
import { TASKS } from "../sim/tasks";
import type { Scene } from "../sim/world";
import type { FluidScales } from "../sim/fluid";
import { type AirMode, paletteCss } from "../view/air";
import { BIRD_COLORS } from "../view/bird";
import { type BirdSlot, type ItemRef, type PlayEdit, SandboxPanel } from "./sandboxPanel";
import { TASK_NAMES, counterpart, ending, label } from "./words";

export const SPEEDS: [number, string][] = [
  [1 / 64, "1/64"], [1 / 16, "1/16"], [1 / 4, "1/4"], [1, "1"], [2, "2"], [4, "4"],
];
export type CameraMode = "follow" | "overview" | "side" | "top" | "behind" | "eye";
export type Mode = "recorded" | "live" | "sandbox" | "play";
const MODES = ["recorded", "live", "sandbox", "play"] as const;

const ABOUT: Record<Mode, string> = {
  recorded: "Hummingbird RL · recorded test episodes, replayed exactly as they were measured.",
  live: "Hummingbird RL · live: new random situations, flown in your browser.",
  sandbox: "Hummingbird RL · sandbox: build a scene, then fly up to four controllers in it.",
  play: "Hummingbird RL · play: move the target and the obstacles while the birds fly.",
};

/** The tasks in the order the Play menu lists their networks: the closest to Play first. */
const PLAY_ORDER = ["goto", "obstacles", "navigation", "hover"];

/** What the live mode should fly. */
export interface LiveChoice {
  task: string;
  controller: string; // "pid", "lqr" or the name of an exported network
  file: string | null; // the network's file in public/policies/ (null for PID and LQR)
  regime: RegimeName;
  seed: number;
}

/** A controller of the sandbox: its name, and its network file (null for PID and LQR). */
export interface Pilot {
  controller: string;
  file: string | null;
  task?: string | null; // the task the network was trained on (Play: does it know obstacles?)
  id?: number; // the bird's identity (Play)
}

/** What Play should fly: the birds and the conditions; the scene is moved by hand in the page. */
export interface PlayChoice {
  pilots: Pilot[];
  followed: number;
  conditions: Conditions;
  seesTarget: boolean;
}

/**
 * What the sandbox should fly: up to four controllers in a scene and conditions set by hand.
 * No pilots: the scene cannot be flown yet (a navigation course without waypoints), only shown.
 */
export interface SandboxChoice {
  task: string;
  pilots: Pilot[];
  followed: number; // the bird the camera and the HUD follow
  scene: Scene;
  conditions: Conditions;
  seed: number;
  resetCamera: boolean; // false while editing, so that the view stays where the visitor put it
}

export interface PanelHandlers {
  select(entry: IndexEntry): void;
  live(choice: LiveChoice): void;
  sandbox(choice: SandboxChoice): void;
  play(choice: PlayChoice): void;
  playEdit(edit: PlayEdit): void;
  sandboxPicked(item: ItemRef | null): void; // the Sandbox's object picked (it glows)
  current(): Episode | null; // the episode on screen, which the sandbox starts from
  tool(tool: Tool): void;
  looks(slots: BirdSlot[]): void; // the sandbox birds' look changed (3D model or drawn)
  follow(index: number): void; // the sandbox's followed bird changed
  playPause(): void;
  restart(): void;
  speed(s: number): void;
  seek(t: number): void;
  camera(mode: CameraMode): void;
  scale(s: number): void;
  wind(show: boolean): void;
  air(mode: AirMode): void;
  airBox(on: boolean): void;
  trails(on: boolean): void;
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
  birds: BirdState[]; // the sandbox's birds, when there are several (empty otherwise)
}

/** What the HUD shows in Play. */
export interface PlayReadout {
  time: number;
  speed: number;
  playing: boolean;
  seesTarget: boolean;
  following: boolean;
  birds: { controller: string; state: string; falls: number; followed: boolean }[];
  altitude: number;
  velocity: number;
  distance: number;
  thrustRatio: number;
  wings: [number, number, number];
  hold: number;
}

/** The state of one of the sandbox's birds, for the HUD's first line. */
export interface BirdState {
  ep: Episode;
  ended: boolean;
  followed: boolean;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** PID and LQR first, then the agents in alphabetical order. */
const rank = (p: string) => (p === "pid" ? 0 : p === "lqr" ? 1 : 2);
const byRank = (a: string, b: string) => rank(a) - rank(b) || a.localeCompare(b);
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
  private readonly modeButtons = {
    recorded: $<HTMLButtonElement>("mode-recorded"),
    live: $<HTMLButtonElement>("mode-live"),
    sandbox: $<HTMLButtonElement>("mode-sandbox"),
    play: $<HTMLButtonElement>("mode-play"),
  };
  readonly sandbox: SandboxPanel;
  private readonly play = $<HTMLButtonElement>("play");
  private readonly slider = $<HTMLInputElement>("time");
  private readonly info = $<HTMLDivElement>("info");
  private readonly readout = $<HTMLDivElement>("readout");
  private readonly clock = $<HTMLSpanElement>("clock");
  private readonly markFailure = $<HTMLSpanElement>("mark-failure");
  private readonly afterBand = $<HTMLSpanElement>("after-band");
  private speedButtons: HTMLButtonElement[] = [];
  private seeking = false;
  private readonly folds: Record<string, (folded: boolean) => void> = {};
  private playSeen = false;
  private playLabels = new Map<string, string>(); // Play's names for its controllers, with their task

  constructor(private readonly h: PanelHandlers) {
    this.sandbox = new SandboxPanel({
      changed: (resetCamera) => (this.mode === "play" ? this.choosePlay() : this.chooseSandbox(resetCamera)),
      tool: (tool) => h.tool(tool),
      looks: (slots) => h.looks(slots),
      follow: (index) => h.follow(index),
      play: (edit) => h.playEdit(edit),
      picked: (item) => h.sandboxPicked(item),
    });
    this.task.onchange = () => this.fillPolicies(true);
    this.policy.onchange = () => (this.mode === "live" ? this.chooseLive() : this.fillEpisodes()); // hidden in the sandbox
    this.episode.onchange = () => this.choose();
    this.regime.onchange = () => this.chooseLive();
    this.newFlight.onclick = () => this.chooseLive(true);
    this.modeButtons.recorded.onclick = () => this.setMode("recorded");
    this.modeButtons.live.onclick = () => this.setMode("live");
    this.modeButtons.sandbox.onclick = () => this.setMode("sandbox");
    this.modeButtons.play.onclick = () => this.setMode("play");
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
    const airMenu = $<HTMLSelectElement>("air");
    // The air is simulated around the bird at its real size: drawn 3 times larger, the bird would
    // spread its wings far outside the air they push. So with the air on we draw the bird at its
    // real size, and the visitor's choice comes back when the air is turned off.
    const applyScale = () => {
      const air = airMenu.value !== "off";
      const big = scale.dataset.big === "1" && !air;
      scale.disabled = air;
      scale.title = air ? "With the air shown, the bird is drawn at its real size, as the air is" : "Draw the birds 3 times larger, to see them from afar";
      scale.textContent = big ? "Bird ×3" : "Bird ×1";
      h.scale(big ? 3 : 1);
    };
    scale.title = "Draw the birds 3 times larger, to see them from afar";
    scale.onclick = () => {
      scale.dataset.big = scale.dataset.big === "1" ? "0" : "1";
      applyScale();
    };
    const wind = $<HTMLInputElement>("wind");
    wind.onchange = () => h.wind(wind.checked);
    const airBox = $<HTMLInputElement>("air-box");
    airMenu.onchange = () => {
      h.air(airMenu.value as AirMode);
      $<HTMLLabelElement>("air-box-label").hidden = airMenu.value === "off";
      applyScale();
    };
    airBox.onchange = () => h.airBox(airBox.checked);
    const trails = $<HTMLInputElement>("trails");
    trails.onchange = () => h.trails(trails.checked);
    const model = $<HTMLButtonElement>("model");
    model.onclick = () => {
      const on = model.dataset.on !== "1";
      model.dataset.on = on ? "1" : "0";
      model.textContent = on ? "3D model" : "Drawn bird";
      h.model(on);
    };
    // Fold buttons. On a phone the HUD starts folded: the two panels would hide half the scene.
    for (const [buttonId, panelId] of [["fold-hud", "hud"], ["fold-controls", "controls"], ["fold-sandbox", "sandbox"]]) {
      const button = $<HTMLButtonElement>(buttonId);
      const box = $<HTMLDivElement>(panelId);
      const fold = (folded: boolean) => {
        box.classList.toggle("folded", folded);
        button.textContent = folded ? "+" : "−";
        button.title = folded ? "Show everything" : "Fold";
      };
      button.onclick = () => fold(!box.classList.contains("folded"));
      if (panelId === "hud" && window.innerWidth < 600) fold(true);
      this.folds[panelId] = fold;
    }

    // Two small panels, one at a time, just above the controls (whose height depends on the
    // screen width): what the scene shows, and the credits.
    const controls = $<HTMLDivElement>("controls");
    const popups = [$<HTMLDivElement>("legend"), $<HTMLDivElement>("credits")];
    ["legend-button", "credits-button"].forEach((id, i) => {
      $<HTMLButtonElement>(id).onclick = () => {
        popups[1 - i].classList.remove("open");
        if (!popups[i].dataset.moved) popups[i].style.bottom = `${controls.offsetHeight + 20}px`;
        popups[i].classList.toggle("open");
      };
    });
    for (const id of ["hud", "controls", "sandbox", "legend", "credits"]) draggable($<HTMLDivElement>(id));

    window.addEventListener("keydown", (e) => {
      if (e.target instanceof HTMLSelectElement) return;
      if (e.code === "Space") {
        e.preventDefault();
        h.playPause();
      } else if (e.key === "r") h.restart();
      else if (e.key === "n" && this.mode === "live") this.chooseLive(true);
      else if (this.mode === "sandbox" && (e.key === "Escape" || e.key === "Delete" || e.key === "Backspace")) {
        if (e.key === "Escape") this.sandbox.pick(null);
        else this.sandbox.removePicked();
      } else if (this.mode === "play" && !(e.target instanceof HTMLInputElement && e.target.type === "text")) {
        // (a checkbox or a slider just clicked keeps the focus; the keys must still work)
        if (e.key === "f") this.sandbox.setFollowing(!this.sandbox.following);
        else if (e.key === "Escape") this.sandbox.setFollowing(false);
        else if (e.key === "Delete" || e.key === "Backspace") h.playEdit({ kind: "remove" });
      }
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
    if (this.mode === "play") {
      this.sandbox.setControllers(this.playControllers());
      this.choosePlay();
    } else if (this.mode !== "recorded") this.fillPolicies(false, false); // the current flight stays on screen
  }

  setMode(mode: Mode): void {
    this.mode = mode;
    const play = mode === "play";
    for (const m of MODES) this.modeButtons[m].classList.toggle("on", m === mode);
    this.episode.hidden = mode !== "recorded";
    this.regime.hidden = mode !== "live";
    this.newFlight.hidden = mode !== "live";
    // In the sandbox and in Play each bird has its controller and its look, in their panel. Play
    // mixes the tasks and has no end, so no task menu and no timeline either.
    this.task.hidden = play;
    this.policy.hidden = mode === "sandbox" || play;
    $<HTMLButtonElement>("model").hidden = mode === "sandbox" || play;
    $<HTMLSpanElement>("timeline").hidden = play;
    this.sandbox.show(mode === "sandbox" || play, play);
    $<HTMLDivElement>("about").textContent = ABOUT[mode];
    if (!play) return this.fillTasks();
    // On a phone the panel would cover the scene one plays in: folded the first time (its + opens it).
    if (!this.playSeen && window.innerWidth < 600) this.folds.sandbox?.(true);
    this.playSeen = true;
    this.sandbox.setControllers(this.playControllers());
    this.choosePlay();
  }

  /**
   * Play's controllers: PID, LQR and every network, of every task, named with its task, the
   * go-to-point ones first (trained to reach one target, the closest to this game).
   */
  private playControllers(): [string, string][] {
    const rankOf = (t: string | null) => (t === null ? PLAY_ORDER.length : PLAY_ORDER.indexOf(t));
    const nets = [...this.networks].sort((a, b) => rankOf(a.task) - rankOf(b.task) || a.name.localeCompare(b.name));
    const options: [string, string][] = [
      ["pid", "PID"],
      ["lqr", "LQR"],
      ...nets.map((n): [string, string] => [n.name, `${n.task ? (TASK_NAMES[n.task] ?? n.task) : "?"} · ${label(n.name)}`]),
    ];
    this.playLabels = new Map(options);
    return options;
  }

  private choosePlay(): void {
    this.h.play({
      pilots: this.sandbox.slots.map((s) => ({ ...this.pilot(s.controller), id: s.id })),
      followed: this.sandbox.followed,
      conditions: this.sandbox.conditions(),
      seesTarget: this.sandbox.seesTarget,
    });
  }

  private fillTasks(): void {
    const tasks = this.mode !== "recorded" ? [...TASKS] : [...new Set(this.entries.map((e) => e.task))];
    tasks.sort((a, b) => Object.keys(TASK_NAMES).indexOf(a) - Object.keys(TASK_NAMES).indexOf(b));
    keep(this.task, tasks.map((t) => new Option(TASK_NAMES[t] ?? t, t)));
    this.fillPolicies(true);
  }

  /** The controllers available for the chosen task; a new task gets a new live seed. */
  private fillPolicies(newTask: boolean, fly = true): void {
    const task = this.task.value;
    const names =
      this.mode !== "recorded"
        ? ["pid", "lqr", ...this.networks.filter((n) => n.task === task).map((n) => n.name)]
        : [...new Set(this.entries.filter((e) => e.task === task).map((e) => e.policy))];
    names.sort(byRank);
    keep(this.policy, names.map((p) => new Option(label(p), p)), counterpart); // same role on a new task
    if (this.mode === "sandbox") this.sandbox.setControllers(names.map((p) => [p, label(p)]));
    if (!fly) return;
    if (this.mode === "live") this.chooseLive(newTask);
    else if (this.mode === "sandbox") {
      if (newTask) this.sandbox.start(task, this.h.current(), this.policy.value);
      this.chooseSandbox(newTask);
    } else this.fillEpisodes();
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

  private pilot(name: string): Pilot {
    const network = this.networks.find((n) => n.name === name);
    return { controller: name, file: network ? network.file : null, task: network ? network.task : null };
  }

  private chooseSandbox(resetCamera: boolean): void {
    const ready = this.sandbox.check(); // false for a navigation course without waypoints yet
    this.h.sandbox({
      task: this.task.value,
      pilots: ready ? this.sandbox.slots.map((s) => this.pilot(s.controller)) : [],
      followed: this.sandbox.followed,
      scene: this.sandbox.scene,
      conditions: this.sandbox.conditions(),
      seed: this.sandbox.seed,
      resetCamera,
    });
  }

  /**
   * The line about the episode on screen, and the timeline. `end`: the timeline's length, which
   * in the sandbox is the longest of the birds' flights, not necessarily this one's; `birds`: how
   * many fly (the sandbox), `followed`: which of them flies `ep`.
   */
  showEpisode(ep: Episode, weight: number, end: number, birds = 1, followed = 0): void {
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
    const which =
      ep.suite === "live" || ep.suite === "sandbox" ? `${ep.suite} flight, seed ${ep.seed}` : `${ep.suite ?? ""} episode ${ep.seed}`;
    const who = birds > 1 ? `${birds} birds, following ${dot(followed)} ${label(ep.policy ?? "?")}` : label(ep.policy ?? "?");
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
    const head = ep.result.reason === "no_course" ? "no course yet" : `${who} · ${which}`;
    this.info.innerHTML =
      `<b>${TASK_NAMES[ep.task] ?? ep.task}</b> · ${head}` + `<span class="more"><br>${air}${mass}${weak}${failure}</span>`;
    this.playInfo = ""; // Play's line was written over: write it again when Play comes back
    // Marks in fractions of the whole timeline. The failure is the same for every bird, so it is
    // marked as long as one of them is still flying then; the band starts where `ep` ended.
    this.slider.max = String(end);
    this.slider.step = "any"; // a flight can end between two decisions: the end must be reachable
    place(this.markFailure, c.failure_time !== null && c.failure_time < end ? c.failure_time / end : null);
    place(this.afterBand, ep.endFrame !== undefined ? ep.frames.t[ep.endFrame] / end : null);
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
    // Several birds: one line with the state of each, the followed one underlined; then the
    // numbers of the followed bird, as for a single bird.
    const first = v.birds.length
      ? v.birds
          .map((b, i) => {
            const name = b.followed ? `<u>${label(b.ep.policy ?? "?")}</u>` : label(b.ep.policy ?? "?");
            const state = b.ended ? `<b>${ending(b.ep.task, b.ep.result.success, b.ep.result.reason, false)}</b>` : "flying";
            // The followed bird's flight after its end is shown, not counted: say so next to it.
            const after = b.followed && v.after ? " <i>(after its end: shown, not counted)</i>" : "";
            return `${dot(i)} ${name} ${state}${after}`;
          })
          .join(" · ")
      : status;
    this.readout.innerHTML =
      `${first}<span class="more"><br>` +
      `altitude ${(100 * v.altitude).toFixed(1)} cm · speed ${(100 * v.velocity).toFixed(0)} cm/s · ` +
      `distance ${(100 * v.distance).toFixed(1)} cm · thrust ${v.thrustRatio.toFixed(2)} × weight` +
      (v.hold > 0 ? ` · holding ${Math.round(100 * v.hold)}%` : "") +
      `<br>wings: left ${deg(v.wings[0])}°, right ${deg(v.wings[1])}° of half-stroke, ${v.wings[2].toFixed(0)} Hz</span>`;
  }

  private playInfo = "";

  /** The HUD in Play: what can be done, the state of every bird, the numbers of the one followed. */
  updatePlay(v: PlayReadout): void {
    this.play.textContent = v.playing ? "Pause" : "Play";
    this.speedButtons.forEach((b, i) => b.classList.toggle("on", SPEEDS[i][0] === v.speed));
    this.clock.textContent = `${v.time.toFixed(1)} s`;
    const told = v.seesTarget ? "the birds chase the target" : "<b>the target is hidden from the birds</b>";
    const info =
      `<b>Play</b> · ${told}${v.following ? " · <b>the picked object follows the mouse</b> (Shift: up and down; F or Esc to stop)" : ""}` +
      `<span class="more"><br>drag the target or an obstacle · Shift + drag: up and down · F: the picked object follows the mouse</span>`;
    if (info !== this.playInfo) this.info.innerHTML = this.playInfo = info; // not rewritten 60 times a second
    const birds = v.birds
      .map((b, i) => {
        // With the task: two "PPO · normal · #1" of different tasks can fly together here.
        const words = this.playLabels.get(b.controller) ?? label(b.controller);
        const name = b.followed && v.birds.length > 1 ? `<u>${words}</u>` : words;
        const falls = b.falls ? ` (${b.falls} restart${b.falls > 1 ? "s" : ""})` : "";
        return `${dot(i)} ${name} ${b.state}${falls}`;
      })
      .join(" · ");
    this.readout.innerHTML =
      `${birds}<span class="more"><br>` +
      `altitude ${(100 * v.altitude).toFixed(1)} cm · speed ${(100 * v.velocity).toFixed(0)} cm/s · ` +
      (v.seesTarget ? `distance ${(100 * v.distance).toFixed(1)} cm · ` : "") +
      `thrust ${v.thrustRatio.toFixed(2)} × weight` +
      (v.seesTarget && v.hold > 0 ? ` · holding ${Math.round(100 * v.hold)}%` : "") +
      `<br>wings: left ${deg(v.wings[0])}°, right ${deg(v.wings[1])}° of half-stroke, ${v.wings[2].toFixed(0)} Hz</span>`;
  }

  private keyShown = "";
  private keyTime = 0;

  /**
   * The key of the air's colours, next to its menu: the palette and its current range (the
   * scales adapt to the flow in view). Its numbers are refreshed four times a second, to be read.
   */
  airKey(mode: AirMode, s: FluidScales): void {
    const key = $<HTMLSpanElement>("air-key");
    const now = performance.now();
    if (mode === this.keyShown && now - this.keyTime < 250) return;
    this.keyShown = mode;
    this.keyTime = now;
    key.hidden = mode === "off";
    if (mode === "off") return;
    const fmt = (x: number) => (x >= 10 ? x.toFixed(0) : x >= 1 ? x.toFixed(1) : x.toPrecision(2));
    const range: Record<string, [string, string]> = {
      smoke: ["smoke from the wings", ""],
      sheet: [`tracers in a slice; brighter = faster, up to ${fmt(s.speed)} m/s`, ""],
      tubes: ["vortices, coloured by speed: 0", `${fmt(s.speed)} m/s`],
      speed: ["0", `${fmt(s.speed)} m/s`],
      pressure: [`−${fmt(s.pressure)}`, `+${fmt(s.pressure)} Pa`],
      spin: ["0", `${fmt(s.vortices)} /s`],
    };
    const bar = key.querySelector<HTMLSpanElement>(".bar")!;
    const [lo, hi] = range[mode] ?? ["", ""];
    bar.style.background = paletteCss(mode);
    bar.hidden = mode === "smoke" || mode === "sheet";
    key.querySelector<HTMLSpanElement>(".lo")!.textContent = lo;
    key.querySelector<HTMLSpanElement>(".hi")!.textContent = hi;
  }

  /** Marks the camera button in use. */
  cameraMode(mode: CameraMode): void {
    document.querySelectorAll<HTMLButtonElement>("[data-camera]").forEach((b) => b.classList.toggle("on", b.dataset.camera === mode));
  }

  /** Enables the model button once the 3D model has loaded (or says it could not be). */
  modelReady(ok = true): void {
    const model = $<HTMLButtonElement>("model");
    model.disabled = !ok;
    model.dataset.on = ok ? "1" : "0";
    model.textContent = ok ? "3D model" : "Drawn bird";
    this.sandbox.setModelAvailable(ok);
  }

  message(html: string): void {
    this.info.innerHTML = html;
    this.playInfo = "";
  }
}

const deg = (rad: number) => ((rad * 180) / Math.PI).toFixed(0);

/** A dot in the colour of the sandbox's bird `i`. */
const dot = (i: number) => `<span style="color:#${BIRD_COLORS[i].toString(16).padStart(6, "0")}">●</span>`;

/**
 * Lets a panel be moved by dragging an empty part of it (not a button, a menu or a slider, which
 * keep their own use); a double click on an empty part puts it back in its place. The panel is
 * kept on screen. While dragged it gets a fixed width, since its place was set by both its sides.
 */
function draggable(box: HTMLElement): void {
  const controls = "button, select, input, a, label, option, textarea";
  box.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || (e.target as HTMLElement).closest(controls)) return;
    e.preventDefault(); // no text selection while dragging
    const r = box.getBoundingClientRect();
    const dx = e.clientX - r.left;
    const dy = e.clientY - r.top;
    const cs = getComputedStyle(box); // the width set is the content's: without the padding
    const width = r.width - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    Object.assign(box.style, { width: `${width}px`, maxWidth: "none", left: `${r.left}px`, top: `${r.top}px`, right: "auto", bottom: "auto" });
    box.classList.add("dragging");
    box.setPointerCapture(e.pointerId);
    const move = (m: PointerEvent) => {
      box.style.left = `${Math.min(Math.max(m.clientX - dx, 0), window.innerWidth - 60)}px`;
      box.style.top = `${Math.min(Math.max(m.clientY - dy, 0), window.innerHeight - 40)}px`;
    };
    const end = () => {
      box.classList.remove("dragging");
      box.dataset.moved = "1";
      box.removeEventListener("pointermove", move);
      box.removeEventListener("pointerup", end);
      box.removeEventListener("pointercancel", end);
    };
    box.addEventListener("pointermove", move);
    box.addEventListener("pointerup", end);
    box.addEventListener("pointercancel", end);
  });
  box.addEventListener("dblclick", (e) => {
    if ((e.target as HTMLElement).closest(controls)) return;
    for (const p of ["width", "max-width", "left", "top", "right", "bottom"]) box.style.removeProperty(p);
    delete box.dataset.moved;
  });
}

/** Puts a timeline mark (or the start of the band) at fraction `f` of the slider's length, or hides it (null). */
function place(mark: HTMLElement, f: number | null): void {
  mark.hidden = f === null;
  if (f !== null) mark.style.left = `calc(8px + (100% - 16px) * ${f.toFixed(4)})`;
}

/**
 * New options for a menu, keeping its choice if it is still offered. Otherwise `instead` may name
 * another value to show (the counterpart of a controller on a new task); else the first option.
 */
function keep(select: HTMLSelectElement, options: HTMLOptionElement[],
              instead?: (before: string, values: string[]) => string | undefined): void {
  const before = select.value;
  select.replaceChildren(...options);
  const values = options.map((o) => o.value);
  const value = values.includes(before) ? before : instead?.(before, values);
  if (value !== undefined) select.value = value;
}
