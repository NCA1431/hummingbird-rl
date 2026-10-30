/**
 * The sandbox's controls: tools to build the scene, the birds to fly it, and the conditions.
 *
 * A panel on the right of the page, built here rather than in index.html
 * because most of it is a list of sliders. Every change calls `changed`, and
 * the page flies the episode again: a slider acts when it is released, not
 * while it moves, so that one drag is one flight. Help stays out of the way:
 * a one-line hint at the top until it is closed (remembered by the browser),
 * and an (i) next to each section that explains it when the mouse rests on it,
 * or, on a phone where nothing "rests", when it is tapped.
 */

import { newSeed } from "../modes/live";
import {
  DEFAULT_SETTINGS,
  MAX_BIRDS,
  MAX_WAYPOINTS,
  type SandboxSettings,
  type Tool,
  canErase,
  clearScene,
  erase,
  place,
  sandboxConditions,
  sceneProblems,
  settingsFrom,
  toolsFor,
} from "../modes/sandbox";
import type { Episode } from "../modes/watch";
import type { Conditions } from "../sim/disturbances";
import { DEFAULT_PARAMS, type Vec3 } from "../sim/params";
import { Rng } from "../sim/random";
import { sampleScene } from "../sim/sampling";
import type { Scene } from "../sim/world";
import { BIRD_COLORS } from "../view/bird";
import { counterpart } from "./words";

const p = DEFAULT_PARAMS;
const HINT_KEY = "hummingbird-rl.sandbox-hint-closed";

/** One bird of the sandbox: who flies it, and whether it is drawn as the 3D model. */
export interface BirdSlot {
  controller: string;
  model: boolean;
}

export interface SandboxHandlers {
  changed(resetCamera: boolean): void; // fly again
  tool(tool: Tool): void;
  looks(slots: BirdSlot[]): void; // 3D model or drawn bird, without flying again
  follow(index: number): void; // which bird the camera and the HUD follow
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, text = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text) e.textContent = text;
  return e;
}

const css = (color: number) => `#${color.toString(16).padStart(6, "0")}`;

/**
 * What the Scene section says, task by task: what can be placed there, and how many. One text
 * for all tasks read as if every task had a course of waypoints.
 */
function sceneHelp(task: string): string {
  const see = "The see-through ball shows where the next one goes, and how high (the height slider).";
  const nearest = "The controllers only see the 5 nearest obstacles, but the bird can hit any of them.";
  switch (task) {
    case "hover":
      return "Hover & recover has nothing to place: the target is the start point (the grey ring), " +
        "where the bird is pushed at the start and must come back. Set the wind and the bird below.";
    case "goto":
      return `Go-to-point has one target and no obstacles. Target, then a click on the ground, moves it. ${see}`;
    case "obstacles":
      return "One target, and as many obstacles as you like. Target moves the target, Obstacle adds a sphere " +
        `(its size: the obstacle size slider), Erase removes the obstacle clicked, Clear removes them all. ${see} ${nearest}`;
    default:
      return `A course of up to ${MAX_WAYPOINTS} waypoints, flown in order (their numbers), among obstacles. ` +
        "Target adds the next waypoint, Obstacle adds a sphere, Erase removes the waypoint or obstacle clicked, " +
        `Clear removes them all to build a course from scratch. ${see} ${nearest}`;
  }
}

/** Changes the explanation of a section made by `section`. */
function setHelp(box: HTMLDivElement, help: string): void {
  box.querySelector<HTMLButtonElement>(".info")!.title = help;
  box.querySelector<HTMLDivElement>(".help")!.textContent = help;
}

/** A section title with an (i): its explanation shows on hover, and stays open under the title once clicked. */
function section(title: string, help: string): HTMLDivElement {
  const box = el("div");
  const head = el("div", { class: "section" }, title);
  const info = el("button", { class: "info", title: help, "aria-label": `About ${title}` }, "i");
  const text = el("div", { class: "help" }, help);
  text.hidden = true;
  info.onclick = () => (text.hidden = !text.hidden);
  head.append(info);
  box.append(head, text);
  return box;
}

export class SandboxPanel {
  readonly root = document.getElementById("sandbox") as HTMLDivElement;
  settings: SandboxSettings = structuredClone(DEFAULT_SETTINGS) as SandboxSettings;
  scene: Scene = { start: [0, 0.5, 0], targets: [[0, 0.5, 0]], obstacles: [] };
  task = "hover";
  tool: Tool = null;
  seed = newSeed();
  height = 0.5; // m, height of what is placed next
  radius = 0.07; // m, radius of the next obstacle
  slots: BirdSlot[] = [{ controller: "pid", model: true }];
  followed = 0;
  private modelAvailable = true; // false if the 3D model could not be loaded
  private controllers: [string, string][] = [["pid", "PID"], ["lqr", "LQR"]]; // (value, label)
  private readonly birdsBox: HTMLDivElement;
  private readonly toolButtons: Record<"target" | "obstacle" | "erase", HTMLButtonElement>;
  private readonly clearButton: HTMLButtonElement;
  private readonly randomButton: HTMLButtonElement;
  private readonly sceneSection: HTMLDivElement;
  private readonly heightRow: HTMLLabelElement;
  private readonly sizeRow: HTMLLabelElement;
  private readonly notes: HTMLDivElement;
  private readonly sliders: (() => void)[] = []; // refreshers, after the settings change from outside

  constructor(private readonly h: SandboxHandlers) {
    const body = this.root.querySelector<HTMLDivElement>(".sandbox-body")!;

    // The hint, until it is closed once.
    let closed = false;
    try {
      closed = localStorage.getItem(HINT_KEY) === "1";
    } catch {
      // storage unavailable (private window): the hint simply shows again
    }
    if (!closed) {
      const hint = el("div", { class: "hint" });
      hint.append("Choose Target or Obstacle, then click on the ground; drag to turn the view. Every change flies again.");
      const x = el("button", { class: "close", title: "Hide this hint" }, "×");
      x.onclick = () => {
        hint.remove();
        try {
          localStorage.setItem(HINT_KEY, "1");
        } catch {
          // nothing to remember it with
        }
      };
      hint.append(x);
      body.append(hint);
    }

    // Scene. Its explanation and the tools' tooltips depend on the task (updateTools).
    this.sceneSection = section("Scene", sceneHelp(this.task));
    body.append(this.sceneSection);
    const tools = el("div", { class: "seg" });
    this.toolButtons = { target: el("button", {}, "Target"), obstacle: el("button", {}, "Obstacle"), erase: el("button", {}, "Erase") };
    for (const t of ["target", "obstacle", "erase"] as const) this.toolButtons[t].onclick = () => this.setTool(this.tool === t ? null : t);
    const clear = el("button", {}, "Clear");
    this.clearButton = clear;
    clear.onclick = () => {
      this.scene = clearScene(this.scene, this.task);
      this.h.changed(false);
    };
    const random = el("button", {}, "Random scene");
    this.randomButton = random;
    random.onclick = () => {
      this.scene = sampleScene(this.task, new Rng(newSeed()), p.r_bird);
      this.h.changed(true);
    };
    tools.append(this.toolButtons.target, this.toolButtons.obstacle, this.toolButtons.erase);
    const edits = el("div", { class: "row" });
    edits.append(clear, random);
    body.append(tools, edits);
    const cm = (v: number) => `${Math.round(100 * v)} cm`;
    this.heightRow = this.slider(body, "height", 0.1, 1.0, 0.01, () => this.height, (v) => (this.height = v), cm, "height of the next target or obstacle", false);
    this.sizeRow = this.slider(body, "obstacle size", 0.03, 0.15, 0.005, () => this.radius, (v) => (this.radius = v), (v) => `r ${cm(v)}`, "radius of the next obstacle", false);
    // What is wrong with the scene (or full), right under the tools that can change it.
    this.notes = el("div", { class: "label sandbox-notes" });
    body.append(this.notes);

    // Birds.
    body.append(
      section(
        "Birds",
        "Up to 4 controllers fly the same situation together: same scene, same wind and gusts, same start. " +
          "The colour square tells them apart (their trail has the same colour). 3D/drawn changes the look; " +
          "◎ makes the camera and the top panel follow that bird.",
      ),
    );
    this.birdsBox = el("div", { class: "birds" });
    body.append(this.birdsBox);
    const again = el("button", { title: "Same scene and conditions, a new start push and new noise" }, "Fly again");
    again.onclick = () => {
      this.seed = newSeed();
      this.h.changed(false);
    };
    const add = el("button", { class: "add", title: "Add a bird" }, "+ bird");
    add.onclick = () => {
      if (this.slots.length >= MAX_BIRDS) return;
      const used = new Set(this.slots.map((s) => s.controller));
      const next = this.controllers.find(([v]) => !used.has(v)) ?? this.controllers[0];
      this.slots.push({ controller: next[0], model: false });
      this.renderBirds();
      this.h.changed(false);
    };
    const row = el("div", { class: "row" });
    row.append(add, again);
    body.append(row);

    // Conditions.
    const s = this.settings;
    const pct = (v: number) => `${Math.round(100 * v)}%`;
    const lost = (v: number) => (v > 0 ? `−${pct(v)}` : "intact");
    body.append(
      section(
        "Wind",
        "A steady push on the bird, as a percentage of the normal bird's weight, from the direction set " +
          "(the blue arrow shows it), plus gusts that vary around it. At 30% the bird must tilt by about 17° to stay in place.",
      ),
    );
    this.slider(body, "strength", 0, 0.4, 0.01, () => s.wind, (v) => (s.wind = v), pct, "in % of the normal bird's weight");
    this.slider(body, "direction", 0, 355, 5, () => s.windDir, (v) => (s.windDir = v), (v) => `${Math.round(v)}°`);
    this.slider(body, "gusts", 0, 0.1, 0.005, () => s.gusts, (v) => (s.gusts = v), pct);
    body.append(
      section(
        "Bird",
        "Mass: a lighter or heavier bird (the controllers are not told). Left/right wing: force lost by that wing " +
          "from the start, as with worn feathers; the damaged wing turns red.",
      ),
    );
    this.slider(body, "mass", 0.6, 1.4, 0.05, () => s.mass, (v) => (s.mass = v), (v) => `×${v.toFixed(2)}`);
    this.slider(body, "left wing", 0, 0.6, 0.05, () => s.damage[0], (v) => (s.damage[0] = v), lost, "force lost from the start");
    this.slider(body, "right wing", 0, 0.6, 0.05, () => s.damage[1], (v) => (s.damage[1] = v), lost, "force lost from the start");

    body.append(section("Sudden failure", "One wing suddenly loses part of its force during the flight, at the time set (red tick on the timeline)."));
    const wing = el("select", { "aria-label": "Sudden failure" });
    wing.append(new Option("none", "-1"), new Option("left wing", "0"), new Option("right wing", "1"));
    wing.onchange = () => {
      s.failureWing = Number(wing.value) as -1 | 0 | 1;
      this.h.changed(false);
    };
    this.sliders.push(() => (wing.value = String(s.failureWing)));
    body.append(wing);
    this.slider(body, "at", 0.2, 4.5, 0.1, () => s.failureTime, (v) => (s.failureTime = v), (v) => `${v.toFixed(1)} s`);
    this.slider(body, "force lost", 0.1, 0.6, 0.05, () => s.failureAmount, (v) => (s.failureAmount = v), pct);

    const noise = el("input", { type: "checkbox" });
    noise.onchange = () => {
      s.noise = noise.checked;
      this.h.changed(false);
    };
    this.sliders.push(() => (noise.checked = s.noise));
    const noiseLabel = el("label", {
      class: "label",
      title: "Sensor noise: small errors in the 48 numbers the controllers receive. Actuator noise: the wings do not do exactly what is commanded.",
    });
    noiseLabel.append(noise, " sensor and actuator noise");
    body.append(noiseLabel);

    this.refresh();
    this.renderBirds();
  }

  /** A labelled slider; it acts on release. `fly` false: it only sets what the next click places. */
  private slider(
    parent: HTMLElement,
    name: string,
    min: number,
    max: number,
    step: number,
    get: () => number,
    set: (v: number) => void,
    show: (v: number) => string,
    tip = "",
    fly = true,
  ): HTMLLabelElement {
    const row = el("label", { class: "slider", title: tip });
    const input = el("input", { type: "range", min: String(min), max: String(max), step: String(step) });
    const value = el("span", { class: "value" });
    input.oninput = () => (value.textContent = show(Number(input.value)));
    input.onchange = () => {
      set(Number(input.value));
      if (fly) this.h.changed(false);
    };
    row.append(el("span", { class: "name" }, name), input, value);
    parent.append(row);
    this.sliders.push(() => {
      input.value = String(get());
      value.textContent = show(get());
    });
    return row;
  }

  /** One row per bird: colour, controller, look, followed by the camera, remove. */
  private renderBirds(): void {
    this.birdsBox.replaceChildren(
      ...this.slots.map((slot, i) => {
        const row = el("div", { class: "bird" });
        const chip = el("span", { class: "chip", title: `bird ${i + 1}` });
        chip.style.background = css(BIRD_COLORS[i]);
        const pick = el("select", { "aria-label": `Controller of bird ${i + 1}` });
        pick.append(...this.controllers.map(([v, l]) => new Option(l, v)));
        pick.value = slot.controller;
        pick.onchange = () => {
          slot.controller = pick.value;
          this.h.changed(false);
        };
        const model = slot.model && this.modelAvailable;
        const look = el("button", { title: "3D model or drawn bird (the drawn bird shows the commanded wing angles)" }, model ? "3D" : "drawn");
        look.disabled = !this.modelAvailable;
        look.onclick = () => {
          slot.model = !slot.model;
          look.textContent = slot.model ? "3D" : "drawn";
          this.h.looks(this.slots);
        };
        const eye = el("button", { title: "The camera and the top panel follow this bird" }, "◎");
        eye.classList.toggle("on", i === this.followed);
        eye.onclick = () => {
          this.followed = i;
          this.renderBirds();
          this.h.follow(i);
        };
        row.append(chip, pick, look);
        if (this.slots.length > 1) row.append(eye); // with one bird there is nothing to choose
        if (i > 0) {
          const remove = el("button", { title: "Remove this bird" }, "×");
          remove.onclick = () => {
            this.slots.splice(i, 1);
            // The birds after it move up one place (and take the next colour up).
            if (this.followed === i) this.followed = 0;
            else if (this.followed > i) this.followed -= 1;
            this.renderBirds();
            this.h.follow(this.followed);
            this.h.changed(false);
          };
          row.append(remove);
        }
        return row;
      }),
    );
    const add = this.root.querySelector<HTMLButtonElement>("button.add");
    if (add) add.disabled = this.slots.length >= MAX_BIRDS;
  }

  /** Once the 3D model has loaded, or failed to: without it every bird is drawn. */
  setModelAvailable(ok: boolean): void {
    this.modelAvailable = ok;
    this.renderBirds();
  }

  private refresh(): void {
    for (const r of this.sliders) r();
    this.updateTools();
  }

  /**
   * Greys out the tools that cannot act on this scene, and drops the current one if it is among
   * them (its preview with it): e.g. Target once a navigation course has all its waypoints.
   */
  private updateTools(): void {
    const task = this.task;
    const tools = toolsFor(task, this.scene);
    for (const t of ["target", "obstacle", "erase"] as const) this.toolButtons[t].disabled = !tools[t];
    this.clearButton.disabled = !tools.clear;
    const nav = task === "navigation";
    const b = this.toolButtons;
    b.target.title =
      task === "hover"
        ? "Hover has no target to place: it is the start point"
        : !nav
          ? "Move the target: click on the ground"
          : tools.target
            ? "Add the next waypoint: click on the ground"
            : `A course has ${MAX_WAYPOINTS} waypoints at most: Erase one to place another`;
    b.obstacle.title = tools.obstacle ? "Add an obstacle: click on the ground" : "No obstacles in this task";
    b.erase.title = !tools.erase
      ? "Nothing to erase in this task (Target moves the target)"
      : nav
        ? "Click a waypoint or an obstacle to remove it"
        : "Click an obstacle to remove it (Target moves the target)";
    this.randomButton.disabled = task === "hover";
    this.randomButton.title = task === "hover" ? "Hover always has the same scene" : "A new random scene for this task";
    this.clearButton.title = !tools.clear ? "Nothing to clear in this task" : nav ? "Remove all the waypoints and obstacles" : "Remove all the obstacles";
    setHelp(this.sceneSection, sceneHelp(task));
    // Only the sliders that act on something placed in this task.
    this.heightRow.style.display = task === "hover" ? "none" : "";
    this.sizeRow.style.display = tools.obstacle ? "" : "none";
    if (this.tool && !tools[this.tool]) this.setTool(null);
  }

  setTool(tool: Tool): void {
    this.tool = tool;
    for (const t of ["target", "obstacle", "erase"] as const) this.toolButtons[t].classList.toggle("on", tool === t);
    this.h.tool(tool);
  }

  /**
   * A new task. Coming from an episode of that task (the one on screen when the sandbox opens),
   * we start from its scene, its conditions and its controller (`controller`, flown by the first
   * bird); otherwise from a random scene, and the birds keep their controllers.
   */
  start(task: string, ep: Episode | null, controller: string): void {
    this.task = task;
    if (ep && ep.task === task && ep.frames.t.length > 1) {
      this.scene = structuredClone(ep.scene);
      Object.assign(this.settings, settingsFrom(ep.conditions, p)); // the same object: the sliders hold it
      this.seed = ep.seed;
      this.slots[0].controller = controller;
    } else {
      this.scene = sampleScene(task, new Rng(newSeed()), p.r_bird);
    }
    this.refresh();
    this.renderBirds();
  }

  /** A click on the ground at (x, z) with the Target or Obstacle tool. */
  placeAt(x: number, z: number): void {
    if (this.tool !== "target" && this.tool !== "obstacle") return;
    this.scene = place(this.scene, this.task, this.tool, [x, this.height, z] as Vec3, this.radius);
    this.h.changed(false);
  }

  /** A click on a target or an obstacle with the Erase tool. */
  eraseItem(kind: "target" | "obstacle", index: number): void {
    if (this.tool !== "erase" || !canErase(this.task, kind)) return;
    this.scene = erase(this.scene, kind, index);
    this.h.changed(false);
  }

  /**
   * The controllers that can fly this task, as (value, label). A bird keeps its controller if it is
   * still offered, or takes its counterpart on the new task (same algorithm, variant and seed), so
   * that a comparison set up on one task survives a change of task; PID otherwise.
   */
  setControllers(options: [string, string][]): void {
    this.controllers = options;
    const values = options.map(([v]) => v);
    for (const s of this.slots) s.controller = counterpart(s.controller, values) ?? values[0];
    this.renderBirds();
  }

  conditions(): Conditions {
    return sandboxConditions(this.settings, p);
  }

  /** What prevents a meaningful flight, shown under the controls; true if the flight can go. */
  check(): boolean {
    this.updateTools();
    const problems = sceneProblems(this.scene, this.task, p.r_bird);
    if (this.task === "navigation" && this.scene.targets.length >= MAX_WAYPOINTS) {
      problems.push(`${MAX_WAYPOINTS} waypoints, the most a course can have (Erase removes one)`);
    }
    this.notes.textContent = problems.join(" · ");
    return !(this.task !== "hover" && this.scene.targets.length === 0);
  }

  show(visible: boolean): void {
    this.root.hidden = !visible;
    if (!visible) this.setTool(null);
  }
}
