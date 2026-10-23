/**
 * The viewer: the hummingbird, its targets and its trail, in three modes.
 *
 * Recorded: episodes listed in public/episodes/index.json, written by
 *     uv run python scripts/export_web.py episodes --policy <controller> --tasks ...
 * Live: a new episode simulated in the browser (modes/live.ts) and flown by
 * the PID, the LQR or one of the networks listed in public/policies/index.json,
 * written by
 *     uv run python scripts/export_web.py policy --policy <run name>
 * Sandbox: the same, in a scene and conditions set by hand (modes/sandbox.ts),
 * flown by up to four controllers at once, each bird in its own colour.
 * All are then played by the same machinery (modes/watch.ts).
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";

import type { Controller } from "./control/controller";
import { LQRController } from "./control/lqr";
import { type MlpPolicy, type PolicyEntry, loadPolicy } from "./control/mlp";
import { PIDController } from "./control/pid";
import { simulateEpisode } from "./modes/live";
import { inside, waitingEpisode } from "./modes/sandbox";
import { type Episode, type IndexEntry, Playback, damageAt, duration, endTime, sampleEpisode } from "./modes/watch";
import { decodeAction } from "./sim/aero";
import { DEFAULT_PARAMS, weight } from "./sim/params";
import { type CameraMode, type LiveChoice, Panel, type Pilot, type SandboxChoice } from "./ui/panel";
import { BIRD_COLORS, Bird } from "./view/bird";
import { Stage } from "./view/stage";
import { type Pick, World } from "./view/world";

const p = DEFAULT_PARAMS;
const stage = new Stage(document.querySelector<HTMLCanvasElement>("#view")!);
// Four birds, one per colour. Outside the sandbox only the first one flies.
const birds = BIRD_COLORS.map((color) => new Bird(p, color));
const world = new World();
stage.scene.add(...birds.map((b) => b.root), world.root);

let episodes: Episode[] = []; // one per bird on screen, in the order of their colours
let followed = 0; // the bird the camera and the HUD follow
let episode: Episode | null = null; // its episode, episodes[followed]
let playback: Playback | null = null;
let phases = [0.0]; // wingbeat phase of each bird, integrated from its frequency
let follow = true;
let eye = false; // the bird's-eye camera, kept from one episode to the next
let showWind = true;
let modelOn = true; // the "3D model" button of the controls (the sandbox sets each bird's look)
let birdsReady = false; // the birds stay hidden until the 3D model has arrived (see the end)

/**
 * Puts episodes on screen, one per bird (several only in the sandbox), following bird
 * `followIndex`. `resetCamera` false keeps the view where it is: the sandbox flies again at
 * every edit. The playback lasts as long as the longest of them.
 */
function show(eps: Episode[], resetCamera = true, followIndex = 0): void {
  episodes = eps;
  followed = Math.min(followIndex, eps.length - 1);
  playback = new Playback(eps[followed], Math.max(...eps.map(duration)));
  playback.speed = keepSpeed;
  phases = eps.map(() => 0.0);
  loadWorld();
  refreshBirds();
  if (resetCamera) setCamera(eye ? "eye" : episode!.task === "obstacles" || episode!.task === "navigation" ? "overview" : "follow");
}

/**
 * The world as seen from the followed bird: its progress through the targets, its wind arrow,
 * and its trail. With several birds every trail has its bird's colour (otherwise by speed).
 */
function loadWorld(): void {
  episode = episodes[followed];
  const several = episodes.length > 1;
  world.load(episode, several ? BIRD_COLORS[followed] : null);
  world.setOthers(episodes.map((ep, i) => ({ ep, color: BIRD_COLORS[i] })).filter((_, i) => i !== followed));
  // Conditions in % of the normal bird's weight; the timeline as long as the longest flight.
  panel.showEpisode(episode, weight(p), playback!.end, episodes.length, followed);
}

/** Which birds are seen, and how: the 3D model or the drawn bird. */
function refreshBirds(): void {
  birds.forEach((b, i) => {
    b.root.visible = birdsReady && i < episodes.length;
    b.showModel(panel.mode === "sandbox" ? (panel.sandbox.slots[i]?.model ?? false) : modelOn);
    b.setHidden(eye && i === followed); // the bird's-eye camera is inside the followed bird
  });
}

// Networks are downloaded once (about 0.4 MB each) and kept.
const networks = new Map<string, Promise<MlpPolicy>>();
function controllerFor(c: Pilot): Promise<Controller> | Controller {
  if (c.controller === "pid") return new PIDController(p);
  if (c.controller === "lqr") return new LQRController(p);
  if (!c.file) throw new Error(`no file for ${c.controller}`);
  if (!networks.has(c.file)) networks.set(c.file, loadPolicy(c.file));
  return networks.get(c.file)!;
}

// Only the last request counts: a slow download must not replace a newer choice.
let request = 0;

const panel = new Panel({
  async select(entry: IndexEntry) {
    const mine = ++request;
    const ep = (await (await fetch(`episodes/${entry.file}`)).json()) as Episode;
    if (mine !== request) return;
    ep.policy = entry.policy;
    show([ep]);
  },
  async live(c: LiveChoice) {
    const mine = ++request;
    try {
      const controller = await controllerFor(c);
      if (mine !== request) return;
      show([simulateEpisode(c.task, c.regime, controller, c.seed)]);
    } catch (e) {
      panel.message(`Could not fly ${c.controller}: ${(e as Error).message}`);
    }
  },
  async sandbox(c: SandboxChoice) {
    const mine = ++request;
    const resetCamera = c.resetCamera || !episode;
    if (c.pilots.length === 0) {
      show([waitingEpisode(c.task, c.scene, c.conditions)], resetCamera);
      return;
    }
    try {
      const controllers = await Promise.all(c.pilots.map(controllerFor));
      if (mine !== request) return;
      // The same seed for all: the same start push and the same gusts, so that the birds meet
      // exactly the same situation (env.ts keeps these apart from the sensor noise).
      const options = { scene: c.scene, conditions: c.conditions };
      show(controllers.map((ctrl) => simulateEpisode(c.task, "normal", ctrl, c.seed, options, "sandbox")), resetCamera, c.followed);
    } catch (e) {
      panel.message(`Could not fly: ${(e as Error).message}`);
    }
  },
  current: () => episode,
  tool(tool) {
    stage.renderer.domElement.style.cursor = tool ? "crosshair" : "";
    if (tool !== "target" && tool !== "obstacle") world.setGhost(null, null);
  },
  looks: () => refreshBirds(),
  follow(index: number) {
    // The camera modes that follow a bird move to the new one by themselves (stage.follow below).
    if (index >= episodes.length) return; // not flown yet: the next flight follows it
    followed = index;
    loadWorld();
    refreshBirds();
  },
  playPause() {
    if (!playback) return;
    if (!playback.playing && playback.time >= playback.end) playback.restart();
    else playback.playing = !playback.playing;
  },
  restart: () => playback?.restart(),
  speed(s: number) {
    keepSpeed = s;
    if (playback) playback.speed = s;
  },
  seek: (t: number) => playback?.seek(t),
  camera: (mode: CameraMode) => setCamera(mode),
  scale: (s: number) => birds.forEach((b) => b.setScale(s)),
  wind: (show: boolean) => (showWind = show),
  model(on: boolean) {
    modelOn = on;
    refreshBirds();
  },
});
let keepSpeed = 1 / 4;

/** The bird's heading (rad about the vertical), from its orientation (w, x, y, z). */
function headingOf(q: number[]): number {
  return new THREE.Euler().setFromQuaternion(new THREE.Quaternion(q[1], q[2], q[3], q[0]), "YZX").y;
}

/** Camera placements relative to the followed bird (or the whole scene for "overview"). */
function setCamera(mode: CameraMode): void {
  if (!episode || !playback) return;
  const s = sampleEpisode(episode, playback.time);
  const at = new THREE.Vector3(...s.pos);
  eye = mode === "eye";
  birds.forEach((b, i) => b.setHidden(eye && i === followed));
  follow = mode !== "overview" && !eye;
  if (eye) return; // placed at every frame, in the loop below
  if (mode === "overview") {
    // Frame the start, the targets and the obstacles together.
    const box = new THREE.Box3().setFromPoints(
      [episode.scene.start, ...episode.scene.targets, ...episode.scene.obstacles.map((o) => o.center)].map(
        (q) => new THREE.Vector3(...q),
      ),
    );
    const centre = box.getCenter(new THREE.Vector3());
    const radius = Math.max(0.5 * box.getSize(new THREE.Vector3()).length(), 0.2);
    // Far enough for that sphere to fit in the narrower of the two fields of view, with a margin:
    // the HUD and the controls cover about a third of the screen.
    const half = THREE.MathUtils.degToRad(stage.camera.fov / 2);
    const fit = Math.min(half, Math.atan(Math.tan(half) * stage.camera.aspect));
    const distance = (1.45 * radius) / Math.sin(fit);
    stage.view(centre, new THREE.Vector3(0.6, 0.8, 1.0).normalize().multiplyScalar(distance));
    return;
  }
  const yaw = headingOf(s.quat);
  const offsets: Record<string, THREE.Vector3> = {
    follow: new THREE.Vector3(0.18, 0.12, 0.3),
    side: new THREE.Vector3(0, 0.02, 0.35),
    top: new THREE.Vector3(0.0001, 0.4, 0),
    behind: new THREE.Vector3(-0.35, 0.06, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw),
  };
  stage.view(at, offsets[mode]);
}

stage.run((dt) => {
  if (!episode || !playback) return;
  const simDt = playback.advance(dt);
  const t = playback.time;
  // Wingbeats per rendered frame: above about a tenth, the eye (and a camera) only sees a blur.
  const realBeat = playback.playing ? playback.speed * dt : 0;
  const samples = episodes.map((ep, i) => {
    const s = sampleEpisode(ep, t);
    const cmd = decodeAction(s.wingAction, p);
    // In the sandbox the playback lasts as long as the longest flight, so the others wait at their
    // last frame. A bird still flying then (a success, or the time limit) keeps beating its wings;
    // one that crashed or flew off lies still instead of flapping on the ground.
    const still = t >= duration(ep) && !ep.result.success && ep.result.reason !== "time_limit";
    const dtWings = still ? 0 : simDt;
    phases[i] = (phases[i] + 2 * Math.PI * cmd.freq * dtWings) % (2 * Math.PI);
    const blur = still ? 0 : THREE.MathUtils.clamp((cmd.freq * realBeat - 0.08) / 0.25, 0, 1);
    birds[i].setPose(s.pos, s.quat);
    birds[i].setWings(phases[i], cmd, blur, damageAt(ep, t));
    birds[i].advanceModel(cmd.freq * dtWings);
    return { s, cmd };
  });
  const { s, cmd } = samples[followed];

  // One wind arrow, above the followed bird: the wind is the same everywhere in the arena (the
  // gusts vary in time, not in space), so every bird feels the one it shows.
  const w = weight(p) * episode.conditions.mass_factor;
  world.update(s.target, s.index, s.pos, s.wind, w, showWind);
  world.updateOthers(samples.filter((_, i) => i !== followed).map((x) => x.s.index));
  if (follow) stage.follow(new THREE.Vector3(...s.pos));
  if (eye) {
    // At the eye: 1.5 cm ahead of the centre of mass and 8 mm above it, along the heading.
    const yaw = headingOf(s.quat);
    const ahead = new THREE.Vector3(0.015, 0.008, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    stage.eye(new THREE.Vector3(...s.pos).add(ahead), yaw);
  }

  const c = episode.conditions;
  panel.update(episode, {
    time: t,
    duration: playback.end,
    speed: playback.speed,
    playing: playback.playing,
    ended: t >= endTime(episode) - 1e-9,
    after: episode.endFrame !== undefined && t > endTime(episode) + 1e-9,
    altitude: s.pos[1],
    velocity: Math.hypot(...s.vel),
    distance: s.distance,
    thrustRatio: s.thrust / w,
    wings: [cmd.ampLeft, cmd.ampRight, cmd.freq],
    hold: s.hold,
    failed: c.failure_time !== null && t >= c.failure_time,
    birds:
      episodes.length > 1 ? episodes.map((ep, i) => ({ ep, ended: t >= endTime(ep) - 1e-9, followed: i === followed })) : [],
  });
});

// Sandbox: a click (not a drag, which turns the camera) places what the tool says, or erases
// what it is on. While the mouse moves, a see-through preview shows where and how high the next
// target or obstacle would go.
const canvas = stage.renderer.domElement;
const raycaster = new THREE.Raycaster();

function rayAt(e: PointerEvent): THREE.Ray {
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, stage.camera);
  return raycaster.ray;
}

/** Where the pointer meets the ground, or null (above the horizon). */
function groundAt(e: PointerEvent): THREE.Vector3 | null {
  return rayAt(e).intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), new THREE.Vector3());
}

/**
 * The target or obstacle under the pointer: the nearest one the ray passes within 2 cm of.
 * The margin matters: a target is a ball of 3 cm, a few pixels wide in the overview.
 */
function pickAt(e: PointerEvent): Pick | null {
  const ray = rayAt(e);
  let best: Pick | null = null;
  let nearest = Infinity;
  for (const o of world.pickables) {
    const geometry = (o as THREE.Mesh).geometry;
    if (!geometry.boundingSphere) geometry.computeBoundingSphere();
    if (ray.distanceToPoint(o.position) > geometry.boundingSphere!.radius + 0.02) continue;
    const along = o.position.clone().sub(ray.origin).dot(ray.direction);
    if (along > 0 && along < nearest) {
      nearest = along;
      best = o.userData.pick as Pick;
    }
  }
  return best;
}

let pressed: { x: number; y: number } | null = null;
canvas.addEventListener("pointerdown", (e) => (pressed = { x: e.clientX, y: e.clientY }));
canvas.addEventListener("pointerup", (e) => {
  const start = pressed;
  pressed = null;
  const tool = panel.sandbox.tool;
  if (!start || panel.mode !== "sandbox" || !tool) return;
  if (Math.hypot(e.clientX - start.x, e.clientY - start.y) > 6) return;
  if (tool === "erase") {
    const pick = pickAt(e);
    if (pick) panel.sandbox.eraseItem(pick.kind, pick.index);
  } else {
    const hit = groundAt(e);
    if (hit) panel.sandbox.placeAt(hit.x, hit.z);
  }
});
canvas.addEventListener("pointermove", (e) => {
  const tool = panel.sandbox.tool;
  if (panel.mode !== "sandbox" || !tool) return;
  if (tool === "erase") {
    canvas.style.cursor = pickAt(e) ? "pointer" : "crosshair";
    return;
  }
  const hit = groundAt(e);
  world.setGhost(tool, hit ? inside([hit.x, panel.sandbox.height, hit.z]) : null, panel.sandbox.radius);
});
canvas.addEventListener("pointerleave", () => world.setGhost(null, null));

// The 3D model (1.1 MB, plus its 0.5 MB specular map) arrives after the page. The birds
// stay hidden until then, otherwise the drawn bird flashes for a few frames at every
// reload. If the model fails, or takes more than 3 s (slow connection), the drawn birds fly.
const showBirds = () => {
  birdsReady = true;
  refreshBirds();
};
setTimeout(showBirds, 3000);
const modelFailed = () => {
  panel.modelReady(false);
  showBirds();
};
new THREE.TextureLoader().load(
  "models/hummingbird_specgloss.webp",
  (specGloss) => {
    specGloss.flipY = false; // glTF textures are stored the other way up from three.js's default
    specGloss.wrapS = specGloss.wrapT = THREE.RepeatWrapping;
    new GLTFLoader().load(
      "models/hummingbird.glb",
      (gltf) => {
        // Each bird needs its own skeleton, since its wings beat on their own: copies of the
        // loaded scene, all made before the first one is changed (the geometry and the textures
        // are shared, so the four cost little more than one). The first bird keeps its natural
        // colours; the others are tinted with theirs.
        const copies = birds.map((_, i) => (i === 0 ? gltf.scene : SkeletonUtils.clone(gltf.scene)));
        birds.forEach((b, i) =>
          b.attachModel(copies[i], gltf.animations, specGloss, stage.sunDirection, i === 0 ? null : BIRD_COLORS[i]),
        );
        panel.modelReady();
        showBirds();
      },
      undefined,
      modelFailed,
    );
  },
  undefined,
  modelFailed,
);

const getJson = <T>(url: string): Promise<T> =>
  fetch(url).then((r) => (r.ok ? (r.json() as Promise<T>) : Promise.reject(new Error(`${url}: ${r.status}`))));

// Without recorded episodes the page starts in live mode; without exported networks, live
// mode offers the PID and the LQR only.
getJson<{ episodes: IndexEntry[] }>("episodes/index.json")
  .then((index) => panel.setIndex(index.episodes))
  .catch(() => panel.setIndex([]));
getJson<{ policies: PolicyEntry[] }>("policies/index.json")
  .then((index) => panel.setPolicies(index.policies))
  .catch(() => panel.setPolicies([]));
