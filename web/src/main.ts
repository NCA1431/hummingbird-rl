/**
 * The viewer: the hummingbird, its targets and its trail, in two modes.
 *
 * Recorded: episodes listed in public/episodes/index.json, written by
 *     uv run python scripts/export_web.py episodes --policy <controller> --tasks ...
 * Live: a new episode simulated in the browser (modes/live.ts) and flown by
 * the PID, the LQR or one of the networks listed in public/policies/index.json,
 * written by
 *     uv run python scripts/export_web.py policy --policy <run name>
 * Both are then played by the same machinery (modes/watch.ts).
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import type { Controller } from "./control/controller";
import { LQRController } from "./control/lqr";
import { type MlpPolicy, type PolicyEntry, loadPolicy } from "./control/mlp";
import { PIDController } from "./control/pid";
import { simulateEpisode } from "./modes/live";
import { type Episode, type IndexEntry, Playback, damageAt, duration, endTime, sampleEpisode } from "./modes/watch";
import { decodeAction } from "./sim/aero";
import { DEFAULT_PARAMS, weight } from "./sim/params";
import { type CameraMode, type LiveChoice, Panel } from "./ui/panel";
import { Bird } from "./view/bird";
import { Stage } from "./view/stage";
import { World } from "./view/world";

const p = DEFAULT_PARAMS;
const stage = new Stage(document.querySelector<HTMLCanvasElement>("#view")!);
const bird = new Bird(p);
const world = new World();
stage.scene.add(bird.root, world.root);

let episode: Episode | null = null;
let playback: Playback | null = null;
let phase = 0.0; // wingbeat phase, integrated from the recorded frequency
let follow = true;
let eye = false; // the bird's-eye camera, kept from one episode to the next
let showWind = true;

/** Puts an episode on screen, recorded or live. */
function show(ep: Episode): void {
  episode = ep;
  playback = new Playback(ep);
  playback.speed = keepSpeed;
  phase = 0.0;
  world.load(ep);
  panel.showEpisode(ep, weight(p) * ep.conditions.mass_factor);
  setCamera(eye ? "eye" : ep.task === "obstacles" || ep.task === "navigation" ? "overview" : "follow");
}

// Networks are downloaded once (about 0.4 MB each) and kept.
const networks = new Map<string, Promise<MlpPolicy>>();
function controllerFor(c: LiveChoice): Promise<Controller> | Controller {
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
    show(ep);
  },
  async live(c: LiveChoice) {
    const mine = ++request;
    try {
      const controller = await controllerFor(c);
      if (mine !== request) return;
      show(simulateEpisode(c.task, c.regime, controller, c.seed));
    } catch (e) {
      panel.message(`Could not fly ${c.controller}: ${(e as Error).message}`);
    }
  },
  playPause() {
    if (!playback) return;
    if (!playback.playing && playback.time >= duration(playback.episode)) playback.restart();
    else playback.playing = !playback.playing;
  },
  restart: () => playback?.restart(),
  speed(s: number) {
    keepSpeed = s;
    if (playback) playback.speed = s;
  },
  seek: (t: number) => playback?.seek(t),
  camera: (mode: CameraMode) => setCamera(mode),
  scale: (s: number) => bird.setScale(s),
  wind: (show: boolean) => (showWind = show),
  model: (on: boolean) => bird.showModel(on),
});
let keepSpeed = 1 / 4;

/** The bird's heading (rad about the vertical), from its orientation (w, x, y, z). */
function headingOf(q: number[]): number {
  return new THREE.Euler().setFromQuaternion(new THREE.Quaternion(q[1], q[2], q[3], q[0]), "YZX").y;
}

/** Camera placements relative to the bird (or the whole scene for "overview"). */
function setCamera(mode: CameraMode): void {
  if (!episode || !playback) return;
  const s = sampleEpisode(episode, playback.time);
  const at = new THREE.Vector3(...s.pos);
  eye = mode === "eye";
  bird.setHidden(eye);
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
  const s = sampleEpisode(episode, playback.time);
  const cmd = decodeAction(s.wingAction, p);
  phase = (phase + 2 * Math.PI * cmd.freq * simDt) % (2 * Math.PI);
  // Wingbeats per rendered frame: above about a tenth, the eye (and a camera) only sees a blur.
  const beatsPerFrame = playback.playing ? cmd.freq * playback.speed * dt : 0;
  const blur = THREE.MathUtils.clamp((beatsPerFrame - 0.08) / 0.25, 0, 1);

  bird.setPose(s.pos, s.quat);
  bird.setWings(phase, cmd, blur, damageAt(episode, playback.time));
  bird.advanceModel(cmd.freq * simDt);
  const w = weight(p) * episode.conditions.mass_factor;
  world.update(s.target, s.index, s.pos, s.wind, w, showWind);
  if (follow) stage.follow(new THREE.Vector3(...s.pos));
  if (eye) {
    // At the eye: 1.5 cm ahead of the centre of mass and 8 mm above it, along the heading.
    const yaw = headingOf(s.quat);
    const ahead = new THREE.Vector3(0.015, 0.008, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    stage.eye(new THREE.Vector3(...s.pos).add(ahead), yaw);
  }

  const c = episode.conditions;
  panel.update(episode, {
    time: playback.time,
    duration: duration(episode),
    speed: playback.speed,
    playing: playback.playing,
    ended: playback.time >= endTime(episode) - 1e-9,
    after: episode.endFrame !== undefined && playback.time > endTime(episode) + 1e-9,
    altitude: s.pos[1],
    velocity: Math.hypot(...s.vel),
    distance: s.distance,
    thrustRatio: s.thrust / w,
    wings: [cmd.ampLeft, cmd.ampRight, cmd.freq],
    hold: s.hold,
    failed: c.failure_time !== null && playback.time >= c.failure_time,
  });
});

// The 3D model (1.1 MB, plus its 0.5 MB specular map) arrives after the page. The bird
// stays hidden until then, otherwise the drawn bird flashes for a few frames at every
// reload. If the model fails, or takes more than 3 s (slow connection), the drawn bird flies.
bird.root.visible = false;
const showBird = () => (bird.root.visible = true);
setTimeout(showBird, 3000);
const modelFailed = () => {
  showBird();
  panel.modelReady(false);
};
new THREE.TextureLoader().load(
  "models/hummingbird_specgloss.webp",
  (specGloss) => {
    specGloss.flipY = false; // glTF textures are stored the other way up from three.js's default
    specGloss.wrapS = specGloss.wrapT = THREE.RepeatWrapping;
    new GLTFLoader().load(
      "models/hummingbird.glb",
      (gltf) => {
        bird.attachModel(gltf, specGloss, stage.sunDirection);
        showBird();
        panel.modelReady();
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
