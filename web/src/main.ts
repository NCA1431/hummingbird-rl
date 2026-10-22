/**
 * The viewer: replays recorded episodes with the hummingbird, its targets and its trail.
 *
 * Episodes are listed in public/episodes/index.json and written by
 *     uv run python scripts/export_web.py episodes --policy <controller> --tasks ...
 * The live mode (the simulator and the controllers running in the browser) comes later.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import { type Episode, type IndexEntry, Playback, damageAt, duration, sampleEpisode } from "./modes/watch";
import { decodeAction } from "./sim/aero";
import { DEFAULT_PARAMS, weight } from "./sim/params";
import { type CameraMode, Panel } from "./ui/panel";
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
let showWind = true;

const panel = new Panel({
  async select(entry: IndexEntry) {
    const ep = (await (await fetch(`episodes/${entry.file}`)).json()) as Episode;
    ep.policy = entry.policy;
    episode = ep;
    playback = new Playback(ep);
    playback.speed = keepSpeed;
    phase = 0.0;
    world.load(ep);
    panel.showEpisode(ep, weight(p) * ep.conditions.mass_factor);
    setCamera(ep.task === "obstacles" || ep.task === "navigation" ? "overview" : "follow");
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

/** Camera placements relative to the bird (or the whole scene for "overview"). */
function setCamera(mode: CameraMode): void {
  if (!episode || !playback) return;
  const s = sampleEpisode(episode, playback.time);
  const at = new THREE.Vector3(...s.pos);
  follow = mode !== "overview";
  if (mode === "overview") {
    // Frame the start, the targets and the obstacles together.
    const box = new THREE.Box3().setFromPoints(
      [episode.scene.start, ...episode.scene.targets, ...episode.scene.obstacles.map((o) => o.center)].map(
        (q) => new THREE.Vector3(...q),
      ),
    );
    const centre = box.getCenter(new THREE.Vector3());
    const size = Math.max(box.getSize(new THREE.Vector3()).length(), 0.4);
    stage.view(centre, new THREE.Vector3(0.6, 0.8, 1.0).multiplyScalar(size));
    return;
  }
  const yaw = new THREE.Euler().setFromQuaternion(
    new THREE.Quaternion(s.quat[1], s.quat[2], s.quat[3], s.quat[0]), "YZX",
  ).y;
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
  const cmd = decodeAction(s.action, p);
  phase = (phase + 2 * Math.PI * cmd.freq * simDt) % (2 * Math.PI);
  // Wingbeats per rendered frame: above about a tenth, the eye (and a camera) only sees a blur.
  const beatsPerFrame = playback.playing ? cmd.freq * playback.speed * dt : 0;
  const blur = THREE.MathUtils.clamp((beatsPerFrame - 0.08) / 0.25, 0, 1);

  bird.setPose(s.pos, s.quat);
  bird.setWings(phase, cmd, blur, damageAt(episode, playback.time));
  bird.advanceModel(cmd.freq * simDt, blur);
  const w = weight(p) * episode.conditions.mass_factor;
  world.update(s.target, s.index, s.pos, s.wind, w, showWind);
  if (follow) stage.follow(new THREE.Vector3(...s.pos));

  const c = episode.conditions;
  panel.update(episode, {
    time: playback.time,
    duration: duration(episode),
    speed: playback.speed,
    playing: playback.playing,
    altitude: s.pos[1],
    velocity: Math.hypot(...s.vel),
    distance: s.distance,
    thrustRatio: s.thrust / w,
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

fetch("episodes/index.json")
  .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
  .then((index: { episodes: IndexEntry[] }) => panel.setIndex(index.episodes))
  .catch(() =>
    panel.message(
      "No recorded episodes yet. Write some with<br><code>uv run python scripts/export_web.py episodes --policy pid --tasks hover goto</code>",
    ),
  );
