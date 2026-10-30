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
 * Play is the exception: the birds are stepped while the page runs (modes/play.ts),
 * since the visitor moves the target and the obstacles during the flight.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";

import type { Controller } from "./control/controller";
import { LQRController } from "./control/lqr";
import { type MlpPolicy, type PolicyEntry, loadPolicy } from "./control/mlp";
import { PIDController } from "./control/pid";
import { newSeed, simulateEpisode } from "./modes/live";
import { type PlayPilot, PlaySession, defaultPlayScene, fallWords } from "./modes/play";
import { canErase, inside, waitingEpisode } from "./modes/sandbox";
import { type Episode, type IndexEntry, Playback, damageAt, duration, endTime, sampleEpisode } from "./modes/watch";
import { decodeAction } from "./sim/aero";
import { DEFAULT_PARAMS, type Vec3, weight } from "./sim/params";
import type { Quat } from "./sim/rotations";
import { HOLD_RADIUS } from "./sim/tasks";
import { clearance } from "./sim/world";
import { type CameraMode, type LiveChoice, Panel, type Pilot, type PlayChoice, type SandboxChoice } from "./ui/panel";
import type { PlayEdit } from "./ui/sandboxPanel";
import { Air, type AirBird, type AirMode } from "./view/air";
import { BIRD_COLORS, Bird } from "./view/bird";
import { Stage } from "./view/stage";
import { type Pick, World } from "./view/world";

const p = DEFAULT_PARAMS;
const stage = new Stage(document.querySelector<HTMLCanvasElement>("#view")!);
// Four birds, one per colour. Outside the sandbox only the first one flies.
const birds = BIRD_COLORS.map((color) => new Bird(p, color));
const world = new World();
const air = new Air(p); // the smoke simulation around the followed bird ("air" menu)
stage.scene.add(...birds.map((b) => b.root), world.root, air.root);
stage.beforeRender.push(() => air.depthPass(stage.renderer, stage.scene, stage.camera));

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

// Play: the session (kept when another mode is opened, so that coming back finds the birds where
// they were), whether the world on screen is its scene, its own pause, what is picked, and whether
// the target follows the mouse.
let session: PlaySession | null = null;
let worldIsPlay = false;
let playOn = true;
let picked: Pick | null = { kind: "target", index: 0 };
let followMouse = false;
let playedOnce = false;

/**
 * Puts episodes on screen, one per bird (several only in the sandbox), following bird
 * `followIndex`. `resetCamera` false keeps the view where it is: the sandbox flies again at
 * every edit. The playback lasts as long as the longest of them.
 */
function show(eps: Episode[], resetCamera = true, followIndex = 0): void {
  worldIsPlay = false;
  episodes = eps;
  followed = Math.min(followIndex, eps.length - 1);
  playback = new Playback(eps[followed], Math.max(...eps.map(duration)));
  playback.speed = keepSpeed;
  phases = eps.map(() => 0.0);
  air.reset();
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
  air.setObstacles(episode.scene.obstacles); // the air flows around them
  world.setOthers(episodes.map((ep, i) => ({ ep, color: BIRD_COLORS[i] })).filter((_, i) => i !== followed));
  // Conditions in % of the normal bird's weight; the timeline as long as the longest flight.
  panel.showEpisode(episode, weight(p), playback!.end, episodes.length, followed);
}

/** Which birds are seen, and how: the 3D model or the drawn bird. */
function refreshBirds(): void {
  const play = panel.mode === "play" && worldIsPlay && session !== null;
  const n = play ? session!.birds.length : episodes.length;
  birds.forEach((b, i) => {
    b.root.visible = birdsReady && i < n;
    b.showModel(panel.mode === "sandbox" || play ? (panel.sandbox.slots[i]?.model ?? false) : modelOn);
    b.setHidden(eye && i === followed); // the bird's-eye camera is inside the followed bird
  });
}

/** PID and LQR see the obstacles; a network only if it was trained with obstacles, as in its training. */
function seesObstacles(c: Pilot): boolean {
  return c.file === null || c.task === "obstacles" || c.task === "navigation";
}

/** Play: the session's scene on screen, the camera over it. */
function enterPlay(): void {
  const se = session!;
  worldIsPlay = true;
  episodes = [];
  episode = null;
  playback = null;
  phases = birds.map(() => 0.0);
  world.loadPlay(se.scene);
  world.playTargetSeen(se.seesTarget);
  pick(picked);
  air.reset();
  air.setObstacles(se.scene.obstacles);
  if (!playedOnce) {
    // Real time, the first time: one moves things by hand and watches the birds answer.
    playedOnce = true;
    keepSpeed = 1;
  }
  refreshBirds();
  setCamera(eye ? "eye" : "overview");
}

/** Play: `p` is picked (it glows, and the panel shows its height and size). */
function pick(p: Pick | null): void {
  const se = session;
  if (p?.kind === "obstacle" && (!se || p.index >= se.scene.obstacles.length)) p = null;
  picked = p;
  world.playSelect(p);
  if (!se || !p) return panel.sandbox.showSelection(null);
  const pos = p.kind === "target" ? se.scene.targets[0] : se.scene.obstacles[p.index].center;
  const radius = p.kind === "target" ? HOLD_RADIUS : se.scene.obstacles[p.index].radius;
  panel.sandbox.showSelection({ kind: p.kind, index: p.index, height: pos[1], radius });
}

/**
 * Play: moves the target or an obstacle to `pos` (and resizes an obstacle), inside the arena and
 * above the ground: the target between 5 cm and 1.1 m, an obstacle resting on the ground at worst.
 */
function moveTo(p: Pick, pos: Vec3, radius?: number): void {
  const se = session!;
  if (p.kind === "target") {
    const q = inside([pos[0], Math.min(Math.max(pos[1], 0.05), 1.1), pos[2]]);
    se.moveTarget(q);
    world.playMove(p, q);
  } else {
    const r = radius ?? se.scene.obstacles[p.index].radius;
    const q = inside([pos[0], Math.min(Math.max(pos[1], r), 1.2 - r), pos[2]]);
    se.moveObstacle(p.index, q, r);
    world.playMove(p, q, r);
    air.setObstacles(se.scene.obstacles); // the air flows around them where they are now
  }
  if (picked && picked.kind === p.kind && picked.index === p.index) pick(p); // the sliders follow
}

/**
 * Play: where a new obstacle goes: 15 to 35 cm from the start, clear of every bird and of the
 * target, so that adding one never knocks a bird down. The last try is kept if none is clear.
 */
function freeSpot(se: PlaySession, radius: number): Vec3 {
  let spot: Vec3 = [0, 0.5, 0];
  for (let k = 0; k < 40; k++) {
    const a = Math.random() * 2 * Math.PI;
    const d = 0.15 + 0.2 * Math.random();
    spot = [se.scene.start[0] + d * Math.cos(a), 0.35 + 0.3 * Math.random(), se.scene.start[2] + d * Math.sin(a)];
    const o = { center: spot, radius };
    const clear =
      se.birds.every((b) => clearance(b.state.pos, o, p.r_bird) > 0.06) &&
      clearance(se.scene.targets[0], o, HOLD_RADIUS) > 0.04 &&
      se.scene.obstacles.every((x) => Math.hypot(...x.center.map((c, i) => c - spot[i])) > x.radius + radius + 0.02);
    if (clear) break;
  }
  return spot;
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
  async play(c: PlayChoice) {
    const mine = ++request;
    try {
      const controllers = await Promise.all(c.pilots.map(controllerFor));
      if (mine !== request) return;
      const pilots: PlayPilot[] = controllers.map((controller, i) => ({
        id: c.pilots[i].id ?? i,
        controller,
        seesObstacles: seesObstacles(c.pilots[i]),
      }));
      if (!session) session = new PlaySession(defaultPlayScene(), c.conditions, pilots, newSeed());
      else {
        session.setPilots(pilots);
        session.setConditions(c.conditions);
      }
      session.seesTarget = c.seesTarget;
      followed = Math.min(c.followed, pilots.length - 1);
      if (!worldIsPlay) enterPlay();
      refreshBirds();
    } catch (e) {
      panel.message(`Could not fly: ${(e as Error).message}`);
    }
  },
  playEdit(e: PlayEdit) {
    const se = session;
    if (!se) return;
    switch (e.kind) {
      case "add": {
        const radius = 0.06;
        se.addObstacle({ center: freeSpot(se, radius), radius });
        world.loadPlay(se.scene);
        world.playTargetSeen(se.seesTarget);
        pick({ kind: "obstacle", index: se.scene.obstacles.length - 1 });
        air.setObstacles(se.scene.obstacles);
        break;
      }
      case "remove":
        if (picked?.kind !== "obstacle") return;
        se.removeObstacle(picked.index);
        world.loadPlay(se.scene);
        world.playTargetSeen(se.seesTarget);
        pick({ kind: "target", index: 0 });
        air.setObstacles(se.scene.obstacles);
        break;
      case "height":
        if (!picked) return;
        {
          const at = picked.kind === "target" ? se.scene.targets[0] : se.scene.obstacles[picked.index].center;
          moveTo(picked, [at[0], e.value, at[2]]);
        }
        break;
      case "size":
        if (picked?.kind === "obstacle") moveTo(picked, se.scene.obstacles[picked.index].center, e.value);
        break;
      case "seeTarget":
        se.seesTarget = e.on;
        world.playTargetSeen(e.on);
        break;
      case "follow":
        followMouse = e.on;
        break;
      case "restart":
        se.restartAll();
        air.reset();
        break;
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
    if (panel.mode === "play") {
      if (session && index < session.birds.length) followed = index;
      refreshBirds();
      return;
    }
    if (index >= episodes.length) return; // not flown yet: the next flight follows it
    followed = index;
    loadWorld();
    refreshBirds();
  },
  playPause() {
    if (panel.mode === "play") {
      playOn = !playOn;
      return;
    }
    if (!playback) return;
    if (!playback.playing && playback.time >= playback.end) playback.restart();
    else playback.playing = !playback.playing;
  },
  restart() {
    if (panel.mode === "play") {
      session?.restartAll();
      air.reset();
      return;
    }
    playback?.restart();
    air.reset(); // the air shed later in the flight does not belong at its start
  },
  speed(s: number) {
    keepSpeed = s;
    if (playback) playback.speed = s;
  },
  seek(t: number) {
    playback?.seek(t);
    air.reset();
  },
  camera: (mode: CameraMode) => setCamera(mode),
  scale(s: number) {
    birds.forEach((b) => b.setScale(s));
  },
  wind: (show: boolean) => (showWind = show),
  air: (mode: AirMode) => air.setMode(mode),
  airBox: (on: boolean) => air.showBox(on),
  trails: (on: boolean) => world.setTrails(on),
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

/**
 * What the camera frames: the followed bird (its position and orientation now) and the points of
 * the scene. From the Play session in Play, from the episode on screen otherwise.
 */
function focus(): { pos: Vec3; quat: Quat; points: Vec3[] } | null {
  const scenePoints = (sc: { start: Vec3; targets: Vec3[]; obstacles: { center: Vec3 }[] }) => [
    sc.start,
    ...sc.targets,
    ...sc.obstacles.map((o) => o.center),
  ];
  if (panel.mode === "play") {
    if (!session || !worldIsPlay) return null;
    const s = session.sample(followed);
    return { pos: s.pos, quat: s.quat, points: scenePoints(session.scene) };
  }
  if (!episode || !playback) return null;
  const s = sampleEpisode(episode, playback.time);
  return { pos: s.pos, quat: s.quat, points: scenePoints(episode.scene) };
}

/** Camera placements relative to the followed bird (or the whole scene for "overview"). */
function setCamera(mode: CameraMode): void {
  const s = focus();
  if (!s) return;
  const at = new THREE.Vector3(...s.pos);
  eye = mode === "eye";
  panel.cameraMode(mode);
  birds.forEach((b, i) => b.setHidden(eye && i === followed));
  follow = mode !== "overview" && !eye;
  if (eye) return; // placed at every frame, in the loop below
  if (mode === "overview") {
    // Frame the start, the targets and the obstacles together.
    const box = new THREE.Box3().setFromPoints(s.points.map((q) => new THREE.Vector3(...q)));
    const centre = box.getCenter(new THREE.Vector3());
    // Play's scene is small but things get moved around it: a wider view, about the arena's middle.
    const radius = Math.max(0.5 * box.getSize(new THREE.Vector3()).length(), panel.mode === "play" ? 0.45 : 0.2);
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

/**
 * One frame of Play: the session runs on by the real time elapsed × the speed, and every bird is
 * drawn between its last two decisions. A bird that has fallen blinks until it starts again.
 */
function framePlay(dt: number): void {
  const se = session!;
  const simDt = playOn ? dt * keepSpeed : 0;
  se.advance(simDt);
  const realBeat = playOn ? keepSpeed * dt : 0;
  const blink = Math.floor(performance.now() / 120) % 2 === 0;
  const damage = se.conditions.damage;
  const samples = se.birds.map((b, i) => {
    const s = se.sample(i);
    const cmd = decodeAction(s.action, p);
    const still = b.down !== null;
    const dtWings = still ? 0 : simDt;
    phases[i] = (phases[i] + 2 * Math.PI * cmd.freq * dtWings) % (2 * Math.PI);
    const blur = still ? 0 : THREE.MathUtils.clamp((cmd.freq * realBeat - 0.08) / 0.25, 0, 1);
    birds[i].setPose(s.pos, s.quat);
    birds[i].setWings(phases[i], cmd, blur, damage);
    birds[i].advanceModel(cmd.freq * dtWings);
    birds[i].root.visible = birdsReady && (!still || blink);
    return { s, cmd, still };
  });
  for (let i = samples.length; i < birds.length; i++) birds[i].root.visible = false;
  followed = Math.min(followed, samples.length - 1);
  const f = samples[followed];
  if (!f) return;

  const w = weight(p) * se.conditions.mass_factor;
  const wind = se.wind;
  world.updateWind(f.s.pos, wind, w, showWind);
  world.playTrails(se.birds.map((b) => b.trail));
  air.update(
    simDt,
    f.s.pos,
    wind,
    samples.map((x, i): AirBird => ({
      pos: x.s.pos,
      quat: x.s.quat,
      thrust: x.s.thrust,
      cmd: x.cmd,
      damage,
      phase: phases[i],
      flying: !x.still,
    })),
    stage.camera.position.toArray() as [number, number, number],
  );
  panel.airKey(air.current, air.scales);
  if (follow) stage.follow(new THREE.Vector3(...f.s.pos));
  if (eye) {
    const yaw = headingOf(f.s.quat);
    const ahead = new THREE.Vector3(0.015, 0.008, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    stage.eye(new THREE.Vector3(...f.s.pos).add(ahead), yaw);
  }

  panel.updatePlay({
    time: se.time,
    speed: keepSpeed,
    playing: playOn,
    seesTarget: se.seesTarget,
    following: followMouse,
    birds: se.birds.map((b, i) => ({
      controller: b.pilot.controller.name,
      state: b.down ? `<b>${fallWords(b.down.reason)}</b>, starting again` : "flying",
      falls: b.falls,
      followed: i === followed,
    })),
    altitude: f.s.pos[1],
    velocity: Math.hypot(...f.s.vel),
    distance: f.s.distance,
    thrustRatio: f.s.thrust / w,
    wings: [f.cmd.ampLeft, f.cmd.ampRight, f.cmd.freq],
    hold: f.s.hold,
  });
}

stage.run((dt) => {
  if (panel.mode === "play") {
    if (session && worldIsPlay) framePlay(dt);
    return;
  }
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
    return { s, cmd, still, damage: damageAt(ep, t), phase: phases[i] };
  });
  const { s, cmd } = samples[followed];

  // One wind arrow, above the followed bird: the wind is the same everywhere in the arena (the
  // gusts vary in time, not in space), so every bird feels the one it shows.
  const w = weight(p) * episode.conditions.mass_factor;
  world.update(s.target, s.index, s.pos, s.wind, w, showWind);
  air.update(
    simDt,
    s.pos,
    s.wind,
    samples.map((x): AirBird => ({
      pos: x.s.pos,
      quat: x.s.quat,
      thrust: x.s.thrust,
      cmd: x.cmd,
      damage: x.damage,
      phase: x.phase,
      flying: !x.still,
    })),
    stage.camera.position.toArray() as [number, number, number],
  );
  panel.airKey(air.current, air.scales);
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
 * What Erase would remove under the pointer: the nearest target or obstacle the ray passes within
 * 2 cm of, among those the task lets it remove (the one target of go-to-point and obstacles is
 * moved, not removed). The margin matters: a target is a ball of 3 cm, a few pixels wide in the overview.
 */
function pickAt(e: PointerEvent): Pick | null {
  const ray = rayAt(e);
  let best: Pick | null = null;
  let nearest = Infinity;
  for (const o of world.pickables) {
    if (!canErase(panel.sandbox.task, (o.userData.pick as Pick).kind)) continue;
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

// Play: press on the target or an obstacle and drag it. It slides horizontally at its height;
// with Shift held, it moves up and down instead (the vertical distance under the pointer, at the
// object's distance from the camera). Pressing anywhere else turns the camera, as usual. On a
// touch screen the finger is the pointer, so dragging works the same, with a wider margin around
// the object since a fingertip covers many pixels.
let drag: {
  pick: Pick;
  pointerId: number;
  plane: THREE.Plane; // the horizontal plane the object slides in
  offset: THREE.Vector3; // from the point under the pointer to the object's centre
  vertical: boolean;
  y0: number;
  clientY0: number;
} | null = null;

/** The centre and radius of the target or obstacle `k`. */
function playItem(k: Pick): { center: Vec3; radius: number } {
  const sc = session!.scene;
  return k.kind === "target" ? { center: sc.targets[0], radius: HOLD_RADIUS } : sc.obstacles[k.index];
}

/** What the pointer is on in Play, measured on the screen: the nearest object within a margin, or null. */
function pickPlay(e: PointerEvent): Pick | null {
  if (!session) return null;
  const rect = canvas.getBoundingClientRect();
  const margin = e.pointerType === "touch" ? 28 : 8; // px around the drawn ball
  const half = THREE.MathUtils.degToRad(stage.camera.fov / 2);
  const all: Pick[] = [{ kind: "target", index: 0 }, ...session.scene.obstacles.map((_, i): Pick => ({ kind: "obstacle", index: i }))];
  let best: Pick | null = null;
  let nearest = Infinity;
  for (const k of all) {
    const { center, radius } = playItem(k);
    const c = new THREE.Vector3(...center);
    const ndc = c.clone().project(stage.camera);
    if (ndc.z > 1) continue; // behind the camera
    const x = rect.left + ((ndc.x + 1) / 2) * rect.width;
    const y = rect.top + ((1 - ndc.y) / 2) * rect.height;
    const pixels = (radius / (c.distanceTo(stage.camera.position) * Math.tan(half))) * (rect.height / 2);
    const gap = Math.hypot(e.clientX - x, e.clientY - y) - pixels;
    if (gap < margin && gap < nearest) {
      nearest = gap;
      best = k;
    }
  }
  return best;
}

/** Re-anchors the drag where the object is now (at the start, and when Shift is pressed or released). */
function anchor(d: NonNullable<typeof drag>, e: PointerEvent): void {
  const c = playItem(d.pick).center;
  d.vertical = e.shiftKey;
  d.y0 = c[1];
  d.clientY0 = e.clientY;
  d.plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -c[1]);
  const hit = rayAt(e).intersectPlane(d.plane, new THREE.Vector3());
  d.offset = hit ? new THREE.Vector3(...c).sub(hit) : new THREE.Vector3();
}

const inPlay = () => panel.mode === "play" && session !== null && worldIsPlay;

// Capturing, so that this runs before the camera's own controls and can keep them from turning.
canvas.addEventListener(
  "pointerdown",
  (e) => {
    if (!inPlay()) return;
    const k = pickPlay(e);
    if (!k) return; // the camera turns
    e.stopImmediatePropagation();
    e.preventDefault();
    pick(k);
    drag = { pick: k, pointerId: e.pointerId, plane: new THREE.Plane(), offset: new THREE.Vector3(), vertical: false, y0: 0, clientY0: 0 };
    anchor(drag, e);
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
  },
  { capture: true },
);
canvas.addEventListener("pointermove", (e) => {
  if (!inPlay()) return;
  const se = session!;
  if (drag && e.pointerId === drag.pointerId) {
    if (e.shiftKey !== drag.vertical) anchor(drag, e);
    const c = playItem(drag.pick).center;
    if (drag.vertical) {
      const rect = canvas.getBoundingClientRect();
      const distance = new THREE.Vector3(...c).distanceTo(stage.camera.position);
      const perPixel = (2 * distance * Math.tan(THREE.MathUtils.degToRad(stage.camera.fov / 2))) / rect.height;
      moveTo(drag.pick, [c[0], drag.y0 + (drag.clientY0 - e.clientY) * perPixel, c[2]]);
    } else {
      const hit = rayAt(e).intersectPlane(drag.plane, new THREE.Vector3());
      if (hit) moveTo(drag.pick, [hit.x + drag.offset.x, drag.y0, hit.z + drag.offset.z]);
    }
    return;
  }
  if (followMouse && e.pointerType === "mouse" && e.buttons === 0) {
    // The target goes under the mouse, at its own height.
    const t = se.scene.targets[0];
    const hit = rayAt(e).intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -t[1]), new THREE.Vector3());
    if (hit) moveTo({ kind: "target", index: 0 }, [hit.x, t[1], hit.z]);
    return;
  }
  if (e.buttons === 0) canvas.style.cursor = pickPlay(e) ? "grab" : "";
});
const endDrag = (e: PointerEvent) => {
  if (!drag || e.pointerId !== drag.pointerId) return;
  drag = null;
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  canvas.style.cursor = "";
};
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

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

