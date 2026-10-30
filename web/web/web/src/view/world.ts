/**
 * What surrounds the bird in an episode: its targets, the obstacles, the start
 * point, the wind, and the trail it leaves.
 *
 * Targets are drawn as translucent balls of the radius the task uses: 3 cm for
 * a target to hold (the bird must stay inside for a quarter of a second) and
 * 4 cm for a navigation waypoint that only has to be passed. The current
 * target is bright, the ones already reached are faded, and navigation
 * waypoints carry their number. In the sandbox, the other birds leave trails
 * in their own colour, and a see-through "ghost" shows where the next target
 * or obstacle will go. In Play, the target and the obstacles move while the
 * birds fly: each stands on a dashed stem that gives its height, the one picked
 * glows, and the trails are the last few seconds of each bird's path.
 */

import * as THREE from "three";

import type { Episode } from "../modes/watch";
import type { Vec3 } from "../sim/params";
import type { Scene } from "../sim/world";
import { BIRD_COLORS } from "./bird";

const HOLD_RADIUS = 0.03; // m, as in tasks/base.py
const PASS_RADIUS = 0.04; // m, as in tasks/navigation.py
const WIND_SCALE = 12.8; // m of arrow per newton: a wind equal to the weight (0.039 N) is a 50 cm arrow
const WIND_HEIGHT = 0.07; // m, the arrow is drawn above the bird, clear of its body and wings
const TARGET_COLOR = 0xe8a33d;
const OBSTACLE_COLOR = 0x8a7a68;
const HIDDEN_COLOR = 0x9aa0a6; // Play: a target the birds are not told about, greyed out for us
const PLAY_TRAIL_POINTS = 600; // more than the 4 s of path kept (240 decisions)

/** What a click can pick in the sandbox: a target or an obstacle, by its index in the scene. */
export interface Pick {
  kind: "target" | "obstacle";
  index: number;
}

/** A dashed line from a point down to the ground, to read its height. */
function stem(height: number, color: number): THREE.Line {
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -height, 0)]),
    new THREE.LineDashedMaterial({ color, dashSize: 0.01, gapSize: 0.01 }),
  );
  line.computeLineDistances();
  line.userData.stem = true;
  return line;
}

/** The same stem at a new height (Play: the object has moved). */
function setStem(line: THREE.Line, height: number): void {
  line.geometry.setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -height, 0)]);
  line.computeLineDistances();
}

/** A small round label with a number, which always faces the camera. */
function numberLabel(n: number): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const g = canvas.getContext("2d")!;
  g.fillStyle = "rgba(255, 255, 255, 0.9)";
  g.beginPath();
  g.arc(32, 32, 28, 0, 2 * Math.PI);
  g.fill();
  g.fillStyle = "#7a4a00";
  g.font = "bold 38px system-ui, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(String(n), 32, 34);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthWrite: false }));
  sprite.scale.setScalar(0.035);
  return sprite;
}

export class World {
  readonly root = new THREE.Group();
  /** Targets and obstacles, for the sandbox's Erase tool (each carries a Pick in userData). */
  readonly pickables: THREE.Object3D[] = [];
  private targets: THREE.Mesh[] = [];
  private readonly wind: THREE.ArrowHelper;
  private trail: THREE.Line | null = null;
  private others: THREE.Line[] = []; // trails of the sandbox's other birds
  private showTrails = true;
  private readonly ghost = new THREE.Group();
  // Play: the target and the obstacles (a group each: the ball, scaled to its radius, and its stem),
  // and one trail per bird.
  private playTarget: THREE.Group | null = null;
  private playObstacles: THREE.Group[] = [];
  private playTrailLines: THREE.Line[] = [];

  constructor() {
    this.wind = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.1, 0x2b6cb0, 0.02, 0.012);
    this.root.add(this.wind);
    this.ghost.visible = false;
  }

  /**
   * Rebuilds everything for a new episode. `trailColor`: one colour for the whole trail (the
   * sandbox, with several birds, where each trail must say whose it is) instead of the speed.
   */
  load(ep: Episode, trailColor: number | null = null): void {
    this.clear(ep.scene.start);

    const n = ep.scene.targets.length;
    const numbered = ep.task === "navigation" && n > 1;
    ep.scene.targets.forEach((p, i) => {
      const last = i === n - 1;
      const radius = ep.task === "navigation" && !last ? PASS_RADIUS : HOLD_RADIUS;
      const ball = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 24, 16),
        new THREE.MeshStandardMaterial({ color: TARGET_COLOR, transparent: true, opacity: 0.3, depthWrite: false }),
      );
      ball.position.set(p[0], p[1], p[2]);
      ball.userData.pick = { kind: "target", index: i } satisfies Pick;
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.003, 10, 8), new THREE.MeshBasicMaterial({ color: 0xc26a00 }));
      ball.add(dot, stem(p[1], 0x9a7a40));
      if (numbered) {
        const label = numberLabel(i + 1);
        label.position.set(0, radius + 0.025, 0);
        ball.add(label);
      }
      this.targets.push(ball);
      this.pickables.push(ball);
      this.root.add(ball);
    });

    ep.scene.obstacles.forEach((o, i) => {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(o.radius, 32, 20),
        new THREE.MeshStandardMaterial({ color: OBSTACLE_COLOR, roughness: 0.85 }),
      );
      sphere.position.set(o.center[0], o.center[1], o.center[2]);
      sphere.castShadow = sphere.receiveShadow = true;
      sphere.userData.pick = { kind: "obstacle", index: i } satisfies Pick;
      this.pickables.push(sphere);
      this.root.add(sphere);
    });

    this.trail = makeTrail(ep, trailColor);
    this.trail.visible = this.showTrails;
    this.root.add(this.trail);
  }

  /** Empties the world but for the wind arrow, the ghost and the start ring. */
  private clear(startPoint: Vec3): void {
    this.root.clear();
    this.root.add(this.wind, this.ghost);
    this.targets = [];
    this.pickables.length = 0;
    this.others = [];
    this.trail = null;
    this.playTarget = null;
    this.playObstacles = [];
    this.playTrailLines = [];
    const start = new THREE.Mesh(
      new THREE.RingGeometry(0.012, 0.016, 32),
      new THREE.MeshBasicMaterial({ color: 0x555555, side: THREE.DoubleSide }),
    );
    start.rotation.x = -Math.PI / 2;
    start.position.set(startPoint[0], 0.001, startPoint[2]);
    this.root.add(start);
  }

  /**
   * Play: the scene of a session, whose target and obstacles move afterwards (playMove). Rebuilt
   * when an obstacle is added or removed.
   */
  loadPlay(scene: Scene): void {
    this.clear(scene.start);
    const target = new THREE.Group();
    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(HOLD_RADIUS, 24, 16),
      new THREE.MeshStandardMaterial({ color: 0xf0a020, transparent: true, opacity: 0.5, depthWrite: false }),
    );
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.003, 10, 8), new THREE.MeshBasicMaterial({ color: 0xc26a00 }));
    target.add(ball, dot, stem(1, 0x9a7a40));
    target.userData.pick = { kind: "target", index: 0 } satisfies Pick;
    this.playTarget = target;
    this.root.add(target);
    this.playObstacles = scene.obstacles.map((_, i) => {
      const g = new THREE.Group();
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(1, 32, 20), // unit sphere, scaled to the radius: resizing is cheap
        new THREE.MeshStandardMaterial({ color: OBSTACLE_COLOR, roughness: 0.85 }),
      );
      sphere.castShadow = sphere.receiveShadow = true;
      g.add(sphere, stem(1, 0x6b5f52));
      g.userData.pick = { kind: "obstacle", index: i } satisfies Pick;
      this.root.add(g);
      return g;
    });
    this.playMove({ kind: "target", index: 0 }, scene.targets[0]);
    scene.obstacles.forEach((o, i) => this.playMove({ kind: "obstacle", index: i }, o.center, o.radius));
    this.playTrailLines = BIRD_COLORS.map((color) => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * PLAY_TRAIL_POINTS), 3));
      geo.setDrawRange(0, 0);
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
      line.frustumCulled = false; // its bounds change at every frame
      line.visible = this.showTrails;
      this.root.add(line);
      return line;
    });
  }

  /** Play: the target or an obstacle at a new place (and, for an obstacle, a new radius). */
  playMove(pick: Pick, pos: Vec3, radius?: number): void {
    const g = pick.kind === "target" ? this.playTarget : this.playObstacles[pick.index];
    if (!g) return;
    g.position.set(pos[0], pos[1], pos[2]);
    for (const c of g.children) {
      if (c.userData.stem) setStem(c as THREE.Line, pos[1]);
      else if (pick.kind === "obstacle" && radius !== undefined && c instanceof THREE.Mesh) c.scale.setScalar(radius);
    }
  }

  /** Play: the picked object glows (null: none). */
  playSelect(pick: Pick | null): void {
    const glow = (g: THREE.Group | null, on: boolean) => {
      const m = (g?.children[0] as THREE.Mesh | undefined)?.material as THREE.MeshStandardMaterial | undefined;
      m?.emissive.set(on ? 0x5a3a10 : 0x000000);
    };
    glow(this.playTarget, pick?.kind === "target");
    this.playObstacles.forEach((g, i) => glow(g, pick?.kind === "obstacle" && pick.index === i));
  }

  /** Play: the target in colour when the birds are told about it, greyed out when they are not. */
  playTargetSeen(seen: boolean): void {
    const m = (this.playTarget?.children[0] as THREE.Mesh | undefined)?.material as THREE.MeshStandardMaterial | undefined;
    if (!m) return;
    m.color.set(seen ? 0xf0a020 : HIDDEN_COLOR);
    m.opacity = seen ? 0.5 : 0.25;
  }

  /** Play: the last seconds of each bird's path (bird i in colour i). */
  playTrails(paths: Vec3[][]): void {
    this.playTrailLines.forEach((line, i) => {
      const path = paths[i] ?? [];
      const n = Math.min(path.length, PLAY_TRAIL_POINTS);
      const attr = line.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let k = 0; k < n; k++) attr.setXYZ(k, path[path.length - n + k][0], path[path.length - n + k][1], path[path.length - n + k][2]);
      attr.needsUpdate = true;
      line.geometry.setDrawRange(0, n);
    });
  }

  /** Sandbox: a target or an obstacle dragged to `pos`, on screen until the flight is redone. */
  moveItem(pick: Pick, pos: Vec3): void {
    const o = this.pickables.find((x) => {
      const k = x.userData.pick as Pick;
      return k.kind === pick.kind && k.index === pick.index;
    });
    if (!o) return;
    o.position.set(pos[0], pos[1], pos[2]);
    for (const c of o.children) if (c.userData.stem) setStem(c as THREE.Line, pos[1]);
  }

  /** The trails of the other birds of the sandbox, each in its colour. */
  setOthers(others: { ep: Episode; color: number }[]): void {
    for (const t of this.others) this.root.remove(t);
    this.others = others.map(({ ep, color }) => {
      const line = makeTrail(ep, color);
      line.visible = this.showTrails;
      this.root.add(line);
      return line;
    });
  }

  /** The birds' paths shown or not. */
  setTrails(on: boolean): void {
    this.showTrails = on;
    for (const t of [this.trail, ...this.others, ...this.playTrailLines]) if (t) t.visible = on;
  }

  /** How much of each other trail is drawn (frames up to its index). */
  updateOthers(frameIndices: number[]): void {
    this.others.forEach((t, i) => t.geometry.setDrawRange(0, frameIndices[i] + 1));
  }

  /**
   * The sandbox's preview of the next placement at `pos` (null hides it): a target ball, or an
   * obstacle of the given radius, each with its stem to the ground.
   */
  setGhost(kind: "target" | "obstacle" | null, pos: Vec3 | null, radius = 0.05): void {
    this.ghost.visible = kind !== null && pos !== null;
    if (!kind || !pos) return;
    const r = kind === "target" ? HOLD_RADIUS : radius;
    const key = `${kind}:${r}:${pos[1]}`;
    if (this.ghost.userData.key !== key) {
      // Rebuilt only when the kind, the size or the height change, not at every mouse move.
      this.ghost.clear();
      const ball = new THREE.Mesh(
        new THREE.SphereGeometry(r, 24, 16),
        new THREE.MeshBasicMaterial({
          color: kind === "target" ? TARGET_COLOR : OBSTACLE_COLOR,
          transparent: true,
          opacity: 0.35,
          depthWrite: false,
        }),
      );
      this.ghost.add(ball, stem(pos[1], 0x555555));
      this.ghost.userData.key = key;
    }
    this.ghost.position.set(pos[0], pos[1], pos[2]);
  }

  /** Per-frame state: which target is current, the trail so far, the wind at the bird. */
  update(current: number, frameIndex: number, birdPos: Vec3, wind: Vec3, weight: number, showWind: boolean): void {
    this.targets.forEach((t, i) => {
      const m = t.material as THREE.MeshStandardMaterial;
      m.opacity = i === current ? 0.45 : i < current ? 0.08 : 0.2;
      m.color.set(i === current ? 0xf0a020 : 0xb0a080);
    });
    if (this.trail) this.trail.geometry.setDrawRange(0, frameIndex + 1);
    this.updateWind(birdPos, wind, weight, showWind);
  }

  /** The wind arrow above the bird followed. */
  updateWind(birdPos: Vec3, wind: Vec3, weight: number, showWind: boolean): void {
    const f = new THREE.Vector3(wind[0], wind[1], wind[2]);
    const size = f.length();
    this.wind.visible = showWind && size > 0.01 * weight;
    if (this.wind.visible) {
      // The head grows with the arrow, so that a light breeze is a small arrow, not a blue block.
      const length = WIND_SCALE * size;
      this.wind.position.set(birdPos[0], birdPos[1] + WIND_HEIGHT, birdPos[2]);
      this.wind.setDirection(f.normalize());
      this.wind.setLength(length, 0.3 * length, 0.15 * length);
    }
  }
}

/**
 * The path of the whole episode, coloured by speed (blue slow, red at 1 m/s and above) or in
 * one colour, and grey for what is shown after the episode's end (after a loss of control).
 */
function makeTrail(ep: Episode, color: number | null): THREE.Line {
  const f = ep.frames;
  const positions = new Float32Array(f.pos.length * 3);
  const colors = new Float32Array(f.pos.length * 3);
  const slow = new THREE.Color(0x2563eb);
  const fast = new THREE.Color(0xdc2626);
  const after = new THREE.Color(0x8a8f96);
  const plain = color === null ? null : new THREE.Color(color);
  const end = ep.endFrame ?? f.pos.length;
  const c = new THREE.Color();
  f.pos.forEach((p, i) => {
    positions.set(p, 3 * i);
    const v = f.vel[i];
    if (i > end) c.copy(after);
    else if (plain) c.copy(plain);
    else c.copy(slow).lerp(fast, Math.min(1, Math.hypot(v[0], v[1], v[2])));
    c.toArray(colors, 3 * i);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true }));
}
