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
 * or obstacle will go.
 */

import * as THREE from "three";

import type { Vec3 } from "../sim/params";
import type { Episode } from "../modes/watch";

const HOLD_RADIUS = 0.03; // m, as in tasks/base.py
const PASS_RADIUS = 0.04; // m, as in tasks/navigation.py
const WIND_SCALE = 12.8; // m of arrow per newton: a wind equal to the weight (0.039 N) is a 50 cm arrow
const WIND_HEIGHT = 0.07; // m, the arrow is drawn above the bird, clear of its body and wings
const TARGET_COLOR = 0xe8a33d;
const OBSTACLE_COLOR = 0x8a7a68;

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
  return line;
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
  private readonly ghost = new THREE.Group();

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
    this.root.clear();
    this.root.add(this.wind, this.ghost);
    this.targets = [];
    this.pickables.length = 0;
    this.others = [];

    const start = new THREE.Mesh(
      new THREE.RingGeometry(0.012, 0.016, 32),
      new THREE.MeshBasicMaterial({ color: 0x555555, side: THREE.DoubleSide }),
    );
    start.rotation.x = -Math.PI / 2;
    start.position.set(ep.scene.start[0], 0.001, ep.scene.start[2]);
    this.root.add(start);

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
    this.root.add(this.trail);
  }

  /** The trails of the other birds of the sandbox, each in its colour. */
  setOthers(others: { ep: Episode; color: number }[]): void {
    for (const t of this.others) this.root.remove(t);
    this.others = others.map(({ ep, color }) => {
      const line = makeTrail(ep, color);
      this.root.add(line);
      return line;
    });
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
