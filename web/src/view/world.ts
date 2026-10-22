/**
 * What surrounds the bird in an episode: its targets, the obstacles, the start
 * point, the wind, and the trail it leaves.
 *
 * Targets are drawn as translucent balls of the radius the task uses: 3 cm for
 * a target to hold (the bird must stay inside for a quarter of a second) and
 * 4 cm for a navigation waypoint that only has to be passed. The current
 * target is bright, the ones already reached are faded.
 */

import * as THREE from "three";

import type { Vec3 } from "../sim/params";
import type { Episode } from "../modes/watch";

const HOLD_RADIUS = 0.03; // m, as in tasks/base.py
const PASS_RADIUS = 0.04; // m, as in tasks/navigation.py
const WIND_SCALE = 12.8; // m of arrow per newton: a wind equal to the weight (0.039 N) is a 50 cm arrow
const WIND_HEIGHT = 0.07; // m, the arrow is drawn above the bird, clear of its body and wings

export class World {
  readonly root = new THREE.Group();
  private targets: THREE.Mesh[] = [];
  private readonly wind: THREE.ArrowHelper;
  private trail: THREE.Line | null = null;

  constructor() {
    this.wind = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.1, 0x2b6cb0, 0.02, 0.012);
    this.root.add(this.wind);
  }

  /** Rebuilds everything for a new episode. */
  load(ep: Episode): void {
    this.root.clear();
    this.root.add(this.wind);
    this.targets = [];

    const start = new THREE.Mesh(
      new THREE.RingGeometry(0.012, 0.016, 32),
      new THREE.MeshBasicMaterial({ color: 0x555555, side: THREE.DoubleSide }),
    );
    start.rotation.x = -Math.PI / 2;
    start.position.set(ep.scene.start[0], 0.001, ep.scene.start[2]);
    this.root.add(start);

    const n = ep.scene.targets.length;
    ep.scene.targets.forEach((p, i) => {
      const last = i === n - 1;
      const radius = ep.task === "navigation" && !last ? PASS_RADIUS : HOLD_RADIUS;
      const ball = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 24, 16),
        new THREE.MeshStandardMaterial({ color: 0xe8a33d, transparent: true, opacity: 0.3, depthWrite: false }),
      );
      ball.position.set(p[0], p[1], p[2]);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.003, 10, 8), new THREE.MeshBasicMaterial({ color: 0xc26a00 }));
      ball.add(dot);
      // A thin stem down to the ground shows the target's height.
      const stem = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -p[1], 0)]),
        new THREE.LineDashedMaterial({ color: 0x9a7a40, dashSize: 0.01, gapSize: 0.01 }),
      );
      stem.computeLineDistances();
      ball.add(stem);
      this.targets.push(ball);
      this.root.add(ball);
    });

    for (const o of ep.scene.obstacles) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(o.radius, 32, 20),
        new THREE.MeshStandardMaterial({ color: 0x8a7a68, roughness: 0.85 }),
      );
      sphere.position.set(o.center[0], o.center[1], o.center[2]);
      sphere.castShadow = sphere.receiveShadow = true;
      this.root.add(sphere);
    }

    this.trail = makeTrail(ep);
    this.root.add(this.trail);
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
 * The path of the whole episode, coloured by speed (blue slow, red at 1 m/s and above),
 * and grey for what is shown after the episode's end (live mode, after a loss of control).
 */
function makeTrail(ep: Episode): THREE.Line {
  const f = ep.frames;
  const positions = new Float32Array(f.pos.length * 3);
  const colors = new Float32Array(f.pos.length * 3);
  const slow = new THREE.Color(0x2563eb);
  const fast = new THREE.Color(0xdc2626);
  const after = new THREE.Color(0x8a8f96);
  const end = ep.endFrame ?? f.pos.length;
  const c = new THREE.Color();
  f.pos.forEach((p, i) => {
    positions.set(p, 3 * i);
    const v = f.vel[i];
    if (i > end) c.copy(after);
    else c.copy(slow).lerp(fast, Math.min(1, Math.hypot(v[0], v[1], v[2])));
    c.toArray(colors, 3 * i);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true }));
}
