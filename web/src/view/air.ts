/**
 * The air around the followed bird, drawn from the fluid simulation of sim/fluid.ts.
 *
 * The solver runs in a worker (sim/fluidWorker.ts); each frame, this page sends it the time
 * elapsed in the simulated flight and the birds, and draws the last picture it sent back. Six
 * views (the "air" menu):
 *   smoke         smoke released by the wings and around the bird, carried by the air, tinted
 *                 by direction; drawn as a volume by ray marching a 3D texture: for each pixel
 *                 we walk along the view ray through the texture and pile up the smoke, front
 *                 to back, with a little self-shadowing for depth;
 *   laser sheet   tracer particles in a thin vertical slice through the bird, facing the camera,
 *                 against a dark backdrop: what an experiment shows when it fills the air with
 *                 fog and lights one slice with a laser. Each tracer is a short streak along its
 *                 motion, brighter when faster;
 *   vortex tubes  the surfaces Q = const of the Q-criterion, which wrap the vortex cores (the
 *                 way simulations of flapping flight show their wakes), coloured by the air's speed;
 *   speed, pressure, spin   the smoke coloured by the speed of the air it is in (Viridis), its
 *                 pressure above or below the surrounding air (blue below, red above), or how
 *                 fast it spins (Inferno).
 * The air does not act back on the birds: their flight is the simulated one, with or without it.
 *
 * Occlusion: the box is a volume, not a surface, so the usual depth test (done on the box's own
 * faces) would hide all the smoke in front of the bird wherever the bird is behind the box's back,
 * and make the vortex tubes look stuck behind it. So before each frame we render the scene's depth
 * without the air (a depth pass), and each ray stops where it meets the scene: smoke and tubes in
 * front of the bird are drawn over it, those behind it are hidden.
 */

import * as THREE from "three";

import type { WingCommand } from "../sim/aero";
import { Fluid, type FluidBird, type FluidScales, type FluidView, N, SIZE } from "../sim/fluid";
import { TRACERS, windAirSpeed } from "../sim/fluidView";
import type { StepReply, StepRequest } from "../sim/fluidWorker";
import type { BirdParams, Vec3 } from "../sim/params";
import type { Quat } from "../sim/rotations";

export { paletteCss } from "../sim/fluidView";

/** One bird, as the air needs it. `flying` false: it pushes no air (it crashed, or waits). */
export interface AirBird {
  pos: Vec3;
  quat: Quat;
  thrust: number; // N
  cmd: WingCommand;
  damage: [number, number];
  phase: number; // rad
  flying: boolean;
}

export type AirMode = "off" | FluidView;

const vertexShader = /* glsl */ `
out vec3 vWorld;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

const fragmentShader = /* glsl */ `
precision highp float;
precision highp sampler3D;
uniform sampler3D volume;
uniform vec3 boxMin;
uniform vec3 boxMax;
uniform float texel;     // one cell, in texture coordinates
uniform float density;   // opacity of one cell of full smoke
uniform vec3 toLight;    // towards the sun, in texture coordinates
uniform int surfaces;    // 1: draw the surface where the opacity channel crosses "level"
uniform float level;
uniform sampler2D sceneDepth;  // depth of the scene without the air
uniform vec2 resolution;       // of the drawing buffer, pixels
uniform mat4 projInv;          // camera projection, inverted
uniform mat4 viewInv;          // camera to world
in vec3 vWorld;
out vec4 outColor;

void main() {
  vec3 ro = cameraPosition;
  vec3 rd = normalize(vWorld - ro);
  // Where the view ray enters and leaves the box (the camera may be inside it).
  vec3 inv = 1.0 / rd;
  vec3 t0 = (boxMin - ro) * inv;
  vec3 t1 = (boxMax - ro) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  float tn = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
  float tf = min(min(tmax.x, tmax.y), tmax.z);
  // The ray stops at the scene: where this pixel's depth puts the nearest opaque thing.
  vec2 suv = gl_FragCoord.xy / resolution;
  float d = texture(sceneDepth, suv).r;
  if (d < 1.0) {
    vec4 view = projInv * vec4(suv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    vec3 world = (viewInv * vec4(view.xyz / view.w, 1.0)).xyz;
    tf = min(tf, dot(world - ro, rd));
  }
  if (tf <= tn) discard;
  vec3 size = boxMax - boxMin;
  float cell = size.x * texel;
  // Half a cell per step: the texture is interpolated, so nothing is skipped.
  float stepLen = 0.5 * cell;

  if (surfaces == 1) {
    // The first crossing of the level along the ray, refined between the two samples around it;
    // the normal is the gradient of the field there, by central differences.
    float prev = 0.0;
    for (int s = 0; s < 200; s++) {
      float t = tn + (float(s) + 0.5) * stepLen;
      if (t > tf) break;
      float a = texture(volume, (ro + rd * t - boxMin) / size).a;
      if (a > level) {
        float tt = t - stepLen * (a - level) / max(a - prev, 1e-4);
        vec3 uv = (ro + rd * tt - boxMin) / size;
        vec3 g = vec3(
          texture(volume, uv + vec3(texel, 0.0, 0.0)).a - texture(volume, uv - vec3(texel, 0.0, 0.0)).a,
          texture(volume, uv + vec3(0.0, texel, 0.0)).a - texture(volume, uv - vec3(0.0, texel, 0.0)).a,
          texture(volume, uv + vec3(0.0, 0.0, texel)).a - texture(volume, uv - vec3(0.0, 0.0, texel)).a);
        vec3 nrm = -normalize(g + vec3(1e-6));
        if (dot(nrm, rd) > 0.0) nrm = -nrm;
        vec3 col = texture(volume, uv).rgb;
        vec3 sun = normalize(toLight);
        float diff = max(dot(nrm, sun), 0.0);
        float spec = pow(max(dot(nrm, normalize(sun - rd)), 0.0), 40.0);
        outColor = vec4(col * (0.4 + 0.7 * diff) + 0.25 * spec, 0.95);
        return;
      }
      prev = a;
    }
    discard;
  }

  vec4 acc = vec4(0.0);
  for (int s = 0; s < 200; s++) {
    float t = tn + (float(s) + 0.5) * stepLen;
    if (t > tf) break;
    vec3 uvw = (ro + rd * t - boxMin) / size;
    vec4 c = texture(volume, uvw);
    if (c.a < 0.002) continue;
    // Opacity of this step: density per cell, scaled to the step's length.
    float a = 1.0 - pow(1.0 - clamp(c.a * density, 0.0, 0.99), 0.5);
    // A cheap self-shadow: smoke between this point and the sun darkens it (two samples
    // towards the light), which gives the volume its depth.
    float shadow = texture(volume, uvw + toLight).a + texture(volume, uvw + 2.5 * toLight).a;
    vec3 lit = c.rgb * (1.0 - 0.35 * clamp(shadow * density, 0.0, 1.0));
    acc.rgb += (1.0 - acc.a) * a * lit;
    acc.a += (1.0 - acc.a) * a;
    if (acc.a > 0.97) break;
  }
  if (acc.a < 0.004) discard;
  outColor = vec4(acc.rgb / acc.a, acc.a);
}
`;

const TUBE_LEVEL = 0.06; // the vortex surfaces: Q at 6% of its current maximum

export class Air {
  readonly root = new THREE.Group();
  private readonly worker: Worker;
  private readonly texels = new Uint8Array(N * N * N * 4);
  private readonly texture: THREE.Data3DTexture;
  private readonly box: THREE.Mesh;
  private readonly material: THREE.ShaderMaterial;
  private readonly lines: THREE.LineSegments;
  private readonly backdrop: THREE.Mesh;
  private readonly outline: THREE.LineSegments;
  private mode: AirMode = "off";
  private buffers: { texture: ArrayBuffer; lines: ArrayBuffer; colors: ArrayBuffer } | null = {
    texture: new ArrayBuffer(N * N * N * 4),
    lines: new ArrayBuffer(TRACERS * 6 * 4),
    colors: new ArrayBuffer(TRACERS * 8 * 4),
  };
  private pending = 0; // simulated time not yet given to the worker
  private repaint = false; // the view changed: a new picture of the same air is due
  private resetNext = true;
  private obstaclesNext: { center: Vec3; radius: number }[] | null = [];
  private lastScales: FluidScales = { speed: 0.5, pressure: 0.05, vortices: 20, q: 100 };
  private sheetNormal = new THREE.Vector3(0, 0, 1);
  private depth: THREE.WebGLRenderTarget | null = null;

  constructor(private readonly p: BirdParams) {
    this.texture = new THREE.Data3DTexture(this.texels, N, N, N);
    this.texture.format = THREE.RGBAFormat;
    this.texture.type = THREE.UnsignedByteType;
    this.texture.minFilter = this.texture.magFilter = THREE.LinearFilter;
    this.texture.unpackAlignment = 1;
    this.texture.needsUpdate = true;
    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader,
      fragmentShader,
      uniforms: {
        volume: { value: this.texture },
        boxMin: { value: new THREE.Vector3() },
        boxMax: { value: new THREE.Vector3() },
        texel: { value: 1 / N },
        density: { value: 0.9 },
        toLight: { value: new THREE.Vector3(0.6, 1.5, 0.4).normalize().multiplyScalar(1.5 / N) },
        surfaces: { value: 0 },
        level: { value: TUBE_LEVEL },
        sceneDepth: { value: null },
        resolution: { value: new THREE.Vector2(1, 1) },
        projInv: { value: new THREE.Matrix4() },
        viewInv: { value: new THREE.Matrix4() },
      },
      transparent: true,
      depthWrite: false,
      depthTest: false, // the depth pass does it, per ray (see above)
      side: THREE.BackSide, // the back faces: the box is still drawn when the camera is inside it
    });
    this.box = new THREE.Mesh(new THREE.BoxGeometry(SIZE, SIZE, SIZE), this.material);
    this.box.frustumCulled = false;
    this.box.renderOrder = 10; // after the other transparent things (targets, trails)

    // The laser sheet: a dark backdrop, as in a dark lab, and the tracers' streaks in front of it.
    this.backdrop = new THREE.Mesh(
      new THREE.PlaneGeometry(SIZE, SIZE),
      new THREE.MeshBasicMaterial({ color: 0x050a08, transparent: true, opacity: 0.6, depthWrite: false, side: THREE.DoubleSide }),
    );
    this.backdrop.renderOrder = 9;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRACERS * 6), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(TRACERS * 8), 4));
    this.lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
    this.lines.frustumCulled = false;
    this.lines.renderOrder = 11;
    // The edges of the box of simulated air, on request.
    this.outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(SIZE, SIZE, SIZE)),
      new THREE.LineBasicMaterial({ color: 0x2d3a46, transparent: true, opacity: 0.55 }),
    );
    this.outline.visible = false;
    this.root.add(this.box, this.backdrop, this.lines, this.outline);
    this.root.visible = false;

    this.worker = new Worker(new URL("../sim/fluidWorker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<StepReply>) => this.receive(e.data);
  }

  get current(): AirMode {
    return this.mode;
  }

  /** The colour scales of the last picture, for the key. */
  get scales(): FluidScales {
    return this.lastScales;
  }

  setMode(mode: AirMode): void {
    const was = this.mode;
    this.mode = mode;
    this.root.visible = mode !== "off";
    this.material.uniforms.surfaces.value = mode === "tubes" ? 1 : 0;
    this.box.visible = mode !== "sheet";
    this.backdrop.visible = this.lines.visible = mode === "sheet";
    if (was === "off" && mode !== "off") this.reset();
    this.repaint = true;
  }

  /** Shows the edges of the box of air around the followed bird (the region simulated). */
  showBox(on: boolean): void {
    this.outline.visible = on;
  }

  /** The obstacles of the scene on screen: the air flows around them. */
  setObstacles(obstacles: { center: Vec3; radius: number }[]): void {
    this.obstaclesNext = obstacles.map((o) => ({ center: [...o.center] as Vec3, radius: o.radius }));
  }

  /** Still air and no smoke: a new flight, a seek, a restart (the old smoke belongs to another moment). */
  reset(): void {
    this.resetNext = true;
    this.pending = 0;
    this.texels.fill(0);
    this.texture.needsUpdate = true;
  }

  /**
   * Moves the air on by `dt` seconds of simulated time. `center`: the followed bird, which the
   * box follows; `windForce`: the wind's push (N), the same everywhere; `birds`: every bird on
   * screen (those inside the box push its air); `eye`: the camera, which the laser sheet faces.
   * The worker gets the time as soon as it is free; while it works, the time adds up.
   */
  update(dt: number, center: Vec3, windForce: Vec3, birds: AirBird[], eye: Vec3): void {
    if (this.mode === "off") return;
    if (this.mode === "sheet") {
      // The backdrop follows the bird and the camera at every frame (not at the solver's pace,
      // nor by whole cells as the box does), facing the camera, just behind the sheet.
      const n = this.sheetNormal.set(center[0] - eye[0], 0, center[2] - eye[2]);
      if (n.lengthSq() < 1e-12) n.set(0, 0, 1);
      n.normalize();
      const middle = Fluid.originFor(center)[1] + SIZE / 2;
      this.backdrop.position.set(center[0], middle, center[2]).addScaledVector(n, 0.012);
      this.backdrop.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
    }
    this.pending += dt;
    if (!this.buffers) return; // the worker is still busy with the last step
    if (this.pending <= 0 && !this.repaint && !this.resetNext) return;
    const step = Math.min(this.pending, 0.1); // a long hiccup is cut, not jumped
    this.pending = 0;
    this.repaint = false;
    const request: StepRequest = {
      dt: step,
      center: [...center] as Vec3,
      eye,
      wind: windAirSpeed(windForce),
      birds: birds.filter((b) => b.flying).map((b) => this.fluidBird(b)),
      obstacles: this.obstaclesNext,
      view: this.mode,
      reset: this.resetNext,
      ...this.buffers,
    };
    this.obstaclesNext = null;
    this.resetNext = false;
    this.worker.postMessage(request, [this.buffers.texture, this.buffers.lines, this.buffers.colors]);
    this.buffers = null;
  }

  /**
   * The depth pass: the scene without the air, into a depth texture the size of the screen (its
   * shadow maps are not redrawn: they are those of the frame). Called just before each frame.
   */
  depthPass(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera): void {
    if (this.mode === "off" || this.mode === "sheet") return;
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    if (!this.depth || this.depth.width !== size.x || this.depth.height !== size.y) {
      this.depth?.dispose();
      this.depth = new THREE.WebGLRenderTarget(size.x, size.y, { depthTexture: new THREE.DepthTexture(size.x, size.y) });
    }
    this.root.visible = false;
    const shadows = renderer.shadowMap.autoUpdate;
    renderer.shadowMap.autoUpdate = false;
    renderer.setRenderTarget(this.depth);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.shadowMap.autoUpdate = shadows;
    this.root.visible = true;
    const u = this.material.uniforms;
    u.sceneDepth.value = this.depth.depthTexture;
    (u.resolution.value as THREE.Vector2).copy(size);
    (u.projInv.value as THREE.Matrix4).copy(camera.projectionMatrixInverse);
    (u.viewInv.value as THREE.Matrix4).copy(camera.matrixWorld);
  }

  private receive(r: StepReply): void {
    this.buffers = { texture: r.texture, lines: r.lines, colors: r.colors };
    this.lastScales = r.scales;
    if (this.mode === "off") return;
    this.texels.set(new Uint8Array(r.texture));
    this.texture.needsUpdate = true;
    const [x, y, z] = r.origin;
    this.box.position.set(x + SIZE / 2, y + SIZE / 2, z + SIZE / 2);
    this.outline.position.copy(this.box.position);
    (this.material.uniforms.boxMin.value as THREE.Vector3).set(x, y, z);
    (this.material.uniforms.boxMax.value as THREE.Vector3).set(x + SIZE, y + SIZE, z + SIZE);
    if (r.sheet) {
      const pos = this.lines.geometry.attributes.position as THREE.BufferAttribute;
      const col = this.lines.geometry.attributes.color as THREE.BufferAttribute;
      (pos.array as Float32Array).set(new Float32Array(r.lines));
      (col.array as Float32Array).set(new Float32Array(r.colors));
      pos.needsUpdate = col.needsUpdate = true;
    }
  }

  private fluidBird(b: AirBird): FluidBird {
    const q = new THREE.Quaternion(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
    const axis = (x: number, y: number, z: number): Vec3 => new THREE.Vector3(x, y, z).applyQuaternion(q).toArray() as Vec3;
    return {
      pos: [...b.pos] as Vec3,
      up: axis(0, 1, 0),
      fwd: axis(1, 0, 0),
      right: axis(0, 0, 1),
      thrust: b.thrust,
      ampLeft: b.cmd.ampLeft,
      ampRight: b.cmd.ampRight,
      damage: [...b.damage] as [number, number],
      phase: b.phase,
      freq: b.cmd.freq,
      span: this.p.span,
      root: this.p.wing_root,
    };
  }
}
