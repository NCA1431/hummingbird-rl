/**
 * The hummingbird drawn in the viewer: a simple model built from basic shapes.
 *
 * Two looks. By default, a rigged 3D model with its own wingbeat animation
 * ("Animated Hovering Flying Hummingbird Loop" by LasquetiSpice, CC BY 4.0),
 * which we attach once it is loaded. As a fallback, and to see every wing
 * setting the controller sends, a bird drawn from basic shapes. Both live in
 * the simulator's body frame: x towards the beak, y out of the back, z along
 * the right wing; the position and orientation come straight from the
 * recorded or simulated state. The drawn bird has the simulator's sizes
 * (10 cm span, wings from 5 mm off the centre line).
 *
 * Each drawn wing hangs from a chain of pivots, one rotation each, so that every
 * setting the controller sends is visible:
 *   shoulder -> plane tilt (stroke-plane tilt, and the opposite tilt for yaw)
 *            -> stroke (sweep forward and back) -> deviation (tip up/down)
 *            -> pitch (rotation about the span) -> the wing itself.
 * A 40 Hz wingbeat cannot be shown at 60 frames per second (each frame would
 * catch the wing at a random point of its beat), so at normal speed the wing
 * fades into a translucent fan covering its sweep, as a camera would blur it;
 * slow the playback down to see the individual beats.
 */

import * as THREE from "three";
import type { GLTF } from "three/addons/loaders/GLTFLoader.js";

import type { WingCommand } from "../sim/aero";
import type { BirdParams, Vec3 } from "../sim/params";
import type { Quat } from "../sim/rotations";
import { featherMaterial, featherShadowMaterial } from "./feathers";
import { wingAngles } from "./wingKinematics";

const BACK = new THREE.Color(0x1d7a4c); // iridescent green
const BELLY = new THREE.Color(0xd9ddd2);
const THROAT = new THREE.Color(0xb3122e); // the ruby throat
const WING = new THREE.Color(0x3a3a38);
const DAMAGED = new THREE.Color(0xd9480f);
const FAN_SEGMENTS = 24;

// The 3D model's own animation beats its wings 41 times in its 6.83 s "hover" clip: one beat
// every 1/6 s of clip time. We play it at the simulated frequency instead.
const MODEL_BEAT = 1 / 6; // s of clip time per wingbeat
const MODEL_SCALE = 1.2; // its span is 8.3 cm, the simulated bird's 10 cm
// Between its hips and its shoulders, in the model's own frame (m): where we put the centre of mass.
const MODEL_CENTRE = new THREE.Vector3(0, 0.013, 0);
// The clip also moves and sways the whole body (its root bone); the simulation does that here.
const MODEL_ROOT_BONE = "Root.2";

/** An ellipsoid whose vertices are coloured by direction: green back, pale belly, optional red throat. */
function bodyPart(radii: Vec3, throat: boolean): THREE.Mesh {
  const geo = new THREE.SphereGeometry(1, 28, 18);
  const n = geo.attributes.normal;
  const colors = new Float32Array(n.count * 3);
  const c = new THREE.Color();
  for (let i = 0; i < n.count; i++) {
    const up = THREE.MathUtils.smoothstep(n.getY(i), -0.35, 0.35);
    c.copy(BELLY).lerp(BACK, up);
    if (throat && n.getY(i) < -0.15 && n.getX(i) > -0.2) c.copy(THROAT);
    c.toArray(colors, 3 * i);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.scale(radii[0], radii[1], radii[2]);
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.3, roughness: 0.45 }),
  );
  mesh.castShadow = true;
  return mesh;
}

/** A flat wing outline: span along +z, chord along x (leading edge forward), lying in the x-z plane. */
function wingGeometry(length: number, chord: number): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  const n = 12;
  // Leading edge from root to tip, then trailing edge back; the chord narrows to a rounded point.
  shape.moveTo(0.35 * chord, 0);
  for (let k = 1; k <= n; k++) {
    const u = k / n;
    shape.lineTo(0.35 * chord * Math.sqrt(1 - 0.9 * u) + 0.1 * chord * u, u * length);
  }
  for (let k = n; k >= 0; k--) {
    const u = k / n;
    shape.lineTo(-0.65 * chord * Math.pow(1 - 0.95 * u, 0.7), u * length);
  }
  const geo = new THREE.ShapeGeometry(shape);
  geo.rotateX(Math.PI / 2); // (chord, span, 0) -> (chord, 0, span)
  return geo;
}

class Wing {
  readonly shoulder = new THREE.Group();
  private readonly plane = new THREE.Group();
  private readonly stroke = new THREE.Group();
  private readonly deviation = new THREE.Group();
  private readonly pitch = new THREE.Group();
  private readonly wingMaterial: THREE.MeshStandardMaterial;
  private readonly fanMaterial: THREE.MeshBasicMaterial;
  private readonly fan: THREE.Mesh;
  private readonly fanPositions: Float32Array;
  private readonly solid: THREE.Mesh;

  /** side = +1 for the right wing (along +z), -1 for the left one. */
  constructor(
    private readonly side: number,
    root: number,
    private readonly length: number,
    chord: number,
  ) {
    this.wingMaterial = new THREE.MeshStandardMaterial({
      color: WING, side: THREE.DoubleSide, transparent: true, opacity: 0.9, roughness: 0.6,
    });
    const geo = wingGeometry(length, chord);
    if (side < 0) geo.scale(1, 1, -1);
    const wing = new THREE.Mesh(geo, this.wingMaterial);
    wing.castShadow = true;
    this.solid = wing;

    this.fanPositions = new Float32Array((FAN_SEGMENTS + 1) * 2 * 3);
    const fanGeo = new THREE.BufferGeometry();
    fanGeo.setAttribute("position", new THREE.BufferAttribute(this.fanPositions, 3));
    const index: number[] = [];
    for (let k = 0; k < FAN_SEGMENTS; k++) {
      const a = 2 * k;
      index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    fanGeo.setIndex(index);
    this.fanMaterial = new THREE.MeshBasicMaterial({
      color: WING, side: THREE.DoubleSide, transparent: true, opacity: 0, depthWrite: false,
    });
    this.fan = new THREE.Mesh(fanGeo, this.fanMaterial);

    this.shoulder.position.set(0.006, 0.006, side * root);
    this.shoulder.add(this.plane);
    this.plane.add(this.fan, this.stroke);
    this.stroke.add(this.deviation);
    this.deviation.add(this.pitch);
    this.pitch.add(wing);
  }

  /** Shows or hides the drawn wing; its fan stays, the 3D model uses it too. */
  setSolid(visible: boolean): void {
    this.solid.visible = visible;
  }

  /** Pose of the wing at this phase; `blur` in [0, 1] fades the wing into its swept fan. */
  update(phase: number, amplitude: number, planeTilt: number, blur: number, damage: number): void {
    const a = wingAngles(phase, amplitude);
    const s = this.side;
    this.plane.rotation.z = planeTilt;
    this.stroke.rotation.y = s * a.stroke; // positive stroke sweeps the wing forward
    this.deviation.rotation.x = -s * a.deviation; // positive deviation lifts the tip
    this.pitch.rotation.z = a.pitch; // positive pitch lifts the leading edge

    // The fan covers the sweep from -A to +A, in the stroke plane.
    for (let k = 0; k <= FAN_SEGMENTS; k++) {
      const th = -amplitude + (2 * amplitude * k) / FAN_SEGMENTS;
      const dx = Math.sin(th);
      const dz = s * Math.cos(th);
      this.fanPositions.set([0.15 * this.length * dx, 0, 0.15 * this.length * dz], 6 * k);
      this.fanPositions.set([this.length * dx, 0, this.length * dz], 6 * k + 3);
    }
    const attr = this.fan.geometry.attributes.position as THREE.BufferAttribute;
    attr.needsUpdate = true;
    this.fan.geometry.computeBoundingSphere();

    const color = WING.clone().lerp(DAMAGED, Math.min(1, damage / 0.4));
    this.wingMaterial.color.copy(color);
    this.fanMaterial.color.copy(color);
    this.wingMaterial.opacity = 0.9 * (1 - 0.85 * blur);
    this.fanMaterial.opacity = 0.35 * blur;
  }
}

export class Bird {
  readonly root = new THREE.Group();
  private readonly body = new THREE.Group(); // the bird in its own frame, scaled for visibility
  private readonly drawn = new THREE.Group(); // the drawn body (the wings hang from the shoulders)
  private readonly left: Wing;
  private readonly right: Wing;
  private model: THREE.Group | null = null;
  private mixer: THREE.AnimationMixer | null = null;

  constructor(p: BirdParams) {
    const length = p.span / 2 - p.wing_root;
    this.left = new Wing(-1, p.wing_root, length, 0.011);
    this.right = new Wing(1, p.wing_root, length, 0.011);

    const torso = bodyPart([0.025, 0.012, 0.0125], false);
    const head = bodyPart([0.0095, 0.009, 0.009], true);
    head.position.set(0.025, 0.005, 0);
    const beak = new THREE.Mesh(
      new THREE.ConeGeometry(0.0012, 0.022, 10),
      new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.5 }),
    );
    beak.rotation.z = -Math.PI / 2; // the cone points along +y; turn it to +x
    beak.position.set(0.044, 0.004, 0);
    const eyeGeo = new THREE.SphereGeometry(0.0016, 10, 8);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0x050505, roughness: 0.2 });
    const eyes = [-1, 1].map((s) => {
      const e = new THREE.Mesh(eyeGeo, eyeMat);
      e.position.set(0.03, 0.0075, s * 0.0068);
      return e;
    });
    const tailShape = new THREE.Shape();
    tailShape.moveTo(-0.003, 0);
    tailShape.lineTo(-0.008, 0.024);
    tailShape.lineTo(0.008, 0.024);
    tailShape.lineTo(0.003, 0);
    const tailGeo = new THREE.ShapeGeometry(tailShape);
    tailGeo.rotateX(Math.PI / 2);
    tailGeo.rotateY(-Math.PI / 2); // the fan now points backwards (-x)
    const tail = new THREE.Mesh(
      tailGeo,
      new THREE.MeshStandardMaterial({ color: 0x173d2b, side: THREE.DoubleSide, roughness: 0.6 }),
    );
    tail.position.set(-0.02, 0.0, 0);
    tail.rotation.z = 0.15; // slightly drooping
    tail.castShadow = true;

    this.drawn.add(torso, head, beak, ...eyes, tail);
    this.body.add(this.drawn, this.left.shoulder, this.right.shoulder);
    this.root.add(this.body);
  }

  /**
   * Adds the 3D model, once loaded, and shows it instead of the drawn bird. `specGloss` is the
   * model's specular-glossiness map (shipped as a separate file, see feathers.ts) and `keyDir`
   * the direction the sunlight comes from.
   */
  attachModel(gltf: GLTF, specGloss: THREE.Texture, keyDir: THREE.Vector3): void {
    const wrapper = new THREE.Group();
    // The model faces +z with its left wing along +x; our bird faces +x with its right wing
    // along +z. A quarter turn about the vertical maps one onto the other.
    wrapper.rotation.y = Math.PI / 2;
    wrapper.scale.setScalar(MODEL_SCALE);
    gltf.scene.position.copy(MODEL_CENTRE).negate();
    gltf.scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) {
        mesh.castShadow = true;
        mesh.frustumCulled = false; // the bounds of a skinned mesh do not follow its bones
        const original = mesh.material as THREE.MeshStandardMaterial;
        mesh.customDepthMaterial = featherShadowMaterial(original);
        mesh.material = featherMaterial(original, specGloss, keyDir);
      }
    });
    wrapper.add(gltf.scene);
    this.body.add(wrapper);
    this.model = wrapper;
    this.mixer = new THREE.AnimationMixer(gltf.scene);
    const clip = THREE.AnimationClip.findByName(gltf.animations, "hover") ?? gltf.animations[0];
    if (clip) {
      const root = THREE.PropertyBinding.sanitizeNodeName(MODEL_ROOT_BONE) + ".";
      clip.tracks = clip.tracks.filter((t) => !t.name.startsWith(root)); // keep the wings, not the root motion
      this.mixer.clipAction(clip).play();
    }
    this.showModel(true);
  }

  /** The 3D model (if loaded) or the drawn bird. */
  showModel(on: boolean): void {
    const model = on && this.model !== null;
    if (this.model) this.model.visible = model;
    this.drawn.visible = !model;
    this.left.setSolid(!model);
    this.right.setSolid(!model);
  }

  /** Moves the model's own wing animation on by `beats` wingbeats; frozen while the wings are a blur. */
  advanceModel(beats: number, blur: number): void {
    if (this.mixer && blur < 0.5) this.mixer.update(beats * MODEL_BEAT);
  }

  setPose(pos: Vec3, quat: Quat): void {
    this.root.position.set(pos[0], pos[1], pos[2]);
    this.root.quaternion.set(quat[1], quat[2], quat[3], quat[0]); // three.js stores (x, y, z, w)
  }

  /** Drawing scale only (x1 = true size); the position and orientation are unaffected. */
  setScale(s: number): void {
    this.body.scale.setScalar(s);
  }

  /** The wings at this phase of the beat, for the wing settings decoded from the action. */
  setWings(phase: number, cmd: WingCommand, blur: number, damage: [number, number]): void {
    this.left.update(phase, cmd.ampLeft, cmd.tilt - cmd.yawTilt, blur, damage[0]);
    this.right.update(phase, cmd.ampRight, cmd.tilt + cmd.yawTilt, blur, damage[1]);
  }
}
