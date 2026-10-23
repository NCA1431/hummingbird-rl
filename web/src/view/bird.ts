/**
 * The hummingbird drawn in the viewer: a simple model built from basic shapes.
 *
 * Two looks. By default, a rigged 3D model with its own wingbeat animation
 * ("Animated Hovering Flying Hummingbird Loop" by LasquetiSpice, CC BY 4.0),
 * which we attach once it is loaded. Its clip is an artist's hover animation,
 * not measured data; we replay it at the simulated frequency and widen or
 * narrow each wing's stroke to the amplitude the controller sends (so a
 * weakened wing visibly beats wider). As a fallback, and to see every wing
 * setting with measured wing kinematics, a bird drawn from basic shapes. Both live in
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
 * catch the wing at a random point of its beat). The drawn wing then fades into
 * a translucent fan covering its sweep, as a camera would blur it. The 3D
 * model has no such fan (it would not match its wings); its beat is instead
 * slowed down to at most MODEL_MAX_BEATS per frame, a visual convention only.
 * Slow the playback down (×1/16 or less) to see the true beat.
 */

import * as THREE from "three";

import type { WingCommand } from "../sim/aero";
import { type BirdParams, type Vec3, hoverAmplitude, radians } from "../sim/params";
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
const MODEL_MAX_BEATS = 0.12; // wingbeats shown per rendered frame, at most (about 7 per second)
// The clip was keyed at 30 frames per second, only 5 poses per wingbeat: in slow motion the
// wing moved in straight segments with a jolt at each pose. We resample it SMOOTHING times
// more finely with a cubic (Catmull-Rom) curve through those poses.
const SMOOTHING = 4;
const MODEL_SCALE = 1.2; // its span is 8.3 cm, the simulated bird's 10 cm
// Between its hips and its shoulders, in the model's own frame (m): where we put the centre of mass.
const MODEL_CENTRE = new THREE.Vector3(0, 0.013, 0);
// The clip also moves and sways the whole body (its root bone); the simulation does that here.
const MODEL_ROOT_BONE = "Root.2";
// The bones that sweep the wings (left, right). In the clip they swing up to about 70° either
// side of their mean orientation, close to our hover amplitude (64°), so we take the clip as
// the hover stroke and scale each shoulder's swing by (commanded amplitude / hover amplitude).
const MODEL_SHOULDERS = ["Shldr_L.9", "Shldr_R.27"];

/** The colour of each bird of the sandbox: the first one keeps its natural colours. */
export const BIRD_COLORS = [0x1d7a4c, 0x2f6fb5, 0xd9731a, 0x8a4fbf];

// The clip is prepared once (root motion removed, rotations smoothed) and shared by every bird.
const prepared = new WeakMap<THREE.AnimationClip, THREE.AnimationClip>();
function prepareClip(source: THREE.AnimationClip): THREE.AnimationClip {
  let clip = prepared.get(source);
  if (!clip) {
    const root = THREE.PropertyBinding.sanitizeNodeName(MODEL_ROOT_BONE) + ".";
    clip = source.clone();
    clip.tracks = clip.tracks
      .filter((t) => !t.name.startsWith(root)) // keep the wings, not the root motion
      .map((t) => (t instanceof THREE.QuaternionKeyframeTrack ? smoothTrack(t) : t));
    prepared.set(source, clip);
  }
  return clip;
}

interface Shoulder {
  bone: THREE.Object3D;
  mean: THREE.Quaternion; // the middle of its swing
  animated: THREE.Quaternion; // where the clip puts it now
}

/** The mean orientation of a bone over its quaternion track (keys aligned on the same hemisphere). */
function meanRotation(track: THREE.KeyframeTrack): THREE.Quaternion {
  const v = track.values;
  const sum = new THREE.Vector4();
  for (let i = 0; i < v.length; i += 4) {
    const sign = v[i] * v[0] + v[i + 1] * v[1] + v[i + 2] * v[2] + v[i + 3] * v[3] < 0 ? -1 : 1;
    sum.add(new THREE.Vector4(v[i], v[i + 1], v[i + 2], v[i + 3]).multiplyScalar(sign));
  }
  sum.normalize();
  return new THREE.Quaternion(sum.x, sum.y, sum.z, sum.w);
}

/**
 * The same rotation track, SMOOTHING times denser, along a Catmull-Rom curve through its keys
 * (component by component, then normalised: accurate for keys a few degrees apart).
 */
function smoothTrack(track: THREE.QuaternionKeyframeTrack): THREE.QuaternionKeyframeTrack {
  const t = track.times;
  const v = Array.from(track.values);
  const n = t.length;
  for (let i = 1; i < n; i++) {
    // Same hemisphere as the previous key, so that the curve does not take the long way round.
    let d = 0;
    for (let c = 0; c < 4; c++) d += v[4 * i + c] * v[4 * (i - 1) + c];
    if (d < 0) for (let c = 0; c < 4; c++) v[4 * i + c] = -v[4 * i + c];
  }
  const key = (i: number, c: number) => v[4 * Math.min(Math.max(i, 0), n - 1) + c];
  const times: number[] = [];
  const values: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < SMOOTHING; j++) {
      const u = j / SMOOTHING;
      times.push(t[i] + u * (t[i + 1] - t[i]));
      const q = [0, 1, 2, 3].map((c) => {
        const [p0, p1, p2, p3] = [key(i - 1, c), key(i, c), key(i + 1, c), key(i + 2, c)];
        return 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (3 * p1 - p0 - 3 * p2 + p3) * u * u * u);
      });
      const norm = Math.hypot(...q);
      values.push(...q.map((x) => x / norm));
    }
  }
  times.push(t[n - 1]);
  values.push(key(n - 1, 0), key(n - 1, 1), key(n - 1, 2), key(n - 1, 3));
  return new THREE.QuaternionKeyframeTrack(track.name, times, values);
}

/** For each vertex, how much it belongs to the left and to the right wing, from its skinning weights. */
function wingWeights(mesh: THREE.Mesh, shoulders: THREE.Object3D[]): THREE.BufferAttribute {
  const geo = mesh.geometry;
  const out = new Float32Array(2 * geo.attributes.position.count);
  const skinned = mesh as THREE.SkinnedMesh;
  const index = geo.attributes.skinIndex;
  const weight = geo.attributes.skinWeight;
  if (skinned.isSkinnedMesh && index && weight) {
    // Which wing each bone moves: the shoulder and everything below it.
    const side = new Int8Array(skinned.skeleton.bones.length).fill(-1);
    shoulders.forEach((shoulder, w) =>
      shoulder.traverse((o) => {
        const i = skinned.skeleton.bones.indexOf(o as THREE.Bone);
        if (i >= 0) side[i] = w;
      }),
    );
    for (let v = 0; v < index.count; v++) {
      for (let j = 0; j < 4; j++) {
        const w = side[index.getComponent(v, j)];
        if (w >= 0) out[2 * v + w] += weight.getComponent(v, j);
      }
    }
  }
  return new THREE.BufferAttribute(out, 2);
}

const delta = new THREE.Quaternion();
const axis = new THREE.Vector3();
const turn = new THREE.Quaternion();

/** mean · (the clip's rotation away from mean, with its angle multiplied by k). */
function scaleSwing(s: Shoulder, k: number): void {
  delta.copy(s.mean).invert().multiply(s.animated);
  if (delta.w < 0) delta.set(-delta.x, -delta.y, -delta.z, -delta.w);
  const sin = Math.sqrt(Math.max(1 - delta.w * delta.w, 0));
  if (sin < 1e-6) {
    s.bone.quaternion.copy(s.animated);
    return;
  }
  axis.set(delta.x / sin, delta.y / sin, delta.z / sin);
  turn.setFromAxisAngle(axis, k * 2 * Math.acos(Math.min(delta.w, 1)));
  s.bone.quaternion.copy(s.mean).multiply(turn);
}

/** An ellipsoid whose vertices are coloured by direction: green back, pale belly, optional red throat. */
function bodyPart(radii: Vec3, throat: boolean, taper = 0.0, back: THREE.Color = BACK): THREE.Mesh {
  const geo = new THREE.SphereGeometry(1, 28, 18);
  const n = geo.attributes.normal;
  const pos = geo.attributes.position;
  const colors = new Float32Array(n.count * 3);
  const c = new THREE.Color();
  for (let i = 0; i < n.count; i++) {
    const up = THREE.MathUtils.smoothstep(n.getY(i), -0.35, 0.35);
    c.copy(BELLY).lerp(back, up);
    if (throat && n.getY(i) < -0.15 && n.getX(i) > -0.2) c.copy(THROAT);
    c.toArray(colors, 3 * i);
    // Egg shape: the rear half (x < 0) narrows towards the tail.
    const k = 1 - taper * Math.max(0, -pos.getX(i));
    pos.setY(i, pos.getY(i) * k);
    pos.setZ(i, pos.getZ(i) * k);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.scale(radii[0], radii[1], radii[2]);
  geo.computeVertexNormals();
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

  /** Shows or hides the drawn wing and its fan (hidden while the 3D model is shown). */
  setSolid(visible: boolean): void {
    this.solid.visible = visible;
    this.fan.visible = visible;
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
  private shoulders: Shoulder[] = [];
  private feathers: THREE.ShaderMaterial[] = [];
  private swing: [number, number] = [1, 1]; // each wing's amplitude / hover amplitude
  private readonly hoverAmp: number;

  /** `back`: colour of the drawn bird's back and tail (each of the sandbox's birds has its own). */
  constructor(p: BirdParams, back: number = BACK.getHex()) {
    const backColor = new THREE.Color(back);
    this.hoverAmp = hoverAmplitude(p);
    const length = p.span / 2 - p.wing_root;
    this.left = new Wing(-1, p.wing_root, length, 0.011);
    this.right = new Wing(1, p.wing_root, length, 0.011);

    // Posture of a hovering hummingbird: the body about 45° nose-up below the wings, the head
    // and beak close to level, the tail pointing down and back. The stroke plane stays level
    // (perpendicular to the thrust, our y axis), as in real hovering. Sizes: body 4 cm long and
    // 1.7 cm thick, beak 2 cm, tail 2 cm, as for a ruby-throated hummingbird.
    const posture = new THREE.Group();
    posture.position.set(0.004, 0.004, 0); // pivot just behind the shoulders
    posture.rotation.z = radians(45);
    const torso = bodyPart([0.02, 0.0085, 0.009], false, 0.45, backColor);
    torso.position.set(-0.012, -0.004, 0);

    const neck = new THREE.Group();
    neck.position.set(0.008, 0.001, 0);
    neck.rotation.z = -radians(35); // head and beak 10° above the horizontal in hover
    const head = bodyPart([0.0072, 0.0066, 0.0066], true, 0.0, backColor);
    head.position.set(0.004, 0, 0);
    const beak = new THREE.Mesh(
      new THREE.ConeGeometry(0.0009, 0.02, 10),
      new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.5 }),
    );
    beak.rotation.z = -Math.PI / 2; // the cone points along +y; turn it to +x
    beak.position.set(0.0205, -0.0005, 0);
    const eyeGeo = new THREE.SphereGeometry(0.0013, 10, 8);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0x050505, roughness: 0.2 });
    const eyes = [-1, 1].map((s) => {
      const e = new THREE.Mesh(eyeGeo, eyeMat);
      e.position.set(0.0075, 0.0022, s * 0.0052);
      return e;
    });
    neck.add(head, beak, ...eyes);

    // Forked tail: two outer feathers longer than the middle ones.
    const tailShape = new THREE.Shape();
    tailShape.moveTo(-0.0025, 0);
    tailShape.lineTo(-0.0065, 0.021);
    tailShape.lineTo(0, 0.016);
    tailShape.lineTo(0.0065, 0.021);
    tailShape.lineTo(0.0025, 0);
    const tailGeo = new THREE.ShapeGeometry(tailShape);
    tailGeo.rotateX(Math.PI / 2);
    tailGeo.rotateY(-Math.PI / 2); // the fan now points backwards (-x)
    const tail = new THREE.Mesh(
      tailGeo,
      new THREE.MeshStandardMaterial({ color: backColor.clone().multiplyScalar(0.55), side: THREE.DoubleSide, roughness: 0.6 }),
    );
    tail.position.set(-0.03, -0.005, 0);
    tail.castShadow = true;
    posture.add(torso, neck, tail);

    this.drawn.add(posture);
    this.body.add(this.drawn, this.left.shoulder, this.right.shoulder);
    this.root.add(this.body);
  }

  /**
   * Adds the 3D model, once loaded, and shows it instead of the drawn bird. `model` is a copy
   * of the loaded scene (each bird needs its own skeleton), `specGloss` the model's
   * specular-glossiness map (shipped as a separate file, see feathers.ts), `keyDir` the
   * direction the sunlight comes from and `tint` the colour mixed into its feathers (null:
   * its natural colours).
   */
  attachModel(
    model: THREE.Object3D,
    animations: THREE.AnimationClip[],
    specGloss: THREE.Texture,
    keyDir: THREE.Vector3,
    tint: number | null = null,
  ): void {
    const gltf = { scene: model };
    const wrapper = new THREE.Group();
    // The model faces +z with its left wing along +x; our bird faces +x with its right wing
    // along +z. A quarter turn about the vertical maps one onto the other.
    wrapper.rotation.y = Math.PI / 2;
    wrapper.scale.setScalar(MODEL_SCALE);
    gltf.scene.position.copy(MODEL_CENTRE).negate();
    const shoulderBones = MODEL_SHOULDERS.map((n) => gltf.scene.getObjectByName(THREE.PropertyBinding.sanitizeNodeName(n)));
    gltf.scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) {
        mesh.castShadow = true;
        mesh.frustumCulled = false; // the bounds of a skinned mesh do not follow its bones
        const original = mesh.material as THREE.MeshStandardMaterial;
        mesh.customDepthMaterial = featherShadowMaterial(original);
        const material = featherMaterial(original, specGloss, keyDir);
        if (tint !== null) {
          // About a fifth of the colour, in the raw albedo's units (the shader multiplies by 4.2).
          const c = new THREE.Color(tint);
          (material.uniforms.tint.value as THREE.Vector4).set(0.22 * c.r, 0.22 * c.g, 0.22 * c.b, 0.6);
        }
        mesh.material = material;
        this.feathers.push(material);
        const bones = shoulderBones.every((b) => b) ? (shoulderBones as THREE.Object3D[]) : [];
        mesh.geometry.setAttribute("wingWeight", wingWeights(mesh, bones));
      }
    });
    wrapper.add(gltf.scene);
    this.body.add(wrapper);
    this.model = wrapper;
    this.mixer = new THREE.AnimationMixer(gltf.scene);
    const source = THREE.AnimationClip.findByName(animations, "hover") ?? animations[0];
    if (source) {
      const clip = prepareClip(source);
      this.mixer.clipAction(clip).play();
      this.mixer.update(0);
      for (const name of MODEL_SHOULDERS) {
        const id = THREE.PropertyBinding.sanitizeNodeName(name);
        const bone = gltf.scene.getObjectByName(id);
        const track = clip.tracks.find((t) => t.name === `${id}.quaternion`);
        if (bone && track) this.shoulders.push({ bone, mean: meanRotation(track), animated: bone.quaternion.clone() });
      }
      if (this.shoulders.length !== 2) this.shoulders = []; // a different model: play its clip as it is
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

  /**
   * Moves the model's wing animation on by `beats` wingbeats (at most MODEL_MAX_BEATS per
   * frame), then gives each wing the amplitude last set by setWings.
   */
  advanceModel(beats: number): void {
    if (!this.mixer) return;
    // Put back the clip's own pose first. The mixer only writes a bone when the clip's value
    // changes, so while paused it would leave last frame's widened pose in place, and we would
    // widen it again at every frame: the wing would spin out of control.
    for (const s of this.shoulders) s.bone.quaternion.copy(s.animated);
    this.mixer.update(Math.min(beats, MODEL_MAX_BEATS) * MODEL_BEAT);
    for (const s of this.shoulders) s.animated.copy(s.bone.quaternion);
    this.shoulders.forEach((s, i) => scaleSwing(s, this.swing[i]));
  }

  setPose(pos: Vec3, quat: Quat): void {
    this.root.position.set(pos[0], pos[1], pos[2]);
    this.root.quaternion.set(quat[1], quat[2], quat[3], quat[0]); // three.js stores (x, y, z, w)
  }

  /** Hides the bird (and its shadow) for the bird's-eye camera. */
  setHidden(hidden: boolean): void {
    this.body.visible = !hidden;
  }

  /** Drawing scale only (x1 = true size); the position and orientation are unaffected. */
  setScale(s: number): void {
    this.body.scale.setScalar(s);
  }

  /** The wings at this phase of the beat, for the wing settings decoded from the action. */
  setWings(phase: number, cmd: WingCommand, blur: number, damage: [number, number]): void {
    this.left.update(phase, cmd.ampLeft, cmd.tilt - cmd.yawTilt, blur, damage[0]);
    this.right.update(phase, cmd.ampRight, cmd.tilt + cmd.yawTilt, blur, damage[1]);
    this.swing = [cmd.ampLeft / this.hoverAmp, cmd.ampRight / this.hoverAmp];
    for (const m of this.feathers) (m.uniforms.damage.value as THREE.Vector2).set(damage[0], damage[1]);
  }
}
