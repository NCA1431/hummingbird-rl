/**
 * Physical parameters of the simulated hummingbird.
 *
 * Line-by-line port of hummingbird_rl/sim/params.py, where every value is
 * explained; only what the browser needs is commented here. The field names
 * are the Python ones (snake_case) on purpose: the golden test files written
 * by Python are read directly, and the two files can be compared side by side.
 * The parity tests check that the defaults below are exactly Python's.
 */

export type Vec3 = [number, number, number];

export interface BirdParams {
  // body
  mass: number;
  span: number;
  wing_area: number;
  body_semi_axes: Vec3;
  wing_mass: number;
  wing_root: number;
  // air
  rho: number;
  g: number;
  // wings and steering
  f0: number;
  freq_span: number;
  amp_max: number;
  c_lift: number;
  tilt_max: number;
  pitch_lever: number;
  yaw_tilt_max: number;
  k_yaw: number;
  // rotation
  tau_omega: number;
  k_u_pitch: number;
  k_u_roll: number;
  // drag
  c_fwd: number;
  c_back: number;
  c_side: number;
  c_vert: number;
  // simulation
  r_bird: number;
  dt: number;
  action_repeat: number;
}

/** Same formula as Python's math.radians, x * (pi / 180), so the values agree to the last bit. */
export function radians(deg: number): number {
  return deg * (Math.PI / 180.0);
}

export const DEFAULT_PARAMS: Readonly<BirdParams> = Object.freeze({
  mass: 0.004,
  span: 0.1,
  wing_area: 5.0e-4,
  body_semi_axes: [0.04, 0.01, 0.01] as Vec3,
  wing_mass: 1.0e-4,
  wing_root: 0.005,
  rho: 1.225,
  g: 9.81,
  f0: 40.0,
  freq_span: 0.2,
  amp_max: radians(80.0),
  c_lift: 2.6,
  tilt_max: radians(20.0),
  pitch_lever: 0.005,
  yaw_tilt_max: radians(20.0),
  k_yaw: 0.5,
  tau_omega: 0.4,
  k_u_pitch: 1.91e-5,
  k_u_roll: 1.15e-5,
  c_fwd: 0.0075,
  c_back: 0.015,
  c_side: 0.023,
  c_vert: 0.02,
  r_bird: 0.02,
  dt: 1.0 / 240.0,
  action_repeat: 4,
});

/** Parameters from a Python dict (golden files, exported runs); every field must be present. */
export function paramsFromPython(d: Record<string, unknown>): BirdParams {
  for (const key of Object.keys(DEFAULT_PARAMS)) {
    if (!(key in d)) throw new Error(`parameter ${key} is missing`);
  }
  const axes = d.body_semi_axes as number[];
  return { ...(d as unknown as BirdParams), body_semi_axes: [axes[0], axes[1], axes[2]] };
}

// --- derived quantities (the @property methods of BirdParams) -------------------------

/** Mid-wing radius used for the wing speed: a quarter of the span. */
export function rWing(p: BirdParams): number {
  return p.span / 4.0;
}

export function weight(p: BirdParams): number {
  return p.mass * p.g;
}

/** Moments of inertia [roll, yaw, pitch]: ellipsoid body plus two thin rods for the wings. */
export function inertia(p: BirdParams): Vec3 {
  const [a, b, c] = p.body_semi_axes;
  const mBody = p.mass - 2.0 * p.wing_mass;
  const length = p.span / 2.0 - p.wing_root;
  const z0 = p.wing_root;
  const iWing = p.wing_mass * (z0 * z0 + z0 * length + (length * length) / 3.0);
  const ixx = (mBody * (b * b + c * c)) / 5.0 + 2.0 * iWing; // roll
  const iyy = (mBody * (a * a + c * c)) / 5.0 + 2.0 * iWing; // yaw
  const izz = (mBody * (a * a + b * b)) / 5.0; // pitch
  return [ixx, iyy, izz];
}

export function damping(p: BirdParams): Vec3 {
  const i = inertia(p);
  return [i[0] / p.tau_omega, i[1] / p.tau_omega, i[2] / p.tau_omega];
}

/** k such that one undamaged wing pushes with k (f A)². */
export function thrustGain(p: BirdParams): number {
  const k = 2.0 * Math.PI * rWing(p);
  return 0.5 * p.rho * p.c_lift * (p.wing_area / 2.0) * (k * k);
}

/** Amplitude at which both wings at f0 carry the weight (about 64°). */
export function hoverAmplitude(p: BirdParams, massFactor = 1.0): number {
  const w = massFactor * weight(p);
  return Math.sqrt(w / (2.0 * thrustGain(p))) / p.f0;
}

export function maxThrust(p: BirdParams): number {
  const f = p.f0 * (1.0 + p.freq_span);
  const fa = f * p.amp_max;
  return 2.0 * thrustGain(p) * (fa * fa);
}

export function hoverPower(p: BirdParams): number {
  const aH = hoverAmplitude(p);
  const u = 2.0 * Math.PI * p.f0 * aH * rWing(p);
  return 2.0 * (weight(p) / 2.0) * u;
}
