/**
 * From a desired thrust and torques to the 5 wing actions (the "mixer").
 *
 * Port of hummingbird_rl/controllers/allocation.py, where the inversion of the
 * wing model is explained: the frequency stays at f0 unless a wing would need
 * more than its maximum amplitude, and an impossible request is clipped
 * quantity by quantity.
 */

import { type BirdParams, type Vec3, rWing, thrustGain } from "../sim/params";
import { clip } from "./controller";

/** Actions in [-1, 1]^5 for a total thrust (N) and body torques [roll, yaw, pitch] (N·m). */
export function allocate(thrust: number, torque: Vec3, p: BirdParams): number[] {
  const k = thrustGain(p);
  const r = rWing(p);
  const diff = torque[0] / r;
  const fLeft = Math.max(0.5 * (thrust + diff), 0.0);
  const fRight = Math.max(0.5 * (thrust - diff), 0.0);

  const fNeeded = Math.sqrt(Math.max(fLeft, fRight) / k) / p.amp_max;
  const freq = clip(fNeeded, p.f0, p.f0 * (1.0 + p.freq_span));
  const ampLeft = Math.min(Math.sqrt(fLeft / k) / freq, p.amp_max);
  const ampRight = Math.min(Math.sqrt(fRight / k) / freq, p.amp_max);

  const tReal = k * freq ** 2 * (ampLeft ** 2 + ampRight ** 2); // thrust actually produced
  const tSafe = Math.max(tReal, 1e-6);
  const sinTilt = Math.sin(p.tilt_max);
  const sinYaw = Math.sin(p.yaw_tilt_max);
  const tilt = Math.asin(clip(torque[2] / (tSafe * p.pitch_lever), -sinTilt, sinTilt));
  const yawTilt = Math.asin(clip(torque[1] / (p.k_yaw * tSafe * r), -sinYaw, sinYaw));

  return [
    (2.0 * ampLeft) / p.amp_max - 1.0,
    (2.0 * ampRight) / p.amp_max - 1.0,
    (freq / p.f0 - 1.0) / p.freq_span,
    tilt / p.tilt_max,
    yawTilt / p.yaw_tilt_max,
  ].map((a) => clip(a, -1.0, 1.0));
}
