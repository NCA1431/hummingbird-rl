/**
 * How a wing moves within one wingbeat, from measurements on a real hummingbird.
 *
 * The physics only knows wingbeat-averaged forces, so the motion of the wing
 * within a beat is for the drawing only. We take it from Song et al. (2016),
 * who filmed a calliope hummingbird in hover (Dryad dataset
 * doi:10.5061/dryad.8ch1b); the curves were resampled to 64 points per beat
 * in the original project and are copied in wingData.json.
 *
 * Three angles per wing, as functions of the phase φ in the beat:
 *   stroke     the sweep forward and back, -A cos φ: φ = 0 is the start of the
 *              downstroke, wing at the back (in hover the "downstroke" sweeps forward)
 *   deviation  a small up-and-down motion (6° sin 2φ) that bends the tip path into an oval
 *   pitch      the rotation of the wing about its own span: the measured chord
 *              angle plus part of the measured twist. It flips sign between the
 *              two half-strokes, which is how hummingbirds produce lift both ways.
 * The amplitude A and the frequency come from the simulation (the decoded
 * actions), not from the data, so the drawing follows what the controller asks.
 */

import data from "./wingData.json";

const DEG = Math.PI / 180;
const CHORD: number[] = data.chord_tip;
const TWIST: number[] = data.twist_tip;
const DEVIATION = 6 * DEG;
const TWIST_SHARE = 0.4; // the tip twist is partly included in the chord angle already

export interface WingAngles {
  stroke: number; // rad
  deviation: number; // rad
  pitch: number; // rad
}

/** Linear interpolation in a curve sampled at 64 evenly spaced phases (cyclic). */
export function sampleCurve(curve: number[], phase: number): number {
  const u = ((phase / (2 * Math.PI)) % 1 + 1) % 1;
  const f = u * curve.length;
  const i = Math.floor(f) % curve.length;
  const j = (i + 1) % curve.length;
  const frac = f - Math.floor(f);
  return curve[i] * (1 - frac) + curve[j] * frac;
}

export function wingAngles(phase: number, amplitude: number): WingAngles {
  return {
    stroke: -amplitude * Math.cos(phase),
    deviation: DEVIATION * Math.sin(2 * phase),
    pitch: (sampleCurve(CHORD, phase) + TWIST_SHARE * sampleCurve(TWIST, phase)) * DEG,
  };
}
