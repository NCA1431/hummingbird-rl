/**
 * A trained agent in the browser: the forward pass of its actor network.
 *
 * The policy files are written by `scripts/export_web.py policy` (see
 * hummingbird_rl/export/web_export.py): the layers in order, each with its
 * weights (row by row, y = W x + b), its bias and its activation, all stored
 * as base64-encoded float32. Only the deterministic actor is kept, the action
 * the evaluation used: PPO's mean, clipped to [-1, 1]; tanh of the last layer
 * for SAC and TD3.
 *
 * We compute in float32, like PyTorch, rounding after every layer
 * (Math.fround); the sums themselves are done in float64, so the result can
 * differ from PyTorch's in the last bits, far below anything that matters.
 */

import { type Controller } from "./controller";

export interface PolicyLayer {
  in: number;
  out: number;
  activation: "tanh" | "relu" | "none";
  weight: string; // base64, float32 little-endian, shape (out, in)
  bias: string;
}

export interface PolicyFile {
  format: string;
  name: string;
  algo: string; // ppo, sac, td3
  task?: string | null; // the task it was trained on
  which?: string; // final or best
  obs_size: number;
  action_size: number;
  output: "clip" | "tanh";
  layers: PolicyLayer[];
}

interface Layer {
  nIn: number;
  nOut: number;
  activation: PolicyLayer["activation"];
  weight: Float32Array;
  bias: Float32Array;
}

function decode(b64: string): Float32Array {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const out = new Float32Array(bytes.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(4 * i, true); // little-endian whatever the machine
  return out;
}

export class MlpPolicy implements Controller {
  readonly name: string;
  readonly algo: string;
  private readonly layers: Layer[];
  private readonly output: PolicyFile["output"];

  constructor(file: PolicyFile) {
    if (!file.format.startsWith("hummingbird-policy/")) throw new Error(`not a policy file: ${file.format}`);
    this.name = file.name;
    this.algo = file.algo;
    this.output = file.output;
    this.layers = file.layers.map((l) => {
      const layer = { nIn: l.in, nOut: l.out, activation: l.activation, weight: decode(l.weight), bias: decode(l.bias) };
      if (layer.weight.length !== l.in * l.out || layer.bias.length !== l.out) throw new Error("bad layer size");
      return layer;
    });
  }

  reset(): void {} // a feed-forward network has no memory

  act(obs: ArrayLike<number>): number[] {
    let x = Float32Array.from(obs);
    for (const l of this.layers) {
      const y = new Float32Array(l.nOut);
      for (let r = 0; r < l.nOut; r++) {
        let s = l.bias[r];
        const row = r * l.nIn;
        for (let c = 0; c < l.nIn; c++) s += l.weight[row + c] * x[c];
        y[r] = l.activation === "tanh" ? Math.tanh(s) : l.activation === "relu" ? Math.max(s, 0.0) : s;
      }
      x = y; // storing in a Float32Array rounds to float32
    }
    return Array.from(x, (v) => (this.output === "tanh" ? Math.fround(Math.tanh(v)) : Math.min(Math.max(v, -1.0), 1.0)));
  }
}

/** Downloads and builds a policy from web/public/policies/. */
export async function loadPolicy(file: string): Promise<MlpPolicy> {
  const r = await fetch(`policies/${file}`);
  if (!r.ok) throw new Error(`policies/${file}: ${r.status}`);
  return new MlpPolicy((await r.json()) as PolicyFile);
}

export interface PolicyEntry {
  file: string;
  name: string;
  algo: string;
  task: string | null;
  which: string;
}
