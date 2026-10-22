"""What the browser needs to run our policies and to check its own copy of the simulator.

Two things are exported here.

Policies. The browser cannot run PyTorch, so a trained agent is exported as
the list of its layers (weights, biases, activation) plus what is done to the
output. Only the deterministic actor is kept: the part that maps an
observation to the action used in evaluation (PPO: the mean of its Gaussian,
clipped to [-1, 1]; SAC and TD3: tanh of the last layer). Weights are stored
as base64-encoded float32, the format PyTorch computes in: about 3 times
smaller than writing the numbers as text, and exact.

Golden files. The web viewer has its own TypeScript copy of the simulator and
of the observation builder. To be sure it is the same physics the agents were
trained on, we record here, with the Python code, a set of inputs (start
state, conditions, actions, wind) and the resulting outputs; the TypeScript
tests must reproduce them to 1e-9. The inputs are rounded before they are
simulated, so the stored outputs are exactly those of the stored inputs.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict
from math import pi, radians

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.torch_layers import FlattenExtractor

from hummingbird_rl.controllers import PIDController
from hummingbird_rl.envs.observation import OBS_SIZE, build_observation
from hummingbird_rl.sim import DEFAULT_PARAMS, BirdDynamics, BirdParams, BirdState, check_limits, hover_action
from hummingbird_rl.world.scene import START, Obstacle

POLICY_FORMAT = "hummingbird-policy/1"
ACTIVATIONS = {torch.nn.Tanh: "tanh", torch.nn.ReLU: "relu"}


# --- policies -------------------------------------------------------------------------

def _encode(t: torch.Tensor) -> str:
    return base64.b64encode(t.detach().cpu().numpy().astype("<f4").tobytes()).decode("ascii")


def _decode(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def _layers(modules) -> list[dict]:
    """Linear layers in order, each followed by its activation (or "none")."""
    layers = []
    for m in modules:
        if isinstance(m, torch.nn.Linear):
            layers.append({"in": m.in_features, "out": m.out_features, "activation": "none",
                           "weight": _encode(m.weight), "bias": _encode(m.bias)})
        elif type(m) in ACTIVATIONS:
            layers[-1]["activation"] = ACTIVATIONS[type(m)]
        else:
            raise ValueError(f"cannot export layer {m!r}")
    return layers


def policy_to_dict(model, name: str = "") -> dict:
    """The deterministic actor of a PPO, SAC or TD3 model, as plain data.

    weight is stored row by row, shape (out, in), so that y = W x + b.
    """
    if isinstance(model, PPO):
        extractor = model.policy.pi_features_extractor
        modules, output = [*model.policy.mlp_extractor.policy_net, model.policy.action_net], "clip"
    elif isinstance(model, SAC):
        extractor = model.actor.features_extractor
        modules, output = [*model.actor.latent_pi, model.actor.mu], "tanh"
    elif isinstance(model, TD3):
        extractor = model.actor.features_extractor
        modules = list(model.actor.mu)
        if not isinstance(modules[-1], torch.nn.Tanh):
            raise ValueError("expected TD3's actor to end with a Tanh")
        modules, output = modules[:-1], "tanh"          # the final Tanh becomes the output rule
    else:
        raise TypeError(f"cannot export a {type(model).__name__}")
    # Our observations are already scaled by hand, so the feature extractor must
    # be a plain identity; anything else would have to be exported too.
    if not isinstance(extractor, FlattenExtractor):
        raise ValueError("expected the default FlattenExtractor")
    return {"format": POLICY_FORMAT, "name": name, "algo": type(model).__name__.lower(),
            "obs_size": OBS_SIZE, "action_size": 5, "output": output, "layers": _layers(modules)}


def mlp_forward(policy: dict, obs: np.ndarray) -> np.ndarray:
    """Reference implementation of what the browser computes, used to test the export."""
    x = np.asarray(obs, dtype=np.float32)
    for layer in policy["layers"]:
        w = _decode(layer["weight"]).reshape(layer["out"], layer["in"])
        x = w @ x + _decode(layer["bias"])
        if layer["activation"] == "tanh":
            x = np.tanh(x)
        elif layer["activation"] == "relu":
            x = np.maximum(x, 0.0)
    return np.tanh(x) if policy["output"] == "tanh" else np.clip(x, -1.0, 1.0)


# --- golden files ---------------------------------------------------------------------

def _r(x, digits: int = 12):
    """Round to `digits` significant digits, keeping the structure (numbers, arrays, lists)."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, digits) for v in x]
    return float(f"{float(x):.{digits}g}")


def params_dict(p: BirdParams) -> dict:
    return json.loads(json.dumps(asdict(p)))         # tuples -> lists


def derived_dict(p: BirdParams) -> dict:
    """The derived quantities, checked separately so that a mismatch is easy to locate."""
    return {"r_wing": p.r_wing, "weight": p.weight, "inertia": p.inertia.tolist(), "damping": p.damping.tolist(),
            "thrust_gain": p.thrust_gain, "hover_amplitude": p.hover_amplitude(),
            "hover_amplitude_heavy": p.hover_amplitude(1.2), "max_thrust": p.max_thrust(),
            "hover_power": p.hover_power(), "hover_action": hover_action(p).tolist()}


def _random_state(rng: np.random.Generator, big: bool) -> BirdState:
    """A start state; with big=True far from hover (fast, tilted, spinning)."""
    k = 1.0 if big else 0.1
    return BirdState.from_euler(
        pos=_r(START + rng.uniform(-0.1, 0.1, 3)),
        vel=_r(k * rng.uniform(-1.0, 1.0, 3)),
        yaw=_r(rng.uniform(-pi, pi)),
        pitch=_r(radians(k * rng.uniform(-30.0, 30.0))),
        roll=_r(radians(k * rng.uniform(-30.0, 30.0))),
        omega=_r(k * rng.uniform(-5.0, 5.0, 3)),
    )


def golden_dynamics(n_cases: int = 20, n_decisions: int = 75, seed: int = 0) -> dict:
    """Start state, conditions, actions and wind in; the state after every decision out.

    Case 0 is a calm hover with the hover action, the easiest one to debug by
    hand. Case 1 uses changed parameters (a stable bird with stronger wings)
    to check that the TypeScript side really reads them. The other cases mix
    every effect: random starts, heavier or lighter birds, a damaged wing,
    wind, and actions that sometimes go beyond [-1, 1], which the decoding
    must clip.

    The actions come from the PID plus noise. Purely random actions are no
    good here: with a damaged wing and nobody to correct it, the bird spins up
    to more than 100 rad/s, where a 1/240 s step turns it by half a radian and
    the integration diverges. That never happens in an episode, which ends at
    80 degrees of tilt, but it would make the comparison meaningless. The
    wind is held constant during each decision to keep the file small; the
    gust process itself is checked statistically on the TypeScript side.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n_cases):
        p = DEFAULT_PARAMS.with_changes(k_u_pitch=0.0, k_u_roll=0.0, c_lift=3.0) if i == 1 else DEFAULT_PARAMS
        if i == 0:
            state, mass_factor, damage = BirdState.from_euler(pos=START), 1.0, (0.0, 0.0)
            mean_wind, gusts, noise = np.zeros(3), 0.0, 0.0
        else:
            state = _random_state(rng, big=i % 2 == 0)
            mass_factor = _r(rng.uniform(0.75, 1.25))
            damage = (_r(rng.uniform(0.0, 0.4)), 0.0) if i % 3 == 0 else (0.0, 0.0)
            mean_wind = rng.uniform(0.0, 0.3) * p.weight * np.array([1.0, 0.0, 1.0]) * rng.standard_normal(3)
            gusts, noise = 0.05 * p.weight, 0.1

        start = state.copy()
        dyn = BirdDynamics(p, mass_factor, damage)
        pid = PIDController()
        pid.reset()
        prev = hover_action(p)
        actions, wind = [], []
        out = {k: [] for k in ("pos", "vel", "quat", "omega", "time", "phase", "euler", "thrust", "power")}
        for _ in range(n_decisions):
            if i == 0:
                action = hover_action(p)
            else:
                obs = build_observation(state, [START], [], prev, 0.0, p.r_bird)
                action = np.array(_r(pid.predict(obs)[0] + noise * rng.standard_normal(5)))
            w = np.array(_r(mean_wind + gusts * rng.standard_normal(3))) if i else np.zeros(3)
            for _ in range(p.action_repeat):
                state, info = dyn.step(state, action, w)
            actions.append(action.tolist())
            wind.append(w.tolist())
            prev = np.clip(action, -1.0, 1.0)
            for key in ("pos", "vel", "quat", "omega"):
                out[key].append(_r(getattr(state, key)))
            out["time"].append(_r(state.time))
            out["phase"].append(_r(state.phase))
            out["euler"].append(_r(state.euler))
            out["thrust"].append(_r(info.thrust))
            out["power"].append(_r(info.power))
        cases.append({"params": params_dict(p), "mass_factor": mass_factor, "damage": list(damage),
                      "start": {"pos": start.pos.tolist(), "vel": start.vel.tolist(), "quat": start.quat.tolist(),
                                "omega": start.omega.tolist(), "time": 0.0, "phase": 0.0},
                      "actions": actions, "wind": wind, "expected": out})
    return {"format": "hummingbird-golden-dynamics/1", "n_decisions": n_decisions,
            "default_params": params_dict(DEFAULT_PARAMS), "derived": derived_dict(DEFAULT_PARAMS),
            "cases": cases, "limits": golden_limits()}


def golden_limits() -> list[dict]:
    """A few states on either side of each limit of the arena, with the expected verdict."""
    states = [
        BirdState.from_euler(pos=START),                                     # fine
        BirdState.from_euler(pos=(0.0, 0.019, 0.0)),                         # below 2 cm: crash
        BirdState.from_euler(pos=(1.2, 0.5, 0.91)),                          # 1.51 m from the centre: out
        BirdState.from_euler(pos=(0.0, 1.21, 0.0)),                          # above the ceiling: out
        BirdState.from_euler(pos=START, pitch=radians(79.0), roll=radians(-79.0)),   # tilted, still fine
        BirdState.from_euler(pos=START, pitch=radians(81.0)),                # loss of control
        BirdState.from_euler(pos=START, yaw=2.0, roll=radians(-81.0)),       # loss of control
    ]
    return [{"pos": s.pos.tolist(), "quat": s.quat.tolist(), "expected": check_limits(s)} for s in states]


def golden_observations(n_cases: int = 20, seed: int = 1) -> dict:
    """Bird state, targets, obstacles, previous action and hold in; the 48 observation values out.

    Obstacles are spread up to 1.5 m away so that some fall outside the 1 m
    range and, with up to 8 of them, the "5 nearest" rule is exercised; one
    target in four is far enough for its entries to hit the clip at ±5.
    No sensor noise: noise is random and is not part of the comparison.
    """
    rng = np.random.default_rng(seed)
    p = DEFAULT_PARAMS
    cases = []
    for _ in range(n_cases):
        state = _random_state(rng, big=bool(rng.uniform() < 0.5))
        far = 3.0 if rng.uniform() < 0.25 else 0.5
        targets = [np.array(_r(state.pos + far * rng.uniform(-1.0, 1.0, 3))) for _ in range(rng.integers(0, 3))]
        obstacles = [Obstacle(np.array(_r(state.pos + rng.uniform(-1.5, 1.5, 3))), _r(rng.uniform(0.03, 0.15)))
                     for _ in range(rng.integers(0, 9))]
        prev_action = np.array(_r(rng.uniform(-1.0, 1.0, 5)))
        hold = _r(rng.integers(0, 16) / 15.0)
        obs = build_observation(state, targets, obstacles, prev_action, hold, p.r_bird)
        cases.append({"state": {"pos": state.pos.tolist(), "vel": state.vel.tolist(), "quat": state.quat.tolist(),
                                "omega": state.omega.tolist()},
                      "targets": [t.tolist() for t in targets],
                      "obstacles": [{"center": o.center.tolist(), "radius": o.radius} for o in obstacles],
                      "prev_action": prev_action.tolist(), "hold": hold, "r_bird": p.r_bird,
                      "expected": [float(v) for v in obs]})
    return {"format": "hummingbird-golden-observations/1", "cases": cases}
