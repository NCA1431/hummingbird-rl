"""Tests for the classical controllers: the mixer, the potential field, PID and LQR."""

import numpy as np
import pytest

from hummingbird_rl.controllers import FieldGains, LQRController, ObsView, PIDController, allocate, repulsion
from hummingbird_rl.envs import make_env
from hummingbird_rl.envs import observation as ob
from hummingbird_rl.evaluation import run_episodes, summarise
from hummingbird_rl.sim import BirdState, aero, hover_action
from hummingbird_rl.world.scene import START, Obstacle


def produced(action, p):
    """Thrust and torques the nominal model gives for an action."""
    cmd = aero.decode_action(action, p)
    f_l, f_r = aero.wing_thrusts(cmd, p)
    return f_l + f_r, aero.control_torques(cmd, f_l, f_r, p)


@pytest.mark.parametrize("torque", [(0, 0, 0), (2e-4, 0, 0), (-1e-4, 5e-5, -3e-5), (0, -1e-4, 4e-5)])
def test_allocation_inverts_the_wing_model(params, torque):
    torque = np.array(torque, dtype=float)
    thrust, tau = produced(allocate(params.weight, torque, params), params)
    assert thrust == pytest.approx(params.weight, rel=1e-6)
    assert np.allclose(tau, torque, rtol=1e-6, atol=1e-12)


def test_allocation_of_hover_is_the_hover_action(params):
    assert np.allclose(allocate(params.weight, np.zeros(3), params), hover_action(params))


def test_allocation_raises_frequency_only_when_needed(params):
    assert allocate(1.2 * params.weight, np.zeros(3), params)[2] == 0.0       # amplitude is enough
    assert allocate(2.0 * params.weight, np.zeros(3), params)[2] > 0.0        # beyond 1.56 W: faster beats


def test_obs_view_recovers_physical_units(params):
    state = BirdState.from_euler(pos=START, vel=(0.1, 0, 0), roll=0.1, omega=(1.0, 0, 0))
    obs = ob.build_observation(state, [START + [0.2, 0, 0]], [Obstacle(START + [0.4, 0, 0], 0.05)],
                               np.zeros(5), 0.5, params.r_bird)
    v = ObsView.parse(obs)
    assert np.allclose(v.vel, [0.1, 0, 0], atol=1e-6) and v.roll == pytest.approx(0.1, abs=1e-6)
    assert np.allclose(v.targets[0], [0.2, 0, 0], atol=1e-6) and len(v.targets) == 1
    assert np.allclose(v.obstacles[0][0], [0.4, 0, 0], atol=1e-6) and v.obstacles[0][1] == pytest.approx(0.05)
    assert v.rates[0] == pytest.approx(1.0, abs=1e-5) and v.hold == 0.5


@pytest.mark.parametrize("task,regime,minimum", [("hover", "normal", 0.9), ("goto", "normal", 0.9),
                                                 ("hover", "randomised", 0.7)])
def test_pid_flies(task, regime, minimum):
    """The baseline must be a reasonable one, otherwise beating it means nothing."""
    stats = summarise(run_episodes(make_env(task, regime), PIDController(), range(500, 520)))
    assert stats["success_rate"] >= minimum


def test_predict_handles_a_batch():
    env = make_env("hover", seed=0)
    obs, _ = env.reset(seed=0)
    pid = PIDController()
    single, _ = pid.predict(obs)
    batch, _ = PIDController().predict(np.stack([obs, obs]), episode_start=np.array([True, True]))
    assert batch.shape == (2, 5) and np.allclose(batch[0], single) and np.allclose(batch[1], single)


# --- potential field -----------------------------------------------------------------

def _view_with_obstacle(params, centre, radius=0.05, target=(0.3, 0.0, 0.0)):
    obs = ob.build_observation(BirdState.from_euler(pos=START), [START + np.array(target)],
                               [Obstacle(START + np.array(centre), radius)], np.zeros(5), 0.0, params.r_bird)
    return ObsView.parse(obs)


def test_field_ignores_far_obstacles_and_pushes_away_from_close_ones(params):
    far = _view_with_obstacle(params, (0.5, 0.0, 0.0))
    assert not repulsion(far, params.r_bird, FieldGains()).any()
    close = _view_with_obstacle(params, (0.12, 0.0, 0.0))        # 5 cm gap, straight ahead
    push = repulsion(close, params.r_bird, FieldGains())
    assert push[0] < 0                                           # backwards, away from it
    assert np.linalg.norm(push) == pytest.approx(FieldGains().a_max)


# --- LQR -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lqr():
    return LQRController()


def test_linearisation_is_unstable_and_controllable(lqr):
    """The open loop must show the hover instability we calibrated; the system must be controllable."""
    A, B = lqr.A, lqr.B
    worst = np.abs(np.linalg.eigvals(A)).max()
    doubling = np.log(2) / (np.log(worst) / lqr.dt)
    assert 0.5 < doubling < 1.0                                  # same 0.5-1 s as the physics tests
    ctrb = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(A.shape[0])])
    assert np.linalg.matrix_rank(ctrb) == A.shape[0]


def test_lqr_closed_loop_is_stable(lqr):
    assert np.abs(np.linalg.eigvals(lqr.closed_loop)).max() < 1.0


def test_cost_to_go_is_a_bowl_centred_on_hover(lqr, rng):
    assert lqr.cost_to_go(np.zeros(11)) == 0.0
    for _ in range(20):
        x = 0.01 * rng.standard_normal(11)
        assert lqr.cost_to_go(x) > 0 and lqr.cost_to_go(x) == pytest.approx(lqr.cost_to_go(-x))


@pytest.mark.parametrize("task", ["hover", "goto"])
def test_lqr_flies(lqr, task):
    stats = summarise(run_episodes(make_env(task, "normal"), lqr, range(500, 520)))
    assert stats["success_rate"] >= 0.9
