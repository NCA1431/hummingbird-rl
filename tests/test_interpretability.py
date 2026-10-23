"""E6 helpers: the states we build, the LQR landscape, and the comparison measures."""

import numpy as np
import pytest

from hummingbird_rl.analysis import interpretability as it
from hummingbird_rl.controllers import LQRController
from hummingbird_rl.envs import observation as ob


@pytest.fixture(scope="module")
def lqr():
    return LQRController()


def test_observation_at_an_offset_shows_the_target_on_the_other_side():
    obs = it.observation(it.offset_state(0.10, -0.05))
    assert obs[ob.ALTITUDE] == pytest.approx(0.45)
    np.testing.assert_allclose(obs[ob.TARGETS][:4], [-0.10 / ob.POS_SCALE, 0.05 / ob.POS_SCALE, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(obs[ob.VELOCITY], 0.0)


def test_lqr_landscape_is_best_at_hover_and_symmetric(lqr):
    L = it.value_landscape(None, lqr, n=11)
    v = L["lqr"]
    c = 5
    assert v[c, c] == pytest.approx(0.0) and v.max() == pytest.approx(0.0)
    np.testing.assert_allclose(v, v[::-1, ::-1], atol=1e-12)     # a quadratic form: V(x) = V(-x)


def test_rank_measures(lqr):
    L = it.value_landscape(None, lqr, n=11)
    xs = L["xs"]
    assert it.spearman(L["lqr"], -np.sqrt(-L["lqr"]), xs, it.HALF) == pytest.approx(1.0, abs=1e-4)  # order only
    assert it.spearman(L["lqr"], -L["lqr"], xs, it.HALF) == pytest.approx(-1.0)
    assert it.within(xs, it.NEAR).sum() == 3 * 3      # 3 cm steps: -3, 0 and 3 cm on each axis


def test_lqr_pulls_the_bird_back_vertically(lqr):
    xs, D = it.vector_field(lqr, n=5)
    assert np.all(D[-1, :, 1] < 0)        # above the target: goes down
    assert np.all(D[0, :, 1] > 0)         # below: goes up
    assert it.alignment(D, D, xs, it.HALF) == pytest.approx(1.0)
    assert it.alignment(D, -D, xs, it.HALF) == pytest.approx(-1.0)


def test_path_gap_is_zero_for_identical_paths(lqr):
    paths = [it.fly(lqr, it.offset_state(*s), 10) for s in it.trajectory_starts()[:2]]
    assert it.path_gap(paths, paths) == 0.0
    assert len(paths[0]) == 11
