"""Tests for wind.py: the gusts have the intended mean, spread and duration."""

from math import exp

import numpy as np

from hummingbird_rl.sim.wind import GustProcess


def test_calm_wind_is_zero():
    g = GustProcess.calm()
    for _ in range(100):
        assert np.all(g.step(1 / 240) == 0)


def test_gusts_active_from_first_step():
    g = GustProcess(mean_force=np.array([0.004, 0.0, 0.0]), sigma=0.001, rng=np.random.default_rng(0))
    assert not np.allclose(g.state, 0.0)


def test_stationary_statistics():
    dt, tau, sigma = 1 / 240, 0.5, 0.002
    mean = np.array([0.005, 0.0, -0.003])
    g = GustProcess(mean_force=mean, sigma=sigma, tau=tau, rng=np.random.default_rng(1))
    xs = np.array([g.step(dt).copy() for _ in range(200_000)])   # ≈ 830 s, ≈ 1700 correlation times
    assert np.allclose(xs.mean(axis=0), mean, atol=3e-4)
    assert np.isclose(xs[:, 0].std(), sigma, rtol=0.05)
    assert np.isclose(xs[:, 2].std(), sigma, rtol=0.05)
    assert np.isclose(xs[:, 1].std(), 0.3 * sigma, rtol=0.05)
    # autocorrelation at a lag of one correlation time ≈ e^-1
    lag = int(round(tau / dt))
    x = xs[:, 0] - xs[:, 0].mean()
    rho = np.dot(x[:-lag], x[lag:]) / np.dot(x, x)
    assert abs(rho - exp(-1)) < 0.05


def test_seeded_process_is_reproducible():
    a = GustProcess(mean_force=np.zeros(3), sigma=0.001, rng=np.random.default_rng(7))
    b = GustProcess(mean_force=np.zeros(3), sigma=0.001, rng=np.random.default_rng(7))
    for _ in range(50):
        assert np.array_equal(a.step(1 / 240), b.step(1 / 240))
