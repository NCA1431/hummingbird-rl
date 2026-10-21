"""Tests for the evaluation tools: intervals, recovery time, bearings, and the suites themselves."""

import numpy as np
import pytest

from hummingbird_rl.controllers import PIDController
from hummingbird_rl.envs.disturbances import Conditions
from hummingbird_rl.evaluation import mean_interval, wilson_interval
from hummingbird_rl.evaluation.suites import (
    GRID_DAMAGES,
    GRID_WINDS,
    bearing_deg,
    evaluate_behaviour,
    evaluate_failure,
    evaluate_grid,
    evaluate_test,
    recovery_time,
)


def test_wilson_interval_known_values():
    low, high = wilson_interval(18, 20)
    assert low == pytest.approx(0.699, abs=0.005) and high == pytest.approx(0.972, abs=0.005)
    low, high = wilson_interval(20, 20)              # all successes: still a real interval
    assert high == 1.0 and 0.8 < low < 0.9
    assert wilson_interval(0, 10)[0] == 0.0


def test_mean_interval_shrinks_with_more_data(rng):
    small = mean_interval(rng.normal(5, 1, 10))
    large = mean_interval(rng.normal(5, 1, 1000))
    assert small[1] < 5 < small[2] and (large[2] - large[1]) < (small[2] - small[1])


def test_recovery_time_on_a_made_up_flight():
    """Failure at 1.0 s; the bird is 10 cm off until 1.5 s, then back within 1 cm: 0.5 s to recover."""
    t = np.arange(1, 181) / 60.0
    d = np.where((t >= 1.0) & (t < 1.5), 0.10, 0.01)
    result = {"trace": {"t": t, "distance": d}, "conditions": Conditions(failure_time=1.0)}
    assert recovery_time(result) == pytest.approx(0.5, abs=1 / 60)
    never = {"trace": {"t": t, "distance": np.where(t >= 1.0, 0.10, 0.01)}, "conditions": Conditions(failure_time=1.0)}
    assert np.isnan(recovery_time(never))


BEARINGS = [((1, 0, 0), 0.0), ((0, 0, -1), 90.0), ((0, 0, 1), -90.0), ((-1, 0, 0), 180.0)]


@pytest.mark.parametrize("rel, expected", BEARINGS)
def test_bearing_sign_convention(rel, expected):
    """Ahead is 0°, the left (-z) is +90° like a positive yaw, behind is ±180°."""
    got = bearing_deg(np.array(rel, dtype=float))
    assert got == pytest.approx(expected) or abs(got) == pytest.approx(180.0) == abs(expected)


def test_test_suite_rows_and_intervals():
    rows, summary = evaluate_test(PIDController(), "hover", {}, n=5)
    assert len(rows) == 5 and "trace" not in rows[0]
    assert summary["success_low"] <= summary["success_rate"] <= summary["success_high"]


def test_grid_covers_every_cell():
    cells, summary = evaluate_grid(PIDController(), "hover", {}, n=1)
    assert len(cells) == len(GRID_WINDS) * len(GRID_DAMAGES)
    assert 0.0 <= summary["robust_area"] <= 1.0


def test_behaviour_suite_on_the_pid():
    """The PID never turns: for a target behind, it backs up."""
    with pytest.raises(ValueError):
        evaluate_behaviour(PIDController(), "hover", {}, n=2)
    rows, summary = evaluate_behaviour(PIDController(), "goto", {}, n=20)
    assert summary["p_turn_target_behind"] == 0.0 and summary["p_backed_target_behind"] > 0.8


def test_failure_suite_refuses_go_to_point():
    """Go-to-point episodes end before the failure time, so the suite would measure nothing."""
    with pytest.raises(ValueError):
        evaluate_failure(PIDController(), "goto", {}, n=2)
