"""Tests for params.py: the derived quantities match the targets we chose the parameters for."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_hover_amplitude_is_80_percent_of_max(params):
    # c_lift was chosen so that hovering leaves a 20% amplitude margin.
    assert params.hover_amplitude() / params.amp_max == pytest.approx(0.8, rel=0.01)


def test_heavier_bird_needs_more_amplitude(params):
    # Thrust grows with amplitude squared, so 20% more weight needs sqrt(1.2) more amplitude.
    ratio = params.hover_amplitude(1.2) / params.hover_amplitude()
    assert ratio == pytest.approx(np.sqrt(1.2))


def test_max_thrust_to_weight(params):
    # Full amplitude and 1.2 f0 versus 0.8 amplitude at f0: (1.2 / 0.8)^2 = 2.25.
    assert params.max_thrust() / params.weight == pytest.approx(2.25, rel=0.02)


def test_hover_power_order_of_magnitude(params):
    # A hovering hummingbird spends a fraction of a watt on aerodynamic work.
    assert 0.05 < params.hover_power() < 1.0


def test_inertia_is_positive_and_roll_is_easiest(params):
    ixx, iyy, izz = params.inertia
    assert min(ixx, iyy, izz) > 0
    assert ixx < izz < iyy          # long, thin body: rolling is easiest, yawing hardest
    assert izz / ixx == pytest.approx(3.8, rel=0.1)


def test_damping_gives_the_chosen_time_constant(params):
    assert np.allclose(params.inertia / params.damping, params.tau_omega)


def test_parameters_are_read_only_and_copy_with_changes(params):
    with pytest.raises(FrozenInstanceError):
        params.mass = 1.0
    stable = params.with_changes(k_u_pitch=0.0, k_u_roll=0.0)
    assert stable.k_u_pitch == 0.0 and params.k_u_pitch > 0      # the original is untouched
    assert stable.mass == params.mass


def test_default_frequency_range(params):
    assert params.f0 * (1 - params.freq_span) == pytest.approx(32.0)
    assert params.f0 * (1 + params.freq_span) == pytest.approx(48.0)
