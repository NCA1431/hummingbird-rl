"""Wind gusts, modelled as an Ornstein-Uhlenbeck process around a mean wind.

Real gusts are not white noise: a gust builds up and fades over a fraction of
a second. An Ornstein-Uhlenbeck (OU) process captures exactly that. The wind
w keeps being pulled back towards its mean w̄ while random kicks push it away:

    dw = -(w - w̄) / tau · dt  +  sigma · sqrt(2 / tau) · dW

tau is the correlation time (how long a gust lasts, 0.5 s here) and sigma the
typical size of the fluctuations: in steady state each horizontal component
has standard deviation sigma. Vertical gusts are weaker (0.3 sigma) and have
zero mean.

We model wind as a force on the bird, in newtons. In this simplified bird the
air drag on the body is too small for realistic wind speeds to matter, so a
force (expressed elsewhere as a percentage of the bird's weight) makes the
difficulty easier to interpret. Its limitation: wind does not change how the
wings themselves work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sqrt

import numpy as np


@dataclass
class GustProcess:
    mean_force: np.ndarray                     # N, mean wind w̄ (horizontal)
    sigma: float = 0.0                         # N, size of the horizontal fluctuations
    tau: float = 0.5                           # s, correlation time of a gust
    vertical_ratio: float = 0.3                # vertical fluctuations are this fraction of sigma
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    state: np.ndarray = field(init=False)      # N, current wind force

    def __post_init__(self) -> None:
        self.mean_force = np.asarray(self.mean_force, dtype=float)
        self._sig = np.array([self.sigma, self.vertical_ratio * self.sigma, self.sigma])
        # Start from the steady-state distribution, so the gusts are already
        # fully developed at the first step instead of ramping up.
        self.state = self.mean_force + self._sig * self.rng.standard_normal(3)

    @classmethod
    def calm(cls) -> "GustProcess":
        """No wind at all."""
        return cls(mean_force=np.zeros(3), sigma=0.0)

    def step(self, dt: float) -> np.ndarray:
        """Advance by dt and return the new wind force (N).

        We use the exact solution of the OU equation over one step rather than
        a simple Euler update, so the statistics do not depend on dt:
            w ← w̄ + (w - w̄) e^(-dt/tau) + sigma · sqrt(1 - e^(-2 dt/tau)) · noise
        """
        if self.tau <= 0.0:
            return self.state
        decay = exp(-dt / self.tau)
        noise = sqrt(1.0 - decay * decay)
        self.state = (self.mean_force + (self.state - self.mean_force) * decay
                      + self._sig * noise * self.rng.standard_normal(3))
        return self.state
