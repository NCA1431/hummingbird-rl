"""Uncertainty on our numbers.

A success rate measured on n episodes is a noisy estimate: 18 successes out
of 20 could come from a controller whose true rate is anywhere between about
70% and 97%. We report a 95% interval with every rate, so that two numbers
are only called different when their intervals say so.

For a proportion we use the Wilson score interval rather than the textbook
p ± 1.96·sqrt(p(1-p)/n): the textbook one gives impossible intervals near 0%
and 100% (exactly where good controllers live) and an interval of width zero
when every episode succeeds.
"""

from __future__ import annotations

from math import sqrt

import numpy as np

Z95 = 1.959964   # 95% quantile of the standard normal distribution


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% interval for a success rate measured as `successes` out of `n`."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_interval(values, z: float = Z95) -> tuple[float, float, float]:
    """Mean of `values` with a normal-approximation 95% interval: (mean, low, high)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"),) * 3
    m = float(v.mean())
    half = z * float(v.std(ddof=1)) / sqrt(len(v)) if len(v) > 1 else 0.0
    return (m, m - half, m + half)
