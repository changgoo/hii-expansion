"""Analytic solutions for HII region expansion."""

from __future__ import annotations

import numpy as np

from .constants import K_B, M_H
from .hii_region import stromgren_radius_uniform
from .recombination import alpha_B_case_B


def spitzer_solution(
    Q: float,
    n: float,
    T: float,
    t: np.ndarray | float,
    alpha_B: float | None = None,
) -> np.ndarray | float:
    """Spitzer (1978) analytic D-type expansion in a uniform-density medium.

    The quasi-static (zero-inertia) limit of the thin-shell momentum
    equation for constant density yields the first-order ODE::

        dR/dt = c_II (R_st / R)^(3/4)

    with exact solution::

        R(t) = R_st [1 + (7/4)(c_II / R_st) t]^(4/7)

    where ``c_II = sqrt(k_B T / m_H)`` is the isothermal sound speed.
    At late times ``R ∝ t^{4/7}``.

    Parameters
    ----------
    Q:
        Ionizing photon rate [s⁻¹].
    n:
        Uniform ambient number density [cm⁻³].
    T:
        HII region temperature [K].
    t:
        Time since the Stromgren sphere formed [s].
    alpha_B:
        Case B recombination coefficient [cm³ s⁻¹].  If *None*, derived
        from *T* via the Draine (2011) fitting formula.

    Returns
    -------
    np.ndarray | float
        HII region radius R(t) [cm], same shape as *t*.
    """
    if alpha_B is None:
        alpha_B = alpha_B_case_B(T)

    R_st = stromgren_radius_uniform(Q, n, alpha_B)
    c_II = np.sqrt(2.0 * K_B * T / M_H)  # sqrt(2 k_B T / m_H) for full pressure

    return R_st * (1.0 + (7.0 / 4.0) * (c_II / R_st) * t) ** (4.0 / 7.0)
