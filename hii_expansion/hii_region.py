"""HIIRegion class and Stromgren radius calculations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate, optimize

from .constants import K_B, M_H

if TYPE_CHECKING:
    from scipy.integrate._ivp.ivp import OdeResult
from .recombination import alpha_B_case_B


def stromgren_radius_uniform(Q: float, n: float, alpha_B: float) -> float:
    """Analytic Stromgren radius for a uniform-density medium.

    Derived from the ionization balance condition::

        Q = (4π/3) R_st³ n² α_B

    Parameters
    ----------
    Q:
        Ionizing photon rate [s⁻¹].
    n:
        Hydrogen number density [cm⁻³].
    alpha_B:
        Case B recombination coefficient [cm³ s⁻¹].

    Returns
    -------
    float
        Stromgren radius [cm].
    """
    return (3.0 * Q / (4.0 * np.pi * n**2 * alpha_B)) ** (1.0 / 3.0)


class HIIRegion:
    """HII region expanding into a prescribed hydrogen density field.

    Parameters
    ----------
    Q:
        Ionizing photon rate [s⁻¹].
    n:
        Hydrogen number density [cm⁻³].  Either a constant scalar or a
        radial profile ``n(r)`` where ``r`` is in cm.
    alpha_B:
        Case B recombination coefficient [cm³ s⁻¹].  If *None*, computed
        from *T* using the Draine (2011) fitting formula.
    T:
        Gas temperature [K].  Used only when *alpha_B* is *None*.
    """

    def __init__(
        self,
        Q: float,
        n: float | Callable[[float], float],
        alpha_B: float | None = None,
        T: float = 1.0e4,
    ) -> None:
        self.Q = Q
        self.T = T
        self.alpha_B: float = alpha_B if alpha_B is not None else alpha_B_case_B(T)

        if callable(n):
            self._n_func: Callable[[float], float] = n
            self._n_const: float | None = None
        else:
            n_val = float(n)

            def _const(r: float) -> float:  # avoids E731 (lambda assign)
                return n_val

            self._n_func = _const
            self._n_const = n_val

        self._r_st: float | None = None  # lazy cache

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def n(self, r: float) -> float:
        """Hydrogen number density at radius *r* [cm⁻³]."""
        return self._n_func(r)

    @property
    def c_II(self) -> float:
        """Isothermal sound speed in the ionized region [cm/s].

        Defined as ``sqrt(2 k_B T / m_H)``, consistent with the fully
        ionised-hydrogen interior pressure ``P_in = 2 n_i k_B T``
        (equal contributions from ions and electrons).
        """
        return np.sqrt(2.0 * K_B * self.T / M_H)

    def stromgren_radius(self) -> float:
        """Compute the Stromgren radius [cm].

        For a uniform density the exact analytic result is returned.
        For a radial profile ``n(r)`` the radius is found by numerically
        solving::

            Q = 4π α_B ∫₀^{R_st} n(r)² r² dr

        The result is cached after the first call.

        Returns
        -------
        float
            Stromgren radius [cm].
        """
        if self._r_st is not None:
            return self._r_st

        if self._n_const is not None:
            r_st = stromgren_radius_uniform(self.Q, self._n_const, self.alpha_B)
        else:
            r_st = self._stromgren_radius_numeric()

        self._r_st = r_st
        return r_st

    def evolve(
        self,
        t_span: tuple[float, float],
        n_eval: int = 500,
        v0: float | None = None,
        **ivp_kwargs: object,
    ) -> OdeResult:
        """Evolve the HII region from the Stromgren sphere forward in time.

        Integrates the thin-shell momentum equation::

            d(M_sh dR/dt)/dt = 4π R² P_in

        where the swept-up shell mass ``M_sh`` and shell velocity ``v =
        dR/dt`` are evolved jointly with::

            dM_sh/dt = 4π R² n(R) m_H v

        The interior pressure is set by instantaneous ionisation balance::

            P_in = n_i k_B T,   n_i = sqrt(3 Q / (4π α_B R³))

        The ODE state vector is ``[R, v, M_sh]``.

        Parameters
        ----------
        t_span:
            ``(t_start, t_end)`` integration interval [s].  ``t_start``
            is typically 0 (immediately after the Stromgren sphere forms).
        n_eval:
            Number of evenly-spaced output time points.
        v0:
            Initial shell velocity [cm/s].  Defaults to ``c_II`` (the
            D-critical condition at the Stromgren radius).
        **ivp_kwargs:
            Extra keyword arguments forwarded to
            ``scipy.integrate.solve_ivp``.  Common options:
            ``method`` (default ``'RK45'``), ``rtol``, ``atol``.

        Returns
        -------
        OdeResult
            scipy ODE result.  ``sol.t`` contains the time array and
            ``sol.y`` has rows ``[R(t), v(t), M_sh(t)]``.

        Raises
        ------
        RuntimeError
            If the integrator does not converge.
        """
        R0 = self.stromgren_radius()
        v_init = self.c_II if v0 is None else v0
        M_sh_0 = self._swept_mass(R0)

        t_eval = np.linspace(t_span[0], t_span[1], n_eval)

        sol = integrate.solve_ivp(
            self._ode_rhs,
            t_span,
            [R0, v_init, M_sh_0],
            t_eval=t_eval,
            **ivp_kwargs,
        )

        if not sol.success:
            msg = f"ODE integration failed: {sol.message}"
            raise RuntimeError(msg)

        return sol

    # ------------------------------------------------------------------
    # Internal helpers — Stromgren radius
    # ------------------------------------------------------------------

    def _recomb_integrand(self, r: float) -> float:
        return self._n_func(r) ** 2 * r**2

    def _recomb_rate(self, R: float) -> float:
        """Cumulative recombination rate inside radius *R* [s⁻¹].

        Computes ``α_B · 4π ∫₀^R n(r)² r² dr``.
        """
        result, _ = integrate.quad(self._recomb_integrand, 0.0, R, limit=200)
        return 4.0 * np.pi * self.alpha_B * result

    def _stromgren_objective(self, R: float) -> float:
        return self._recomb_rate(R) - self.Q

    def _stromgren_radius_numeric(self) -> float:
        """Find R_st by bracketed root-finding."""
        from .constants import PC

        r_hi = PC
        while self._recomb_rate(r_hi) < self.Q:
            r_hi *= 10.0

        return optimize.brentq(self._stromgren_objective, 0.0, r_hi)

    # ------------------------------------------------------------------
    # Internal helpers — evolution
    # ------------------------------------------------------------------

    def _n_ionized(self, R: float) -> float:
        """Effective ionized gas density from instantaneous ionisation balance.

        Derived from ``Q = (4π/3) R³ n_i² α_B``, giving
        ``n_i = sqrt(3 Q / (4π α_B R³))``.
        """
        return np.sqrt(3.0 * self.Q / (4.0 * np.pi * self.alpha_B * R**3))

    def _interior_pressure(self, R: float) -> float:
        """Interior gas pressure [dyn cm⁻²]: ``P_in = 2 n_i k_B T``.

        Factor of 2 accounts for equal electron and ion contributions in
        a fully ionised hydrogen plasma.
        """
        return 2.0 * self._n_ionized(R) * K_B * self.T

    def _mass_integrand(self, r: float) -> float:
        return self._n_func(r) * r**2

    def _swept_mass(self, R: float) -> float:
        """Total ambient mass swept up within radius *R* [g].

        ``M_sh = 4π m_H ∫₀^R n(r) r² dr``
        """
        result, _ = integrate.quad(self._mass_integrand, 0.0, R, limit=200)
        return 4.0 * np.pi * M_H * result

    def _ode_rhs(self, _t: float, y: np.ndarray) -> np.ndarray:
        """RHS of the thin-shell ODE: state = [R, v, M_sh]."""
        R, v, M_sh = y
        n_R = self._n_func(R)
        P_in = self._interior_pressure(R)
        area = 4.0 * np.pi * R**2

        dR_dt = v
        dv_dt = area * (P_in - n_R * M_H * v**2) / M_sh
        dM_sh_dt = area * n_R * M_H * v

        return np.array([dR_dt, dv_dt, dM_sh_dt])
