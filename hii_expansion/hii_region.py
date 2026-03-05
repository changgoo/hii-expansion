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
        integration_points: list[float] | None = None,
        max_radius: float | None = None,
    ) -> None:
        self.Q = Q
        self.T = T
        self.alpha_B: float = alpha_B if alpha_B is not None else alpha_B_case_B(T)
        self._integration_points: list[float] = integration_points or []
        # Default max radius: 300 pc (ISM gas scale height; larger R_st is unphysical)
        from .constants import PC
        self.max_radius: float = max_radius if max_radius is not None else 300.0 * PC

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

        if r_st > self.max_radius:
            from .constants import PC
            msg = (
                f"Stromgren radius ({r_st/PC:.1f} pc) exceeds max_radius "
                f"({self.max_radius/PC:.0f} pc): treating as density-bounded"
            )
            raise RuntimeError(msg)

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

    def evolve_modified(
        self,
        t_span: tuple[float, float],
        n_eval: int = 500,
        v0: float | None = None,
        **ivp_kwargs: object,
    ) -> OdeResult:
        """Evolve the HII region using the modified thin-shell ODE.

        Applies two corrections over :meth:`evolve` (classic Spitzer):

        1. **Initial shell mass = 0**: the gas already ionized within R_st
           does not contribute to the shell momentum.
        2. **Ionization mass exchange**: as the ionization front advances,
           gas transfers continuously from the shell into the bubble interior::

               dM_sh/dt = 2π R² m_H v [2 n(R) − n_i(R)]

           instead of ``4π R² n(R) m_H v``.  The momentum equation is
           unchanged.

        Parameters and return value are identical to :meth:`evolve`.
        """
        R0 = self.stromgren_radius()
        v_init = self.c_II if v0 is None else v0
        # Shell starts physically empty; a tiny nonzero seed avoids division
        # by zero in scipy's initial step-size estimator when atol=0.
        M_sh_0 = 4.0 * np.pi * M_H * self._n_func(R0) * R0**2 * R0 * 1e-10

        t_eval = np.linspace(t_span[0], t_span[1], n_eval)

        sol = integrate.solve_ivp(
            self._ode_rhs_modified,
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

        Computes ``α_B · 4π ∫₀^R n(r)² r² dr``.  Returns ``np.inf`` if the
        integrand overflows (signals that *R* is unphysically large).
        """
        pts = [p for p in self._integration_points if 0.0 < p < R] or None
        try:
            result, _ = integrate.quad(
                self._recomb_integrand, 0.0, R, limit=500, points=pts
            )
        except OverflowError:
            return np.inf
        return 4.0 * np.pi * self.alpha_B * result

    def _stromgren_objective(self, R: float) -> float:
        return self._recomb_rate(R) - self.Q

    def _stromgren_radius_numeric(self) -> float:
        """Find R_st by bracketed root-finding."""
        from .constants import PC

        r_hi = PC
        for _ in range(300):
            rate = self._recomb_rate(r_hi)
            if np.isinf(rate):
                # r_hi grew so large it overflowed without rate reaching Q:
                # the medium is density-bounded (total recombination < Q).
                msg = (
                    "No finite Stromgren radius: medium is density-bounded "
                    "(total recombination rate < Q)"
                )
                raise RuntimeError(msg)
            if rate >= self.Q:
                break
            r_hi *= 10.0
        else:
            msg = "Could not bracket Stromgren radius within 300 steps"
            raise RuntimeError(msg)

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
        pts = [p for p in self._integration_points if 0.0 < p < R] or None
        result, _ = integrate.quad(
            self._mass_integrand, 0.0, R, limit=500, points=pts
        )
        return 4.0 * np.pi * M_H * result

    def _ode_rhs(self, _t: float, y: np.ndarray) -> np.ndarray:
        """RHS of the classic thin-shell ODE: state = [R, v, M_sh]."""
        R, v, M_sh = y
        n_R = self._n_func(R)
        P_in = self._interior_pressure(R)
        area = 4.0 * np.pi * R**2

        dR_dt = v
        dv_dt = area * (P_in - n_R * M_H * v**2) / M_sh
        dM_sh_dt = area * n_R * M_H * v

        return np.array([dR_dt, dv_dt, dM_sh_dt])

    def _ode_rhs_modified(self, _t: float, y: np.ndarray) -> np.ndarray:
        """RHS of the modified thin-shell ODE: state = [R, v, M_sh].

        Two corrections over the classic ODE:

        1. The initial shell mass is zero (gas within R_st is already ionized).
        2. The mass equation accounts for gas continuously transferred from the
           shell into the ionized bubble interior as the ionization front
           advances::

               dM_sh/dt = 2π R² m_H v [2 n(R) − n_i(R)]

           Derivation: shell mass = swept mass from R_st − ΔM_ionized, where
           ΔM_ionized = (4π/3) m_H [R³ n_i(R) − R_st³ n_i(R_st)].
           Differentiating gives d(ΔM_ion)/dt = 2π R² m_H n_i v, so
           dM_sh/dt = 4π R² n m_H v − 2π R² m_H n_i v.

           The momentum equation is *unchanged*: ionized gas leaves the shell
           at the shell velocity, so the momentum-flux terms cancel.

        At t = 0, M_sh = 0 and the pressure force is also exactly zero
        (v₀ = c_II ⟹ P_in = n m_H c_II²), so dv/dt = 0/0 → 0.  A small
        guard on M_sh prevents a literal divide-by-zero.
        """
        R, v, M_sh = y
        n_R = self._n_func(R)
        n_i = self._n_ionized(R)
        P_in = self._interior_pressure(R)

        dR_dt = v
        M_sh_safe = max(M_sh, 1e-100)  # force is 0 at t=0, so dv/dt → 0
        dv_dt = 4.0 * np.pi * R**2 * (P_in - n_R * M_H * v**2) / M_sh_safe
        dM_sh_dt = 2.0 * np.pi * R**2 * M_H * v * (2.0 * n_R - n_i)

        return np.array([dR_dt, dv_dt, dM_sh_dt])
