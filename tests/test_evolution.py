"""Tests for HIIRegion evolution and the Spitzer analytic solution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hii_expansion import (
    HIIRegion,
    alpha_B_case_B,
    spitzer_solution,
    stromgren_radius_uniform,
)
from hii_expansion.constants import K_B, M_H

# Fiducial parameters
Q_0: float = 1.0e49   # ionizing photon rate [s⁻¹]
N_0: float = 100.0    # number density [cm⁻³]
T_0: float = 1.0e4    # temperature [K]
A_B: float = alpha_B_case_B(T_0)

# Characteristic time t_dyn = R_st / c_II  (uses factor-of-2 pressure c_II)
_R_st = stromgren_radius_uniform(Q_0, N_0, A_B)
_c_II = np.sqrt(2.0 * K_B * T_0 / M_H)
T_DYN: float = _R_st / _c_II   # ~few × 10¹² s


class TestHIIRegionProperties:
    def test_c_II_value(self) -> None:
        """c_II = sqrt(2 k_B T / m_H) ≈ 12.8 km/s at T = 10⁴ K."""
        hii = HIIRegion(Q=Q_0, n=N_0, T=T_0)
        expected = np.sqrt(2.0 * K_B * T_0 / M_H)
        assert hii.c_II == pytest.approx(expected, rel=1e-10)

    def test_n_ionized_at_rst_equals_ambient(self) -> None:
        """At R_st, n_i equals the ambient density for constant n."""
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        r_st = hii.stromgren_radius()
        assert hii._n_ionized(r_st) == pytest.approx(N_0, rel=1e-6)

    def test_n_ionized_power_law(self) -> None:
        """n_i ∝ R^(-3/2) from ionisation balance."""
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        r_st = hii.stromgren_radius()
        n1 = hii._n_ionized(r_st)
        n2 = hii._n_ionized(2.0 * r_st)
        assert n2 / n1 == pytest.approx(2.0 ** (-1.5), rel=1e-10)

    def test_interior_pressure_factor_two(self) -> None:
        """P_in = 2 n_i k_B T: at R_st, P_in = 2 N_0 k_B T."""
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        r_st = hii.stromgren_radius()
        expected = 2.0 * N_0 * K_B * T_0
        assert hii._interior_pressure(r_st) == pytest.approx(expected, rel=1e-6)

    def test_interior_pressure_power_law(self) -> None:
        """P_in ∝ R^(-3/2): doubling R reduces P_in by 2^(3/2)."""
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        r_st = hii.stromgren_radius()
        p1 = hii._interior_pressure(r_st)
        p2 = hii._interior_pressure(2.0 * r_st)
        assert p2 / p1 == pytest.approx(2.0 ** (-1.5), rel=1e-10)

    def test_swept_mass_uniform_analytic(self) -> None:
        """For constant n, M_sh = (4π/3) R³ n m_H."""
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        r_st = hii.stromgren_radius()
        M_expected = (4.0 * np.pi / 3.0) * r_st**3 * N_0 * M_H
        assert hii._swept_mass(r_st) == pytest.approx(M_expected, rel=1e-5)


class TestSpitzerSolution:
    def test_starts_at_stromgren_radius(self) -> None:
        """R(t=0) = R_st exactly."""
        R_st = stromgren_radius_uniform(Q_0, N_0, A_B)
        R0 = spitzer_solution(Q_0, N_0, T_0, 0.0, alpha_B=A_B)
        assert R0 == pytest.approx(R_st, rel=1e-10)

    def test_monotonically_increasing(self) -> None:
        """R(t) must be strictly increasing for t > 0."""
        t = np.logspace(np.log10(T_DYN * 0.1), np.log10(T_DYN * 100), 50)
        R = spitzer_solution(Q_0, N_0, T_0, t, alpha_B=A_B)
        assert np.all(np.diff(R) > 0)

    def test_power_law_slope_four_sevenths(self) -> None:
        """At late times R ∝ t^(4/7): two-point slope agrees to 0.1%.

        Uses t >> T_dyn so the additive constant in (1 + 7/4 c t/R_st)
        is negligible.
        """
        t = np.array([1.0e4, 2.0e4]) * T_DYN
        R = spitzer_solution(Q_0, N_0, T_0, t, alpha_B=A_B)
        slope = np.log(R[1] / R[0]) / np.log(t[1] / t[0])
        assert slope == pytest.approx(4.0 / 7.0, rel=1e-3)

    def test_array_and_scalar_input(self) -> None:
        """Accepts both scalar and array time inputs."""
        t_scalar = 10.0 * T_DYN
        t_array = np.array([5.0, 10.0]) * T_DYN
        R_s = spitzer_solution(Q_0, N_0, T_0, t_scalar, alpha_B=A_B)
        R_a = spitzer_solution(Q_0, N_0, T_0, t_array, alpha_B=A_B)
        assert isinstance(R_s, float)
        assert R_a.shape == (2,)
        assert R_a[1] == pytest.approx(R_s, rel=1e-10)


class TestHIIRegionEvolve:
    def _run(self, **kwargs: object) -> object:
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        t_end = 20.0 * T_DYN
        return hii, hii.evolve((0.0, t_end), n_eval=300, rtol=1e-9, atol=0.0, **kwargs)

    def test_starts_at_stromgren_radius(self) -> None:
        """R(t=0) = R_st."""
        hii, sol = self._run()
        assert sol.y[0, 0] == pytest.approx(hii.stromgren_radius(), rel=1e-10)

    def test_initial_velocity_default_is_c_II(self) -> None:
        """Default initial velocity equals c_II."""
        hii, sol = self._run()
        assert sol.y[1, 0] == pytest.approx(hii.c_II, rel=1e-10)

    def test_radius_monotonically_increasing(self) -> None:
        """R(t) must be strictly increasing."""
        _, sol = self._run()
        assert np.all(np.diff(sol.y[0]) > 0)

    def test_mass_monotonically_increasing(self) -> None:
        """Swept mass M_sh(t) must be strictly increasing."""
        _, sol = self._run()
        assert np.all(np.diff(sol.y[2]) > 0)

    def test_power_law_slope_four_sevenths(self) -> None:
        """At late times R ∝ t^(4/7): log-log slope agrees to 2%."""
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        t_end = 100.0 * T_DYN
        sol = hii.evolve((0.0, t_end), n_eval=1000, rtol=1e-10, atol=0.0)

        # Fit slope over the last half of the integration (power-law regime)
        mask = sol.t > 0.5 * t_end
        log_t = np.log(sol.t[mask])
        log_R = np.log(sol.y[0, mask])
        slope = np.polyfit(log_t, log_R, 1)[0]
        assert slope == pytest.approx(4.0 / 7.0, rel=0.02)

    def test_agrees_with_spitzer_at_late_times(self) -> None:
        """Numerical R(t) matches the Spitzer analytic formula within 10%."""
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        t_end = 50.0 * T_DYN
        sol = hii.evolve((0.0, t_end), n_eval=500, rtol=1e-10, atol=0.0)

        R_num = sol.y[0, -1]
        R_sp = spitzer_solution(Q_0, N_0, T_0, t_end, alpha_B=A_B)
        assert R_num == pytest.approx(R_sp, rel=0.10)

    def test_custom_v0(self) -> None:
        """A user-supplied v0 is used as the initial velocity."""
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        v_custom = 0.5 * hii.c_II
        sol = hii.evolve((0.0, 5.0 * T_DYN), n_eval=100, v0=v_custom)
        assert sol.y[1, 0] == pytest.approx(v_custom, rel=1e-10)

    def test_failed_integration_raises(self) -> None:
        """If solve_ivp reports failure, evolve() raises RuntimeError."""
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        failed = MagicMock()
        failed.success = False
        failed.message = "mock failure"
        with patch("hii_expansion.hii_region.integrate.solve_ivp", return_value=failed):
            with pytest.raises(RuntimeError, match="ODE integration failed"):
                hii.evolve((0.0, T_DYN))
