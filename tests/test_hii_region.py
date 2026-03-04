"""Tests for HIIRegion and stromgren_radius_uniform."""

import numpy as np
import pytest

from hii_expansion import HIIRegion, alpha_B_case_B, stromgren_radius_uniform

# Fiducial parameters: typical O-star + warm neutral medium
Q_0: float = 1.0e49   # ionizing photon rate [s⁻¹]
N_0: float = 100.0    # number density [cm⁻³]
T_0: float = 1.0e4    # temperature [K]
A_B: float = alpha_B_case_B(T_0)


class TestStromgrenRadiusUniform:
    def test_ionization_balance(self) -> None:
        """Recombination rate inside R_st must equal Q."""
        r_st = stromgren_radius_uniform(Q_0, N_0, A_B)
        recomb = (4.0 * np.pi / 3.0) * r_st**3 * N_0**2 * A_B
        assert recomb == pytest.approx(Q_0, rel=1e-10)

    def test_scales_as_q_one_third(self) -> None:
        """R_st ∝ Q^(1/3): doubling Q³ doubles R_st."""
        r1 = stromgren_radius_uniform(Q_0, N_0, A_B)
        r2 = stromgren_radius_uniform(8.0 * Q_0, N_0, A_B)
        assert r2 / r1 == pytest.approx(2.0, rel=1e-10)

    def test_scales_as_n_minus_two_thirds(self) -> None:
        """R_st ∝ n^(-2/3): multiplying n by 8 divides R_st by 4."""
        r1 = stromgren_radius_uniform(Q_0, N_0, A_B)
        r2 = stromgren_radius_uniform(Q_0, 8.0 * N_0, A_B)
        assert r2 / r1 == pytest.approx(0.25, rel=1e-10)

    def test_scales_as_alpha_minus_one_third(self) -> None:
        """R_st ∝ α_B^(-1/3): multiplying α_B by 8 divides R_st by 2."""
        r1 = stromgren_radius_uniform(Q_0, N_0, A_B)
        r2 = stromgren_radius_uniform(Q_0, N_0, 8.0 * A_B)
        assert r2 / r1 == pytest.approx(0.5, rel=1e-10)

    def test_positive(self) -> None:
        assert stromgren_radius_uniform(Q_0, N_0, A_B) > 0


class TestHIIRegionStromgrenConstantDensity:
    def test_matches_analytic(self) -> None:
        """Constant-density HIIRegion must match the analytic formula exactly."""
        hii = HIIRegion(Q=Q_0, n=N_0, T=T_0)
        r_class = hii.stromgren_radius()
        r_analytic = stromgren_radius_uniform(Q_0, N_0, hii.alpha_B)
        assert r_class == pytest.approx(r_analytic, rel=1e-10)

    def test_explicit_alpha_b(self) -> None:
        """Passing alpha_B explicitly overrides the temperature-derived value."""
        hii = HIIRegion(Q=Q_0, n=N_0, alpha_B=A_B)
        r_class = hii.stromgren_radius()
        r_analytic = stromgren_radius_uniform(Q_0, N_0, A_B)
        assert r_class == pytest.approx(r_analytic, rel=1e-10)

    def test_cached(self) -> None:
        """Calling stromgren_radius() twice returns the same object."""
        hii = HIIRegion(Q=Q_0, n=N_0, T=T_0)
        assert hii.stromgren_radius() is hii.stromgren_radius()


class TestHIIRegionStromgrenDensityProfile:
    def test_uniform_profile_matches_analytic(self) -> None:
        """A callable constant n(r) must agree with the analytic formula."""
        hii = HIIRegion(Q=Q_0, n=lambda r: N_0, alpha_B=A_B)  # noqa: E731
        r_numeric = hii.stromgren_radius()
        r_analytic = stromgren_radius_uniform(Q_0, N_0, A_B)
        assert r_numeric == pytest.approx(r_analytic, rel=1e-6)

    def test_ionization_balance_for_power_law(self) -> None:
        """For n(r) = n₀(r₀/r)^w the result must still satisfy Q = ∫ α n² dV."""
        n0 = N_0
        r0 = 3.086e18  # 1 pc
        w = 1.0

        def n_profile(r: float) -> float:
            return n0 * (r0 / r) ** w

        hii = HIIRegion(Q=Q_0, n=n_profile, alpha_B=A_B)
        r_st = hii.stromgren_radius()

        # Verify by direct integration
        from scipy.integrate import quad

        integrand = lambda r: n_profile(r) ** 2 * r**2  # noqa: E731
        integral, _ = quad(integrand, 0.0, r_st)
        recomb = 4.0 * np.pi * A_B * integral
        assert recomb == pytest.approx(Q_0, rel=1e-4)
