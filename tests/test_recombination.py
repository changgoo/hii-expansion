"""Tests for the Case B recombination coefficient."""

import numpy as np
import pytest

from hii_expansion import alpha_B_case_B


class TestAlphaBCaseB:
    def test_fiducial_value(self) -> None:
        """At T = 10⁴ K the formula should return exactly 2.54e-13 cm³/s."""
        assert alpha_B_case_B(1.0e4) == pytest.approx(2.54e-13)

    def test_agrees_with_osterbrock_at_1e4(self) -> None:
        """Commonly cited value (Osterbrock 2006) is ~2.6e-13; within 5%."""
        assert alpha_B_case_B(1.0e4) == pytest.approx(2.6e-13, rel=0.05)

    def test_agrees_with_storey_hummer_8000k(self) -> None:
        """Tabulated Storey & Hummer (1995) value at 8 000 K is ~3.03e-13."""
        assert alpha_B_case_B(8.0e3) == pytest.approx(3.03e-13, rel=0.02)

    def test_agrees_with_storey_hummer_20000k(self) -> None:
        """Tabulated Storey & Hummer (1995) value at 20 000 K is ~1.43e-13."""
        assert alpha_B_case_B(2.0e4) == pytest.approx(1.43e-13, rel=0.02)

    def test_decreases_with_temperature(self) -> None:
        """α_B is a decreasing function of T."""
        temperatures = [5.0e3, 1.0e4, 2.0e4, 3.0e4]
        values = [alpha_B_case_B(T) for T in temperatures]
        assert all(v1 > v2 for v1, v2 in zip(values, values[1:]))

    def test_positive_over_valid_range(self) -> None:
        """α_B must be positive for all T in the valid range."""
        for T in np.logspace(np.log10(3e3), np.log10(3e4), 20):
            assert alpha_B_case_B(T) > 0
