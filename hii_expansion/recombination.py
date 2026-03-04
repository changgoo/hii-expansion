"""Case B hydrogen recombination coefficient."""

import numpy as np


def alpha_B_case_B(T: float) -> float:
    """Case B hydrogen recombination coefficient.

    Fitting formula from Draine (2011), eq. 14.6, accurate to ~1% for
    3000 K < T < 30 000 K.

    Parameters
    ----------
    T:
        Gas temperature [K].

    Returns
    -------
    float
        Case B recombination coefficient [cm³ s⁻¹].
    """
    t4 = T / 1.0e4
    return 2.54e-13 * t4 ** (-0.8163 - 0.0208 * np.log(t4))
