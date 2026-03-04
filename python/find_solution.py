"""Example: compute the Stromgren radius for a uniform and a power-law medium."""

import numpy as np

from hii_expansion import HIIRegion, alpha_B_case_B, stromgren_radius_uniform
from hii_expansion.constants import PC

# Fiducial parameters
Q = 1.0e49   # ionizing photon rate [s⁻¹], typical O3 star
n0 = 100.0   # ambient number density [cm⁻³]
T = 1.0e4    # HII region temperature [K]
alpha_B = alpha_B_case_B(T)

# --- Uniform density (analytic) ---
r_st = stromgren_radius_uniform(Q, n0, alpha_B)
print(f"Uniform medium:  R_st = {r_st:.3e} cm = {r_st / PC:.3f} pc")

# --- Same result via HIIRegion ---
hii_uniform = HIIRegion(Q=Q, n=n0, T=T)
assert np.isclose(hii_uniform.stromgren_radius(), r_st)

# --- Power-law density profile: n(r) = n0 * (r0 / r)^w ---
r0 = PC        # reference radius = 1 pc
w = 1.0        # density slope


def n_powerlaw(r: float) -> float:
    return n0 * (r0 / r) ** w


hii_pl = HIIRegion(Q=Q, n=n_powerlaw, T=T)
r_st_pl = hii_pl.stromgren_radius()
print(f"Power-law n~r^-{w}: R_st = {r_st_pl:.3e} cm = {r_st_pl / PC:.3f} pc")
