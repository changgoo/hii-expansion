"""Report Stromgren radius and n_rms for cored density profiles.

n(r) = n0 / (1 + (r/r0)^w)

The Stromgren radius requires a 1-D root-find (brentq) on the recombination
integral -- no iteration needed.

n_rms follows directly from ionization balance:
  Q = 4pi alpha_B int_0^R_st n^2 r^2 dr
    = (4pi/3) R_st^3 alpha_B n_rms^2
=> n_rms = sqrt(3 Q / (4 pi alpha_B R_st^3))
"""

import numpy as np

from hii_expansion import HIIRegion, alpha_B_case_B, stromgren_radius_uniform
from hii_expansion.constants import K_B, M_H, PC

Q = 1.0e49
n0 = 100.0
T = 1.0e4
alpha_B = alpha_B_case_B(T)

R0_VALUES = [1.0 * PC, 5.0 * PC, 10.0 * PC]
W_VALUES = [1.0, 1.5, 2.0]

# Uniform reference (w=0)
R_st_uni = stromgren_radius_uniform(Q, n0, alpha_B)

print(f"Uniform (w=0):  R_st = {R_st_uni/PC:.3f} pc,  n_rms = n0 = {n0:.1f} cm^-3")
print()
print(f"{'r0 [pc]':>8}  {'w':>4}  {'R_st [pc]':>10}  {'n_rms [cm^-3]':>14}  {'n_rms/n0':>9}")
print("-" * 55)

for r0 in R0_VALUES:
    r0_pc = r0 / PC
    for w in W_VALUES:
        def n_profile(r: float, _w: float = w, _r0: float = r0) -> float:
            return n0 / (1.0 + (r / _r0) ** _w)

        hii = HIIRegion(Q=Q, n=n_profile, T=T, integration_points=[r0])
        try:
            R_st = hii.stromgren_radius()
        except RuntimeError as exc:
            print(f"{r0_pc:>8.0f}  {w:>4.1f}  {'N/A (density-bounded)':>37}")
            continue
        n_rms = np.sqrt(3.0 * Q / (4.0 * np.pi * alpha_B * R_st**3))
        print(f"{r0_pc:>8.0f}  {w:>4.1f}  {R_st/PC:>10.3f}  {n_rms:>14.2f}  {n_rms/n0:>9.4f}")
    print()
