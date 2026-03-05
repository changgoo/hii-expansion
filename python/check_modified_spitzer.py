"""Smoke test and comparison of classic vs. modified Spitzer ODE."""

import numpy as np

from hii_expansion import HIIRegion
from hii_expansion.constants import PC, YR

Q, n0, T = 1e49, 100.0, 1e4
hii = HIIRegion(Q=Q, n=n0, T=T)
R_st = hii.stromgren_radius()
T_dyn = R_st / hii.c_II
t_end = 50 * T_dyn
MYR = 1e6 * YR

sol_c = hii.evolve((0, t_end), n_eval=300, rtol=1e-10, atol=0)
sol_m = hii.evolve_modified((0, t_end), n_eval=300, rtol=1e-10, atol=0)

print(f"R_st = {R_st/PC:.3f} pc,  T_dyn = {T_dyn/MYR:.4f} Myr")
print()
print(f"{'':12s}  {'R_final [pc]':>14}  {'v_final [km/s]':>15}  {'M_sh_final [g]':>16}")
print("-" * 65)
for label, sol in [("Classic", sol_c), ("Modified", sol_m)]:
    R_f = sol.y[0, -1] / PC
    v_f = sol.y[1, -1] / 1e5
    M_f = sol.y[2, -1]
    print(f"{label:12s}  {R_f:>14.3f}  {v_f:>15.3f}  {M_f:>16.4e}")

print()
# Late-time power-law slope
for label, sol in [("Classic", sol_c), ("Modified", sol_m)]:
    t, R = sol.t, sol.y[0]
    i1, i2 = len(t) // 2, -1
    slope = np.log(R[i2] / R[i1]) / np.log(t[i2] / t[i1])
    print(f"{label} late-time slope = {slope:.4f}  (expected 4/7 = {4/7:.4f})")

print()
# Check M_sh: modified should be less than classic at all times
assert np.all(sol_m.y[2] <= sol_c.y[2] + 1e-10), "Modified M_sh should be <= classic"
print("M_sh check passed: modified <= classic at all times")

# Check M_sh_modified starts negligibly small (seed << classic initial M_sh)
ratio = sol_m.y[2, 0] / sol_c.y[2, 0]
assert ratio < 1e-5, (
    f"Modified M_sh seed should be negligible vs classic: ratio={ratio:.2e}"
)
print(f"Initial M_sh check passed: modified seed / classic = {ratio:.2e}")
