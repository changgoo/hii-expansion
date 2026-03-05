"""Numerical vs analytic HII region expansion for constant ambient density.

Integrates the thin-shell momentum equation and overlays the Spitzer (1978)
analytic solution R(t) = R_st [1 + (7/4)(c_II/R_st) t]^{4/7}.

Run from the project root:
    python python/plot_constant_density.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hii_expansion import HIIRegion, alpha_B_case_B, spitzer_solution
from hii_expansion.constants import PC, YR

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Q = 1.0e49    # ionizing photon rate [s⁻¹], typical O3 star
n0 = 100.0    # ambient number density [cm⁻³]
T = 1.0e4     # HII region temperature [K]
alpha_B = alpha_B_case_B(T)

# ---------------------------------------------------------------------------
# Integrate
# ---------------------------------------------------------------------------
hii = HIIRegion(Q=Q, n=n0, T=T)
R_st = hii.stromgren_radius()
c_II = hii.c_II
T_dyn = R_st / c_II          # characteristic expansion time [s]

t_end = 50.0 * T_dyn
sol = hii.evolve((0.0, t_end), n_eval=500, rtol=1e-10, atol=0.0)
sol_mod = hii.evolve_modified((0.0, t_end), n_eval=500, rtol=1e-10, atol=0.0)

t_num = sol.t
R_num = sol.y[0]

# Analytic Spitzer solution at the same time points
R_sp = spitzer_solution(Q, n0, T, t_num, alpha_B=alpha_B)

# ---------------------------------------------------------------------------
# Convert to physical units (Myr, pc)
# ---------------------------------------------------------------------------
MYR = 1.0e6 * YR
t_Myr = t_num / MYR
R_num_pc = R_num / PC
R_sp_pc = R_sp / PC
R_st_pc = R_st / PC

# ---------------------------------------------------------------------------
# Reference t^{4/7} line anchored to the analytic curve at t = 5 T_dyn
# ---------------------------------------------------------------------------
i_anchor = np.searchsorted(t_num, 5.0 * T_dyn)
t_ref = np.array([t_num[i_anchor], t_end]) / MYR
R_ref = R_sp_pc[i_anchor] * (t_ref / t_Myr[i_anchor]) ** (4.0 / 7.0)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))

ax.loglog(t_Myr, R_num_pc, color="C0", lw=2.0, label="Classic ODE")
ax.loglog(sol_mod.t / MYR, sol_mod.y[0] / PC, color="C1", lw=2.0, ls="--", label="Modified ODE")
ax.loglog(t_Myr, R_sp_pc, color="gray", lw=1.5, ls=":", label="Spitzer (1978) analytic")
ax.loglog(t_ref, R_ref, color="k", lw=1.2, ls="-.", label=r"$\propto t^{4/7}$")

# Mark the initial Stromgren radius
ax.axhline(R_st_pc, color="gray", lw=0.8, ls="--")
ax.text(
    t_Myr[2],
    R_st_pc * 1.12,
    rf"$R_{{\rm st}} = {R_st_pc:.2f}$ pc",
    color="gray",
    fontsize=9,
)

ax.set_xlabel("Time [Myr]")
ax.set_ylabel(r"$R\;[\mathrm{pc}]$")
ax.set_title(
    rf"$Q = {Q:.0e}\ \mathrm{{s}}^{{-1}}$, "
    rf"$n_0 = {n0:.0f}\ \mathrm{{cm}}^{{-3}}$, "
    rf"$T = {T:.0e}\ \mathrm{{K}}$"
)
ax.legend(fontsize=9)
fig.tight_layout()

out = Path(__file__).parent.parent / "figures" / "constant_density.pdf"
fig.savefig(out)
print(f"Saved {out}")
