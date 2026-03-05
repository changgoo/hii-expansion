"""Classic vs. modified Spitzer thin-shell ODE comparison.

Generates a 3-panel figure comparing classic and modified HII region
expansion for uniform density:
  - Top:    R(t) [pc]
  - Middle: v(t) [km/s]
  - Bottom: M_sh(t) [Msun]

Run from the project root:
    python python/plot_classic_vs_modified.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hii_expansion import HIIRegion, alpha_B_case_B, spitzer_solution
from hii_expansion.constants import PC, YR

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Q = 1.0e49    # ionizing photon rate [s⁻¹]
n0 = 100.0    # ambient number density [cm⁻³]
T = 1.0e4     # HII region temperature [K]
alpha_B = alpha_B_case_B(T)

M_SUN = 1.989e33   # solar mass [g]
KM_S = 1.0e5       # km/s in cm/s
MYR = 1.0e6 * YR

# ---------------------------------------------------------------------------
# Integrate
# ---------------------------------------------------------------------------
hii = HIIRegion(Q=Q, n=n0, T=T)
R_st = hii.stromgren_radius()
c_II = hii.c_II
T_dyn = R_st / c_II

t_end = 50.0 * T_dyn
ivp_kw = dict(rtol=1e-10, atol=0.0)

sol_classic  = hii.evolve(         (0.0, t_end), n_eval=1000, **ivp_kw)
sol_modified = hii.evolve_modified((0.0, t_end), n_eval=1000, **ivp_kw)

# Analytic Spitzer solution
R_sp = spitzer_solution(Q, n0, T, sol_classic.t, alpha_B=alpha_B)

# Reference t^{4/7} slope anchored at 5 T_dyn on the analytic curve
i_anchor = np.searchsorted(sol_classic.t, 5.0 * T_dyn)
t_Myr_anchor = sol_classic.t[i_anchor] / MYR
R_anchor = R_sp[i_anchor] / PC
t_ref_Myr = np.array([sol_classic.t[i_anchor], t_end]) / MYR
R_ref_pc = R_anchor * (t_ref_Myr / t_Myr_anchor) ** (4.0 / 7.0)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

t_c = sol_classic.t  / MYR
t_m = sol_modified.t / MYR

# --- Top: R(t) ---
ax = axes[0]
ax.loglog(t_c, sol_classic.y[0]  / PC, color="C0", lw=2.0, label="Classic ODE")
ax.loglog(t_m, sol_modified.y[0] / PC, color="C1", lw=2.0, ls="--", label="Modified ODE")
ax.loglog(t_c, R_sp / PC, color="gray", lw=1.5, ls=":", label="Spitzer analytic")
ax.loglog(t_ref_Myr, R_ref_pc, color="k", lw=1.0, ls="-.", label=r"$\propto t^{4/7}$")
ax.axhline(R_st / PC, color="gray", lw=0.7, ls="--")
ax.text(t_c[1], R_st / PC * 1.08,
        rf"$R_{{\rm st}} = {R_st/PC:.2f}$ pc", color="gray", fontsize=8)
ax.set_ylabel(r"$R\;[\mathrm{pc}]$")
ax.legend(fontsize=8, loc="upper left")
ax.set_title(
    rf"$Q={Q:.0e}\ \mathrm{{s}}^{{-1}}$, "
    rf"$n_0={n0:.0f}\ \mathrm{{cm}}^{{-3}}$, "
    rf"$T={T:.0e}\ \mathrm{{K}}$",
    fontsize=10,
)

# --- Middle: v(t) ---
ax = axes[1]
ax.loglog(t_c, sol_classic.y[1]  / KM_S, color="C0", lw=2.0, label="Classic ODE")
ax.loglog(t_m, sol_modified.y[1] / KM_S, color="C1", lw=2.0, ls="--", label="Modified ODE")
ax.axhline(c_II / KM_S, color="gray", lw=0.7, ls="--")
ax.text(t_c[1], c_II / KM_S * 1.05,
        rf"$c_{{\rm II}} = {c_II/KM_S:.1f}$ km/s", color="gray", fontsize=8)
ax.set_ylabel(r"$v\;[\mathrm{km\,s}^{-1}]$")
ax.legend(fontsize=8, loc="upper right")

# --- Bottom: M_sh(t) ---
ax = axes[2]
ax.loglog(t_c, sol_classic.y[2]  / M_SUN, color="C0", lw=2.0, label="Classic ODE")
ax.loglog(t_m, sol_modified.y[2] / M_SUN, color="C1", lw=2.0, ls="--", label="Modified ODE")
ax.set_ylabel(r"$M_{\rm sh}\;[M_\odot]$")
ax.set_xlabel("Time [Myr]")
ax.legend(fontsize=8, loc="upper left")
# tighten y-limits to the data range (skip seed-mass transient at t=0)
_all_msh = np.concatenate([sol_classic.y[2], sol_modified.y[2]]) / M_SUN
ax.set_ylim(_all_msh[_all_msh > 0].min() * 0.5, _all_msh.max() * 2.0)

for ax in axes:
    ax.grid(True, which="both", alpha=0.3, lw=0.5)

fig.tight_layout()

out = Path(__file__).parent.parent / "figures" / "classic_vs_modified.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
