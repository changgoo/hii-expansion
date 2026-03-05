"""Ram pressure comparison figure.

3-panel log-log figure for uniform density comparing:
  - Spitzer quasi-static analytic solution (no ram pressure)
  - Classic ODE without ram pressure
  - Classic ODE with ram pressure (reference)

Run from the project root:
    python python/plot_ram_pressure.py
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
ivp_kw = dict(rtol=1e-10, atol=0.0, n_eval=1000)

sol_with_ram = hii.evolve((0.0, t_end), ram_pressure=True,  **ivp_kw)
sol_no_ram   = hii.evolve((0.0, t_end), ram_pressure=False, **ivp_kw)

# Analytic Spitzer solution (neglects ram pressure)
R_sp = spitzer_solution(Q, n0, T, sol_with_ram.t, alpha_B=alpha_B)
v_sp = np.gradient(R_sp, sol_with_ram.t)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)

t_Myr = sol_with_ram.t / MYR

# --- Top: R(t) ---
ax = axes[0]
ax.loglog(t_Myr, sol_with_ram.y[0] / PC, color="C0", lw=2.0, label="ODE (with ram)")
ax.loglog(t_Myr, sol_no_ram.y[0]   / PC, color="C1", lw=2.0, ls="--",
          label="ODE (no ram)")
ax.loglog(t_Myr, R_sp               / PC, color="gray", lw=1.5, ls=":",
          label="Spitzer analytic")
ax.axhline(R_st / PC, color="gray", lw=0.7, ls="--")
ax.text(t_Myr[1], R_st / PC * 1.08,
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
ax.loglog(t_Myr, sol_with_ram.y[1] / KM_S, color="C0", lw=2.0, label="ODE (with ram)")
ax.loglog(t_Myr, sol_no_ram.y[1]   / KM_S, color="C1", lw=2.0, ls="--",
          label="ODE (no ram)")
ax.loglog(t_Myr[1:], np.abs(v_sp[1:]) / KM_S, color="gray", lw=1.5, ls=":",
          label="Spitzer analytic")
ax.axhline(c_II / KM_S, color="gray", lw=0.7, ls="--")
ax.text(t_Myr[1], c_II / KM_S * 1.05,
        rf"$c_{{\rm II}} = {c_II/KM_S:.1f}$ km/s", color="gray", fontsize=8)
ax.set_ylabel(r"$v\;[\mathrm{km\,s}^{-1}]$")
ax.legend(fontsize=8, loc="upper right")

# --- Bottom: ratio R_no_ram / R_with_ram and R_spitzer / R_with_ram ---
ax = axes[2]
ratio_no_ram  = sol_no_ram.y[0] / sol_with_ram.y[0]
ratio_spitzer = R_sp            / sol_with_ram.y[0]
ax.semilogx(t_Myr, ratio_no_ram,  color="C1", lw=2.0, ls="--",
            label=r"$R_{\rm no\,ram} / R_{\rm with\,ram}$")
ax.semilogx(t_Myr, ratio_spitzer, color="gray", lw=1.5, ls=":",
            label=r"$R_{\rm Spitzer} / R_{\rm with\,ram}$")
ax.axhline(1.0, color="k", lw=0.7, ls="-")
ax.set_ylabel("Ratio")
ax.set_xlabel("Time [Myr]")
ax.legend(fontsize=8, loc="lower right")

for ax in axes:
    ax.grid(True, which="both", alpha=0.3, lw=0.5)

fig.tight_layout()

fig_dir = Path(__file__).parent.parent / "figures"
for ext in ("pdf", "png"):
    out = fig_dir / f"ram_pressure.{ext}"
    fig.savefig(out, dpi=150 if ext == "png" else None)
    print(f"Saved {out}")
