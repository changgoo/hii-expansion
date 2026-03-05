"""Ram pressure comparison figure.

4-panel log-log figure for uniform density illustrating the role of the
ram pressure term and the D-critical initial condition:

  Panel 1 (R):  Full ODE (v0=cII), no-ram ODE (v0=cII), no-ram ODE (v0=0),
                Spitzer analytic
  Panel 2 (v):  same four curves
  Panel 3:      ratio to Spitzer for the three ODE curves

Key result: the no-ram ODE with v0=cII diverges because the D-critical
initial condition creates a large imbalance (P_in - 0 = P_in, giving an
immediate kick).  With v0=0 the force is zero at t=0 and the no-ram ODE
converges to the Spitzer solution at late times.

Run from the project root:
    python python/plot_ram_pressure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

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

# Full ODE with D-critical IC (standard model)
sol_ram_dcrit  = hii.evolve((0.0, t_end), ram_pressure=True,  v0=c_II,  **ivp_kw)
# No-ram ODE with D-critical IC (inappropriate: large initial imbalance)
sol_noram_dcrit = hii.evolve((0.0, t_end), ram_pressure=False, v0=c_II,  **ivp_kw)
# No-ram ODE with static IC (v0=0: atol must be nonzero to avoid divide-by-zero
# in scipy's step-size estimator)
sol_noram_static = hii.evolve(
    (0.0, t_end), ram_pressure=False, v0=0.0,
    rtol=1e-10, atol=1e-3, n_eval=1000,
)

# Analytic Spitzer solution
t_arr = sol_ram_dcrit.t
R_sp = spitzer_solution(Q, n0, T, t_arr, alpha_B=alpha_B)
v_sp = c_II * (R_st / R_sp) ** 0.75   # quasi-static Spitzer velocity

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
t_Myr = t_arr / MYR

# --- Top: R(t) ---
ax = axes[0]
ax.loglog(t_Myr, sol_ram_dcrit.y[0]   / PC, color="C0", lw=2.0,
          label=r"ODE (ram, $v_0=c_\mathrm{II}$)")
ax.loglog(t_Myr, sol_noram_dcrit.y[0] / PC, color="C1", lw=2.0, ls="--",
          label=r"ODE (no ram, $v_0=c_\mathrm{II}$)")
ax.loglog(t_Myr[1:], sol_noram_static.y[0, 1:] / PC, color="C2", lw=2.0, ls="-.",
          label=r"ODE (no ram, $v_0=0$)")
ax.loglog(t_Myr, R_sp / PC, color="gray", lw=1.5, ls=":",
          label="Spitzer analytic")
ax.axhline(R_st / PC, color="gray", lw=0.7, ls="--")
ax.text(t_Myr[1], R_st / PC * 1.08,
        rf"$R_{{\rm st}} = {R_st/PC:.2f}$ pc", color="gray", fontsize=8)
ax.set_ylabel(r"$R\;[\mathrm{pc}]$")
ax.legend(fontsize=7, loc="upper left")
ax.set_title(
    rf"$Q={Q:.0e}\ \mathrm{{s}}^{{-1}}$, "
    rf"$n_0={n0:.0f}\ \mathrm{{cm}}^{{-3}}$, "
    rf"$T={T:.0e}\ \mathrm{{K}}$",
    fontsize=10,
)

# --- Middle: v(t) ---
ax = axes[1]
ax.loglog(t_Myr, sol_ram_dcrit.y[1]   / KM_S, color="C0", lw=2.0,
          label=r"ODE (ram, $v_0=c_\mathrm{II}$)")
ax.loglog(t_Myr, sol_noram_dcrit.y[1] / KM_S, color="C1", lw=2.0, ls="--",
          label=r"ODE (no ram, $v_0=c_\mathrm{II}$)")
ax.loglog(t_Myr[1:], sol_noram_static.y[1, 1:] / KM_S, color="C2", lw=2.0, ls="-.",
          label=r"ODE (no ram, $v_0=0$)")
ax.loglog(t_Myr[1:], v_sp[1:] / KM_S, color="gray", lw=1.5, ls=":",
          label="Spitzer analytic")
ax.axhline(c_II / KM_S, color="gray", lw=0.7, ls="--")
ax.text(t_Myr[1], c_II / KM_S * 1.05,
        rf"$c_{{\rm II}} = {c_II/KM_S:.1f}$ km/s", color="gray", fontsize=8)
ax.set_ylabel(r"$v\;[\mathrm{km\,s}^{-1}]$")
ax.legend(fontsize=7, loc="upper right")

# --- Bottom: ratio to Spitzer ---
ax = axes[2]
ax.semilogx(t_Myr, sol_ram_dcrit.y[0]    / R_sp, color="C0", lw=2.0,
            label=r"ODE (ram, $v_0=c_\mathrm{II}$) / Spitzer")
ax.semilogx(t_Myr, sol_noram_dcrit.y[0]  / R_sp, color="C1", lw=2.0, ls="--",
            label=r"ODE (no ram, $v_0=c_\mathrm{II}$) / Spitzer")
ax.semilogx(t_Myr[1:], sol_noram_static.y[0, 1:] / R_sp[1:], color="C2", lw=2.0,
            ls="-.", label=r"ODE (no ram, $v_0=0$) / Spitzer")
ax.axhline(1.0, color="k", lw=0.7, ls="-")
ax.set_ylabel("Ratio to Spitzer")
ax.set_xlabel("Time [Myr]")
ax.legend(fontsize=7, loc="center right")

for ax in axes:
    ax.grid(True, which="both", alpha=0.3, lw=0.5)

fig.tight_layout()

fig_dir = Path(__file__).parent.parent / "figures"
for ext in ("pdf", "png"):
    out = fig_dir / f"ram_pressure.{ext}"
    fig.savefig(out, dpi=150 if ext == "png" else None)
    print(f"Saved {out}")
