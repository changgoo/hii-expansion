"""HII region expansion into a cored density profile n(r) = n0/(1+(r/r0)^w).

Layout: 3-row figure (one row per r0 value).  Each panel shows R(t) for
w = (1, 1.5, 2) with an inset showing the corresponding density profile n(r).

Run from the project root:
    python python/plot_cored_profile.py
"""

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from hii_expansion import HIIRegion, alpha_B_case_B
from hii_expansion.constants import K_B, M_H, PC, YR
from hii_expansion import stromgren_radius_uniform

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Q = 1.0e49        # ionizing photon rate [s⁻¹]
n0 = 100.0        # central density [cm⁻³]
T = 1.0e4         # temperature [K]
alpha_B = alpha_B_case_B(T)

R0_VALUES = [1.0 * PC, 5.0 * PC]   # core radii [cm]
W_VALUES = [1.0, 1.5, 2.0]

MYR = 1.0e6 * YR

# Reference timescale from uniform case
R_st_ref = stromgren_radius_uniform(Q, n0, alpha_B)
c_II = np.sqrt(2.0 * K_B * T / M_H)
T_dyn = R_st_ref / c_II

t_end = 200.0 * T_dyn
t_ref_end = 2.0 * t_end

# r grid for inset density profiles (0.01 pc … 200 pc)
r_pc = np.logspace(-2, 2.5, 500)
r_cm = r_pc * PC

colors = plt.cm.plasma(np.linspace(0.1, 0.75, len(W_VALUES)))

# ---------------------------------------------------------------------------
# Layout: 3 rows × 1 column
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

for row, r0 in enumerate(R0_VALUES):
    ax_R = axes[row]
    r0_pc = r0 / PC

    solutions: list[tuple[np.ndarray, np.ndarray, float]] = []

    # ---- inset for density profile ----
    ax_in = ax_R.inset_axes([0.05, 0.55, 0.42, 0.42])
    ax_in.axvline(r0_pc, color="gray", lw=0.8, ls="--")
    ax_in.text(r0_pc * 1.12, n0 * 0.35, rf"$r_0={r0_pc:.0f}$ pc",
               color="gray", fontsize=7)

    for w, color in zip(W_VALUES, colors):
        def n_profile(r: float, _w: float = w, _r0: float = r0) -> float:
            return n0 / (1.0 + (r / _r0) ** _w)

        slope = 4.0 / (7.0 - 2.0 * w)
        lbl = rf"$w={w}$  (slope $\to {slope:.3f}$)"

        # ---- inset: density profile ----
        n_vals = np.array([n_profile(r) for r in r_cm])
        ax_in.loglog(r_pc, n_vals, color=color, lw=1.5)

        # ---- main panel: evolution ----
        hii = HIIRegion(Q=Q, n=n_profile, T=T, integration_points=[r0])
        try:
            sol = hii.evolve((0.0, t_end), n_eval=600, rtol=1e-10, atol=0.0)
        except RuntimeError as exc:
            if "density-bounded" in str(exc):
                continue
            raise
        t_Myr = sol.t / MYR
        R_pc_arr = sol.y[0] / PC
        ax_R.loglog(t_Myr, R_pc_arr, color=color, lw=2.0, alpha=0.5, label=lbl)
        try:
            sol_mod = hii.evolve_modified((0.0, t_end), n_eval=600, rtol=1e-8)
            ax_R.loglog(sol_mod.t / MYR, sol_mod.y[0] / PC,
                        color=color, lw=1.5, ls="--")
        except RuntimeError:
            pass

        solutions.append((t_Myr, R_pc_arr, slope))

    # ---- Franco et al. reference lines ----
    for t_Myr, R_pc_arr, slope in solutions:
        i_anchor = int(0.7 * len(t_Myr))
        t_span = np.array([t_Myr[i_anchor], t_ref_end / MYR])
        R_span = R_pc_arr[i_anchor] * (t_span / t_Myr[i_anchor]) ** slope
        ax_R.loglog(t_span, R_span, color="k", lw=1.3, ls=":", zorder=5)

    ax_R.set_xlim(right=t_ref_end / MYR)
    ax_R.set_ylabel(r"$R\;[\mathrm{pc}]$")
    ax_R.set_title(rf"$r_0 = {r0_pc:.0f}$ pc", fontsize=10)
    ax_R.text(
        0.97, 0.03,
        r"Dotted: $R \propto t^{4/(7-2w)}$ (Franco et al. 1990)",
        transform=ax_R.transAxes, ha="right", va="bottom",
        fontsize=7, color="gray",
    )

    # ---- inset cosmetics ----
    ax_in.set_xlabel(r"$r$ [pc]", fontsize=7)
    ax_in.set_ylabel(r"$n$ [cm$^{-3}$]", fontsize=7)
    ax_in.tick_params(labelsize=6)

    # ---- legend on first panel only ----
    if row == 0:
        _solid = mlines.Line2D([], [], color="gray", lw=2.0, alpha=0.5, label="Classic ODE")
        _dash  = mlines.Line2D([], [], color="gray", lw=1.5, ls="--", label="Modified ODE")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles + [_solid, _dash], labels + ["Classic ODE", "Modified ODE"],
                       fontsize=8, loc="upper left")

axes[-1].set_xlabel("Time [Myr]")

fig.suptitle(
    rf"Cored medium $n(r)=n_0\,[1+(r/r_0)^w]^{{-1}}$,"
    rf"  $n_0={n0:.0f}\ \mathrm{{cm}}^{{-3}}$,"
    rf"  $Q={Q:.0e}\ \mathrm{{s}}^{{-1}}$",
    fontsize=11,
)
fig.tight_layout()

out = Path(__file__).parent.parent / "figures" / "cored_density.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
