"""HII region expansion into a cored density profile n(r) = n0/(1+(r/r0)^w).

Shows both the density profile n(r) vs r and the expansion R(t) vs t for
r0 = (1, 5, 10) pc and w = (1, 1.5, 2).

The late-time asymptotic slope R ∝ t^{4/(7-2w)} (Franco et al. 1990) applies
once R >> r0 and the density profile looks like a pure power law.

Run from the project root:
    python python/plot_cored_profile.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hii_expansion import HIIRegion, alpha_B_case_B
from hii_expansion.constants import PC, YR

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Q = 1.0e49        # ionizing photon rate [s⁻¹]
n0 = 100.0        # central density [cm⁻³]
T = 1.0e4         # temperature [K]
alpha_B = alpha_B_case_B(T)

R0_VALUES = [1.0 * PC, 5.0 * PC, 10.0 * PC]   # core radii [cm]
W_VALUES = [1.0, 1.5, 2.0]          # power-law indices

MYR = 1.0e6 * YR

# Reference timescale: uniform HIIRegion at n0
from hii_expansion import stromgren_radius_uniform  # noqa: E402
R_st_ref = stromgren_radius_uniform(Q, n0, alpha_B)
from hii_expansion.constants import K_B, M_H  # noqa: E402
c_II = np.sqrt(2.0 * K_B * T / M_H)
T_dyn = R_st_ref / c_II

t_end = 200.0 * T_dyn       # coloured curves
t_ref_end = 2.0 * t_end     # black dotted lines extend here

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
colors = plt.cm.plasma(np.linspace(0.1, 0.75, len(W_VALUES)))

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="row", sharey="row")

# r grid for density profile plot (0.01 pc … 200 pc)
r_pc = np.logspace(-2, 2.5, 500)
r_cm = r_pc * PC

# ---------------------------------------------------------------------------
# Loop over columns (r0) and rows (n vs r  /  R vs t)
# ---------------------------------------------------------------------------
for col, r0 in enumerate(R0_VALUES):
    ax_n = axes[0, col]   # density panel
    ax_R = axes[1, col]   # expansion panel

    r0_pc = r0 / PC

    # Mark core radius
    ax_n.axvline(r0_pc, color="gray", lw=0.8, ls="--")
    ax_n.text(r0_pc * 1.1, n0 * 0.5, rf"$r_0={r0_pc:.0f}$ pc",
              color="gray", fontsize=8)

    solutions: list[tuple[np.ndarray, np.ndarray, float]] = []

    for w, color in zip(W_VALUES, colors):
        def n_profile(r: float, _w: float = w, _r0: float = r0) -> float:
            return n0 / (1.0 + (r / _r0) ** _w)

        # ---- density profile ----
        n_vals = np.array([n_profile(r) for r in r_cm])
        slope = 4.0 / (7.0 - 2.0 * w)
        lbl = rf"$w={w}$  (slope $\to {slope:.3f}$)"
        ax_n.loglog(r_pc, n_vals, color=color, lw=2.0, label=lbl)

        # ---- evolution ----
        hii = HIIRegion(Q=Q, n=n_profile, T=T, integration_points=[r0])
        try:
            sol = hii.evolve((0.0, t_end), n_eval=600, rtol=1e-10, atol=0.0)
        except RuntimeError as exc:
            if "density-bounded" in str(exc):
                ax_R.text(
                    0.5, 0.5, rf"$w={w}$: density-bounded",
                    transform=ax_R.transAxes, ha="center", va="center",
                    fontsize=8, color=color,
                )
                continue
            raise
        t_Myr = sol.t / MYR
        R_pc = sol.y[0] / PC
        ax_R.loglog(t_Myr, R_pc, color=color, lw=2.0, label=lbl)

        solutions.append((t_Myr, R_pc, slope))

    # ---- black dotted Franco et al. reference lines ----
    for t_Myr, R_pc, slope in solutions:
        i_anchor = int(0.7 * len(t_Myr))
        t_span = np.array([t_Myr[i_anchor], t_ref_end / MYR])
        R_span = R_pc[i_anchor] * (t_span / t_Myr[i_anchor]) ** slope
        ax_R.loglog(t_span, R_span, color="k", lw=1.3, ls=":", zorder=5)

    ax_R.set_xlim(right=t_ref_end / MYR)

    # ---- labels ----
    ax_n.set_title(rf"$r_0 = {r0_pc:.0f}$ pc")
    ax_n.set_xlabel(r"$r\;[\mathrm{pc}]$")
    ax_R.set_xlabel("Time [Myr]")
    ax_n.legend(fontsize=8)

axes[0, 0].set_ylabel(r"$n\;[\mathrm{cm}^{-3}]$")
axes[1, 0].set_ylabel(r"$R\;[\mathrm{pc}]$")

# ---- shared annotation ----
for ax in axes[1, :]:
    ax.text(
        0.97, 0.03,
        r"Black dotted: $R \propto t^{4/(7-2w)}$ (Franco et al. 1990)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="gray",
    )

fig.suptitle(
    rf"Cored medium $n(r)=n_0\,[1+(r/r_0)^w]^{{-1}}$,"
    rf"  $n_0={n0:.0f}\ \mathrm{{cm}}^{{-3}}$,"
    rf"  $Q={Q:.0e}\ \mathrm{{s}}^{{-1}}$",
    fontsize=11,
)
fig.tight_layout()

out = Path(__file__).parent / "cored_density.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
