"""HII region expansion into power-law density profiles n(r) = n0 (r0/r)^w.

Shows R(t) for several values of the power-law index w on a single log–log
plot.  The expected late-time slopes R ∝ t^{4/(7-2w)} (Franco et al. 1990)
are drawn as black dotted reference lines that extend beyond the coloured
data curves, making them clearly visible.

Valid range: w < 3/2.  For w ≥ 3/2 the recombination integral diverges at
r = 0 and no finite Stromgren radius exists.

Run from the project root:
    python python/plot_powerlaw.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hii_expansion import HIIRegion, alpha_B_case_B
from hii_expansion.constants import PC, YR

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Q = 1.0e49    # ionizing photon rate [s⁻¹]
n0 = 100.0    # density at reference radius r0 [cm⁻³]
T = 1.0e4     # HII region temperature [K]
r0 = PC       # reference radius [cm]  (1 pc)
alpha_B = alpha_B_case_B(T)

# Power-law indices to compare; w=0 is the uniform (Spitzer) case.
# Keep w < 1.5 to ensure the recombination integral converges.
W_VALUES = [0.0, 0.5, 1.0, 1.4]

# ---------------------------------------------------------------------------
# Reference timescale from the uniform case
# ---------------------------------------------------------------------------
hii_ref = HIIRegion(Q=Q, n=n0, T=T)
R_st_ref = hii_ref.stromgren_radius()
T_dyn = R_st_ref / hii_ref.c_II

t_end = 50.0 * T_dyn       # coloured curves go up to here
t_ref_end = 2.0 * t_end    # black dotted reference lines extend to here
MYR = 1.0e6 * YR

# ---------------------------------------------------------------------------
# Integrate each profile; store results for a second pass of reference lines
# ---------------------------------------------------------------------------
colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(W_VALUES)))

fig, ax = plt.subplots(figsize=(6, 4.5))

solutions: list[tuple[np.ndarray, np.ndarray, float]] = []

for w, color in zip(W_VALUES, colors):
    if w == 0.0:
        n_profile: float | object = n0
    else:
        def n_profile(r: float, _w: float = w) -> float:  # noqa: E731
            return n0 * (r0 / r) ** _w

    hii = HIIRegion(Q=Q, n=n_profile, T=T)
    sol = hii.evolve((0.0, t_end), n_eval=600, rtol=1e-10, atol=0.0)
    try:
        sol_mod = hii.evolve_modified((0.0, t_end), n_eval=600, rtol=1e-8)
        ax.loglog(sol_mod.t / MYR, sol_mod.y[0] / PC, color=color, lw=1.5, ls="--")
    except RuntimeError:
        pass  # modified ODE ill-conditioned for steep profiles (n_i > 2n at R_st)

    t_Myr = sol.t / MYR
    R_pc = sol.y[0] / PC

    slope = 4.0 / (7.0 - 2.0 * w)   # Franco et al. (1990)
    lbl = rf"$w={w}$  (slope $\to {slope:.3f}$)"
    ax.loglog(t_Myr, R_pc, color=color, lw=2.0, alpha=0.5, label=lbl)

    solutions.append((t_Myr, R_pc, slope))

# ---------------------------------------------------------------------------
# Black dotted reference lines – anchored to the last data point of each
# curve and extended to t_ref_end.  Because they continue past the coloured
# curves, they are clearly visible even where the two coincide.
# ---------------------------------------------------------------------------
for t_Myr, R_pc, slope in solutions:
    # Anchor at 70 % of the data to show convergence over the final stretch
    i_anchor = int(0.7 * len(t_Myr))
    t_span = np.array([t_Myr[i_anchor], t_ref_end / MYR])
    R_span = R_pc[i_anchor] * (t_span / t_Myr[i_anchor]) ** slope
    ax.loglog(t_span, R_span, color="k", lw=1.3, ls=":", zorder=5)

# ---------------------------------------------------------------------------
# Axis limits and annotations
# ---------------------------------------------------------------------------
ax.set_xlim(right=t_ref_end / MYR)

ax.set_xlabel("Time [Myr]")
ax.set_ylabel(r"$R\;[\mathrm{pc}]$")
ax.set_title(
    rf"Power-law medium $n(r) = n_0\,(r_0/r)^w$,  "
    rf"$n_0 = {n0:.0f}\ \mathrm{{cm}}^{{-3}}$,  $r_0 = 1$ pc"
)
import matplotlib.lines as mlines
_solid = mlines.Line2D([], [], color="gray", lw=2.0, label="Classic ODE")
_dash  = mlines.Line2D([], [], color="gray", lw=1.5, ls="--", label="Modified ODE")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [_solid, _dash], labels + ["Classic ODE", "Modified ODE"],
          fontsize=9, title=rf"$Q = {Q:.0e}\ \mathrm{{s}}^{{-1}}$", title_fontsize=8)
ax.text(
    0.97,
    0.03,
    r"Black dotted: $R \propto t^{4/(7-2w)}$ (Franco et al. 1990)",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.5,
    color="gray",
)

fig.tight_layout()

out = Path(__file__).parent.parent / "figures" / "powerlaw_density.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
