# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (required before running tests)
pip install -e .

# Run all tests
pytest

# Run a single test file
pytest tests/test_evolution.py

# Run a single test by name
pytest tests/test_evolution.py::TestHIIRegionEvolve::test_power_law_slope_four_sevenths

# Lint
ruff check .

# Auto-fix lint issues
ruff check --fix .
```

## Architecture

The package (`hii_expansion/`) models HII region expansion into an arbitrary density field in CGS units.

### Data flow

1. **`constants.py`** — CGS floats sourced from `astropy.constants` and `astropy.units` (`M_H`, `K_B`, `PC`, `YR`).

2. **`recombination.py`** — `alpha_B_case_B(T)`: Draine (2011) fitting formula for the case B recombination coefficient, valid 3 000–30 000 K.

3. **`hii_region.py`** — the central module:
   - `stromgren_radius_uniform(Q, n, alpha_B)`: standalone analytic formula used as a validation target.
   - `HIIRegion(Q, n, alpha_B, T)`: main class. `n` is either a scalar (constant density) or a callable `n(r)`.
     - `stromgren_radius()` — analytic for constant density; numerical root-finding (`brentq` on the recombination integral) for a profile. Result is cached.
     - `evolve(t_span, n_eval, v0, **ivp_kwargs)` — integrates the thin-shell momentum equation via `scipy.integrate.solve_ivp`. ODE state is `[R, v, M_sh]` to avoid an inner integral at every step: `dM_sh/dt = 4π R² n(R) m_H v`. Interior pressure is `P_in = 2 n_i k_B T` (electron + ion), where `n_i = sqrt(3Q / (4π α_B R³))`.

4. **`analytic.py`** — `spitzer_solution(Q, n, T, t, alpha_B)`: the quasi-static analytic solution `R(t) = R_st [1 + (7/4)(c_II/R_st) t]^(4/7)` with `c_II = sqrt(2 k_B T / m_H)`. Used to validate the ODE integrator for constant density.

### Key conventions

- All quantities in CGS throughout (densities in cm⁻³, lengths in cm, time in s).
- Uppercase physics names (`Q`, `T`, `R`, `alpha_B`) are conventional and intentionally kept; ruff rules N802/N803/N806/N816 are ignored for this reason.
- `OdeResult` (return type of `evolve`) is not exported from the public scipy namespace; it is imported under `TYPE_CHECKING` from the private path `scipy.integrate._ivp.ivp`.
- `python/find_solution.py` is a usage example script, not part of the package.
