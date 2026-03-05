# Test Suite

42 tests across three files, run with `pytest`.

---

## `tests/test_recombination.py` — Case B recombination coefficient (6 tests)

**`TestAlphaBCaseB`**

| Test | Description |
|------|-------------|
| `test_fiducial_value` | `alpha_B_case_B(1e4 K)` returns exactly `2.54e-13 cm³/s` (Draine 2011 formula) |
| `test_agrees_with_osterbrock_at_1e4` | Matches commonly cited Osterbrock (2006) value `~2.6e-13` within 5% |
| `test_agrees_with_storey_hummer_8000k` | Matches Storey & Hummer (1995) tabulated value `~3.03e-13` at 8 000 K within 2% |
| `test_agrees_with_storey_hummer_20000k` | Matches Storey & Hummer (1995) tabulated value `~1.43e-13` at 20 000 K within 2% |
| `test_decreases_with_temperature` | `alpha_B` is strictly decreasing over 5 000–30 000 K |
| `test_positive_over_valid_range` | `alpha_B > 0` for 20 log-spaced temperatures in 3 000–30 000 K |

---

## `tests/test_hii_region.py` — Stromgren radius and HIIRegion (11 tests)

**`TestStromgrenRadiusUniform`** — standalone analytic formula

| Test | Description |
|------|-------------|
| `test_ionization_balance` | Recombination rate inside `R_st` equals `Q` to 1 part in 10¹⁰ |
| `test_scales_as_q_one_third` | `R_st ∝ Q^(1/3)`: multiplying `Q` by 8 doubles `R_st` |
| `test_scales_as_n_minus_two_thirds` | `R_st ∝ n^(-2/3)`: multiplying `n` by 8 divides `R_st` by 4 |
| `test_scales_as_alpha_minus_one_third` | `R_st ∝ α_B^(-1/3)`: multiplying `α_B` by 8 divides `R_st` by 2 |
| `test_positive` | `R_st > 0` for fiducial parameters |

**`TestHIIRegionStromgrenConstantDensity`** — `HIIRegion` with scalar `n`

| Test | Description |
|------|-------------|
| `test_matches_analytic` | Numerical root-finding matches analytic formula to 1 part in 10¹⁰ |
| `test_explicit_alpha_b` | Passing `alpha_B` directly overrides the temperature-derived value |
| `test_cached` | Calling `stromgren_radius()` twice returns the same object (result is cached) |

**`TestHIIRegionStromgrenDensityProfile`** — `HIIRegion` with callable `n(r)`

| Test | Description |
|------|-------------|
| `test_uniform_profile_matches_analytic` | Constant callable `n(r) = n₀` agrees with analytic formula within 1 ppm |
| `test_ionization_balance_for_power_law` | Power-law profile `n(r) = n₀(r₀/r)^w` satisfies `Q = 4π α_B ∫ n² r² dr` within 0.01% |

---

## `tests/test_evolution.py` — ODE evolution and analytic solution (25 tests)

**`TestHIIRegionProperties`** — internal quantities

| Test | Description |
|------|-------------|
| `test_c_II_value` | `c_II = sqrt(2 k_B T / m_H) ≈ 12.8 km/s` at `T = 10⁴ K` |
| `test_n_ionized_at_rst_equals_ambient` | At `R_st`, ionized density `n_i` equals ambient density for constant `n` |
| `test_n_ionized_power_law` | `n_i ∝ R^(-3/2)` from instantaneous ionization balance |
| `test_interior_pressure_factor_two` | `P_in = 2 n_i k_B T`; at `R_st`, equals `2 n₀ k_B T` |
| `test_interior_pressure_power_law` | `P_in ∝ R^(-3/2)`: doubling `R` reduces pressure by `2^(3/2)` |
| `test_swept_mass_uniform_analytic` | For constant `n`, swept mass `M_sh = (4π/3) R³ n m_H` |

**`TestSpitzerSolution`** — analytic quasi-static solution

| Test | Description |
|------|-------------|
| `test_starts_at_stromgren_radius` | `R(t=0) = R_st` exactly |
| `test_monotonically_increasing` | `R(t)` strictly increases over 0.1–100 dynamical times |
| `test_power_law_slope_four_sevenths` | Late-time log-log slope equals `4/7` to 0.1% |
| `test_array_and_scalar_input` | Accepts both scalar `float` and `ndarray` time inputs |

**`TestHIIRegionEvolve`** — classic ODE (`evolve`)

| Test | Description |
|------|-------------|
| `test_starts_at_stromgren_radius` | `R(t=0) = R_st` |
| `test_initial_velocity_default_is_c_II` | Default `v(t=0) = c_II` |
| `test_radius_monotonically_increasing` | `R(t)` strictly increasing |
| `test_mass_monotonically_increasing` | `M_sh(t)` strictly increasing |
| `test_power_law_slope_four_sevenths` | Log-log slope of `R(t)` at late times equals `4/7` within 2% |
| `test_agrees_with_spitzer_at_late_times` | Numerical `R(t)` matches Spitzer analytic within 10% at `t = 50 t_dyn` |
| `test_custom_v0` | User-supplied `v0` overrides default initial velocity |
| `test_failed_integration_raises` | Mocked `solve_ivp` failure raises `RuntimeError("ODE integration failed")` |

**`TestModifiedSpitzer`** — modified ODE (`evolve_modified`)

| Test | Description |
|------|-------------|
| `test_modified_starts_at_rst` | `R(t=0) = R_st` |
| `test_modified_initial_velocity_c_II` | `v(t=0) = c_II` |
| `test_modified_initial_mass_negligible` | Initial shell mass seed is < 10⁻⁵ of classic initial mass |
| `test_modified_R_monotone` | Shell radius increases monotonically |
| `test_modified_M_sh_positive` | Shell mass non-negative throughout |
| `test_modified_mass_less_than_classic` | Modified `M_sh(t) ≤` classic `M_sh(t)` at all times (ionized gas excluded from shell) |
| `test_modified_asymptotic_slope` | Late-time `R ∝ t^(4/7)` within 1% for uniform density |
| `test_modified_failed_integration_raises` | Mocked failure raises `RuntimeError("ODE integration failed")` |
