# Supplementary code — rotating tapered Euler–Bernoulli beam, multiprecision Chebyshev collocation

Companion material for "Modal Reorganization of Rotating Tapered Beams
under Centrifugal Stiffening: Physical Insights from High-Accuracy
Spectral Computation" by P.Sri Harikrishna and G.Sudheer.

This `beam_code/` folder provides the core solver and the original
benchmark/verification scripts. All section, table and figure numbers
below refer to the REVISED manuscript (prefix S = Supplementary
Material). Several scripts here produced results of the original
submission that are now superseded by scripts in `../scripts/`; this is
marked in the table. The revision's analyses are in `../scripts/`,
with output in `../results/`.

## Contents

| File | Description |
|---|---|
| `chebyshev_rotating_beam.py` | Multiprecision trigonometric Chebyshev collocation solver (Sections 3.1-3.3). `build_pencil` assembles the generalized eigenvalue pencil for arbitrary polynomial mass/rigidity laws `mu(xi)`, `P(xi)`; `solve` reduces it to standard form and diagonalizes it. The boundary reduction defaults to the well-scaled coefficient set `{a_0,a_1,a_{N-1},a_N}` (`elim="well-scaled"`, Section 3.3 / Appendix B); the alternative `elim="highest-degree"` reproduces the reduction-choice comparison of Appendix B and is not used for any reported benchmark value. `wang_wereley` and `banerjee` construct the two taper-law families of Section 2.3. |
| `dmatrix_solver.py` | Independent nodal Chebyshev differentiation-matrix formulation of the same discretized problem (Supplementary Section S4, Table S4), with matching interior collocation points and boundary reduction. |
| `fem_hermite_solver.py` | Independent cubic-Hermite (Euler–Bernoulli) finite element solver: 2-node, 4-DOF elements, 6-point Gauss–Legendre quadrature per element, consistent mass, bending-stiffness, and centrifugal-stiffness matrices, double precision (NumPy/SciPy) — the "unscaled double-precision Hermite implementation" of Sections 4.5-4.6 and S5. |
| `fem_comparison.py` | FEM convergence table and a present-vs-FEM mode-shape L2 comparison for beam 1 (Supplementary Section S5). Superseded for the mode-shape study by `../scripts/14_mode_shape_rates.py`. Output: `fem_comparison_output.txt`. |
| `slope_curvature_error.py` | The ε_W′, ε_W″ slope- and curvature-error tables of the original Section 4.12 (n_el=16/64/256 only). Superseded by `../scripts/14_mode_shape_rates.py` (revised Section 4.5, Table 8), which uses one scale factor for all derivative orders and n_el=4-512. Output: `slope_curvature_error_output.txt`. |
| `per_case_verification.py` | Verification diagnostics (revised Section 4.3, Table 3, via `../scripts/8_compact_verification.py`): N-refinement relative change, algebraic backward error, dense-grid continuous residual, boundary-condition residual, Rayleigh-quotient discrepancy, cross-resolution MAC, mass-orthogonality error. Output: `per_case_verification_output.txt`. |
| `extended_beam2_convergence.py` | Original beam-2 table at N=80 (revised Table 6 is printed at N_p=90 from `../scripts/9_refinement_and_rounding.py`; this script's N=50-90 sequence underlies Supplementary Section S1) for Γ=0..12, and the N=50→60→70→80→90 relative-change sequence for modes 1/3/5 at Γ=0,6,12. Output: `extended_beam2_output.txt`. |
| `taper_sweep.py` | Supplementary Table S7: the c=0.8 convergence sequence, the c=0.1 comparison against Banerjee–Su–Jackson, and a required-N-vs-taper-ratio sweep for both Banerjee classes (c∈{0.1,0.2,0.4,0.6,0.8}). Output: `taper_sweep_output.txt`. |
| `hub_radius_sensitivity.py` | Hub-radius cases (revised Table 7 is printed at N_p=90 from `../scripts/9_refinement_and_rounding.py`) and Supplementary Table S8: and the logarithmic sensitivity S_R̄^(i) = (R̄/λᵢ)(∂λᵢ/∂R̄) by numerical differentiation of the solver. Output: `hub_radius_output.txt`. |
| `condition_numbers.py` | Supplementary Table S2: the boundary-reduction condition number κ₂(B_I) for the well-scaled vs. highest-degree elimination sets, N=20–80. Output: `condition_numbers_output.txt`. |
| `reproduce_tables.py` | Regenerates the original Tables 4, 5, 6 and 9 (N=50/80) directly from the solver; the revised Tables 4-7 are printed from `../scripts/9_refinement_and_rounding.py` at N_p. Output: `reproduce_tables_output.txt`. |
| `make_figures_corrected.py` | Regenerates the original Figures 2-3 (revised Fig. 2 and Fig. S1, now produced by `../scripts/15_make_figures.py`) from `figdata_conv2.json` / `figdata_floor3.json`. |
| `fig2_convergence.pdf/.png`, `fig3_precision_floor.pdf/.png` | Original Figures 2-3 (superseded by `../figures/`): exponential convergence to the finite-precision floor, and the 34- vs 50-digit floor comparison. |
| `fig8_dof_comparison.py` | Regenerates the original Figure 8 (revised Fig. S2): relative frequency error versus degrees of freedom (mode 1 and mode 5, beam 1, Γ=0), comparing the present method against the Hermite-cubic FEM (Supplementary Section S5), using the same solvers as `fem_comparison.py`. (Renumbered from Figure 7 to Figure 8 in the current manuscript, since the new nodal-migration/tau_n figure -- generated by `../scripts/` -- now occupies the Figure 7 slot.) Output: `fig8_dof_comparison.pdf/.png`, `figdata_fig8.json`. |
| `fig5_modeshapes.pdf`, `fig4_frequency_speed.pdf`, `fig6_node_migration.pdf` | Original Figures 4-6 (superseded by revised Figs. 3-5 in `../figures/`). |
| `make_figures_4_6.py` | Regenerates the original Figures 4-6 directly from the solver (superseded by `../scripts/15_make_figures.py`) (frequency vs. rotational speed, mode shapes, and interior-node migration via root-finding on the reconstructed shape) — no cached figure data required. Output: `fig4_frequency_speed.pdf/.png`, `fig5_modeshapes.pdf/.png`, `fig6_node_migration.pdf/.png`. |
| `table12_generate.py` | Generator for the ORIGINAL Table 12 (superseded by `../scripts/14_mode_shape_rates.py`, revised Table 8): MAC and the amplitude-optimized displacement error ε_W, beam 1, Γ=0, present N=30, FEM n_el=16/64/256, modes 1–5, computed from one shared pair of sampled vectors per case (so MAC and ε_W are always mutually consistent). Output: `table12_generate_output.txt`. |
| `timing.py` | Original Section 4.13 timing (superseded by `../scripts/11_cost_matched_accuracy.py`, revised Table 9): full timing (assembly / reduce+eigensolve / total) and peak-memory tables for beam 1 (N=15–50) and beam 2 (N=80), the 34-digit-vs-double-precision cost ratio, and fitted scaling exponents. Output: `timing_output.txt`. |

## Requirements

Python ≥ 3.9, `mpmath`, `numpy`, `scipy`; `matplotlib` for regenerating figures only.

```
pip install mpmath numpy scipy matplotlib
```

## Execution sequence

Each script is self-contained and can be run independently:

```bash
python3 reproduce_tables.py            # Tables 4, 5, 6, 9
python3 per_case_verification.py       # Sec. 4.16 diagnostics (beam 1, Gamma=0)
python3 extended_beam2_convergence.py  # Tables 6, 7
python3 taper_sweep.py                 # Table 9, taper-sweep data
python3 hub_radius_sensitivity.py      # Appendix B / Table 8, sensitivity
python3 condition_numbers.py           # Appendix B
python3 fem_comparison.py              # FEM convergence, Sec. 4.15
python3 slope_curvature_error.py       # Table (eps_W', eps_W'')
python3 make_figures_corrected.py      # Figures 2-3 (needs figdata_*.json)
python3 fig8_dof_comparison.py         # Figure 8 (Sec. 4.15)
python3 make_figures_4_6.py            # Figures 4-6, direct from the solver
python3 table12_generate.py            # Table 12 (MAC + eps_W), definitive source
python3 timing.py                      # Sec. 4.13 timing/memory tables (N=80 beam-2 row ~5 min)
```

Execution time depends strongly on the processor and the multiprecision
backend; the N=80–90 rows used for the severe-taper beam (beam 2) and
the extended convergence sequences dominate the total run time and can
take several minutes each on a single core.

## Scope

The scripts above provide a representative, fully worked example for
each diagnostic described in the manuscript's Data Availability section
(e.g. `per_case_verification.py` produces the verification
diagnostics for beam 1 at Γ=0;
the same script reproduces the identical diagnostics for every other
benchmark case without modification). Running every script for the full
cross-product of beam × speed × hub-radius × taper combinations
tabulated in the manuscript is mechanical but was not pre-computed here.

An independent MATLAB/Advanpix implementation of the trigonometric
collocation operator was used for the cross-implementation check
described in Appendix B; that implementation is not included here. The
Python/`mpmath` implementation in this package is the one used to
produce every reported calculation.

`build_pencil`/`solve` in `chebyshev_rotating_beam.py` implement only
the `etaF=0` (`F_t=0`, no outboard axial tip force) problem: the
fourth boundary row imposes the `etaF=0` shear condition, and
passing `etaF != 0` raises `NotImplementedError`, since the general
non-zero end-force shear boundary condition is outside the scope of
the numerical results reported in the manuscript. Every tabulated
result in the manuscript uses `etaF=0`. (`dmatrix_solver.py`'s nodal
formulation does implement the general non-zero-`etaF` shear row, for
reference, but it is likewise never exercised at `etaF != 0` by any
script here.)

**Table 12 reproduction.** `table12_generate.py` is the single script
that produces both the MAC and ε_W rows of the original Table 12 (superseded; see revised Section 4.5):
present solution vs. an independent Hermite-cubic FEM, beam 1, Γ=0,
N=30, FEM n_el=16/64/256, modes 1–5, computed from one shared pair of
sampled shape vectors per case so that MAC and ε_W are always mutually
consistent.

## Precision and reproducibility notes

- All benchmark-frequency computations use `mp.mp.dps = 34` (34
  significant decimal digits), matching the manuscript throughout.
  `reproduce_tables.py` prints values to 30 significant digits; as in
  the manuscript itself, the number of digits that are *converged* for
  a given case is established by N-refinement (revised Section 4.3), not by how
  many digits are printed.
- The Rayleigh-quotient discrepancy and mass-orthogonality integrals in
  `per_case_verification.py` are evaluated with a fixed-order
  Gauss-Legendre quadrature rule (96 points, exact for polynomials up to
  degree 191) rather than adaptive refinement. Because the integrands
  are polynomial in the physical coordinate (a degree-N Chebyshev mode
  shape against polynomial mass/rigidity/tension weights), the rule is
  chosen with a comfortable margin above the estimated integrand degree
  for N up to 50, so quadrature error is negligible relative to the
  reported continuum-diagnostic floor for the benchmark cases
  considered, and reproduces the
  manuscript's reported ~1e-21–1e-23 continuum-diagnostic level.
- `chebyshev_rotating_beam.py`, `dmatrix_solver.py`, and
  `fem_hermite_solver.py` are the three independent discretizations
  compared in Sections 4.5-4.6 and S4-S5; each can be run standalone via its
  `solve` / `solve_dmatrix_mp` / `solve_fem` entry point.

## Quick start

```python
import mpmath as mp
from chebyshev_rotating_beam import solve, wang_wereley, banerjee

mp.mp.dps = 34
mu, P = wang_wereley(-0.5, [-1.5, 0.75, -0.125])   # beam 1
print(solve(50, 0, mu, P, nmodes=3))               # well-scaled reduction (default);
                                                    # N=50 matches the benchmark
                                                    # resolution used in Tables 4/5 -
                                                    # confirm against the manuscript

# Appendix B reduction-choice comparison only -- not used for any reported value:
print(solve(40, 0, mu, P, nmodes=1, elim="highest-degree"))
```
