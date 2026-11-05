# Supplementary code (revised submission)

Companion material for "Modal Reorganization of Rotating Tapered Beams
under Centrifugal Stiffening: Physical Insights from High-Accuracy
Spectral Computation" by P. Sri Harikrishna and G. Sudheer.

Section, table, figure and equation numbers below refer to the
**revised** manuscript and Supplementary Material (prefix S).

## Contents

- `beam_code/` — the multiprecision Chebyshev-collocation solver, the
  independent nodal differentiation-matrix and Hermite-cubic
  finite-element implementations, and the original benchmark /
  verification scripts (see `beam_code/README.md`, which maps every
  script to its location in the revised paper).
- `scripts/` — analysis scripts. Scripts 1-5 are from the original
  submission and are retained for traceability; scripts 6-16 were
  written for the revision and produce every new result.
- `results/` — output of every script in `scripts/` (text and JSON),
  plus `master_results_table.csv`, a compact summary of the headline
  numbers of the revised manuscript.
- `figures/` — Figures 1-7 and S1-S3 (PDF and PNG), produced by
  `scripts/15_make_figures.py` (Figure 1 is the original drawing).

## Scripts written for the revision

| Script | Purpose (revised manuscript location; reviewer comment) | Output in `results/` |
|---|---|---|
| `analysis_utils.py` | shared helpers: the five configurations; double-precision post-processing of the 34-digit eigenvectors | - |
| `6_node_shift_magnitude.py` | node positions for 0 <= Gamma <= 20; first-order prediction vs. truncation K; finite-difference estimate of the derivative; reversals; numerical uncertainty (Sec. 5.3, Table 10, Figs. 5-6; R2.2, R2.3) | `6_node_shift_magnitude.json`, `_output.txt` |
| `6b_node_summary.py` | post-processing of script 6 into Table 10 and the reversal statistics | `6b_node_summary_output.txt` |
| `6c_node_fine_grid.py` | node positions on a 0.5-step speed grid: last passing / first failing speed and interpolated 10% crossing Gamma_10; stability of the finite-difference derivative estimate for speed pairs (0.05,0.1), (0.1,0.2), (0.25,0.5) (Sec. 5.3, Fig. 6, Table S12) | `6c_node_fine_grid_<config>.json` |
| `6d_node_fine_summary.py` | post-processing of script 6c (Gamma_10 statistics, tau at Gamma_10, derivative-estimate spread) | `6d_node_fine_summary_output.txt` |
| `7_tau_scaling_uncertainty.py` | exact vs. leading-order vs. WKB energy ratio; fits with 95% CI, R^2 and residuals at Gamma = 2-20 over several mode ranges; transition speeds Gamma*_i (Secs. 5.4-5.5, Tables 11-12, S13-S15; R2.4, R2.5, R3.4) | `7_tau_scaling.json`, `_output.txt` |
| `7b_tau_ratio_decomposition.py` | decomposition of tau_exact/tau_leading-order into the change of the tension and bending integrals (Sec. 5.4) | `7b_tau_ratio_decomposition_output.txt` |
| `8_compact_verification.py` | verification diagnostics for moderate and severe taper (Table 3; R2.9) | `8_compact_verification_output.txt` |
| `9_refinement_and_rounding.py` | every benchmark entry at N_p-10, N_p, N_p+10; estimated truncation-error bound (assuming continued geometric contraction) and interval rounding criterion, tested for every entry at the precision actually printed (12 figures for T4/T5, 11 for T6/T8). Table T6x recomputes the beam-2 row Gamma=8 up to N=110; it is printed from N_p=100, because at N_p=90 mode 3 fails the test at 11 figures (Sec. 4.3, Tables 4-7, S3, S9; R2.7) | `9_refinement_raw_*.json`, `9_analysis_*.txt`, `9_log_*.txt` |
| `10_resolved_modes.py` | complex vs. real-unresolved eigenvalues and resolved-mode counts per N (Sec. 4.2, Table 2; R2.10) | `10_resolved_modes_output.txt` |
| `11_cost_matched_accuracy.py` | elapsed time at matched accuracy and cost per converged mode: 34-digit collocation, double-precision collocation, double-precision FEM (Sec. 4.6, Table 9; R2.12) | `11_cost_matched_accuracy_output.txt` |
| `12_ritz_multiprecision.py` | Rayleigh-Ritz values in 34-digit arithmetic (Sec. 4.3, Table S10; R2.8) | `12_ritz_multiprecision_output.txt` |
| `13_check_endpoint_formulas.py` | exact rational check of Eq. (11) / (A.5) against symbolic differentiation and the solver rows (R2.6) | `13_check_endpoint_formulas_output.txt` |
| `14_mode_shape_rates.py` | single-definition amplitude-optimized mode-shape errors and observed orders, n_el = 4-512 (Sec. 4.5, Tables 8, S11; R2.11) | `14_mode_shape_rates_output.txt` |
| `15_make_figures.py` | all figures | `../figures/` |
| `16_version_consistency.py` | checks that mpmath 1.3.0 reproduces results produced under mpmath 1.4.1 (see "Software environment") | `16_version_consistency_output.txt` |

## Key results of the revised manuscript

- **Nodal migration (Sec. 5.3).** The first-order formula predicts the
  direction of all 140 interior nodes of modes 2-8 in the five
  configurations (the 135 nodes of modes 3-8 of the original submission
  plus the mode-2 node of each configuration). The smallest shift at
  Gamma = 6 exceeds its numerical uncertainty by more than 10^3. With
  K = 16 coupled modes the predicted derivative agrees with an
  independent finite-difference estimate to a median relative
  difference of about 1e-3. The 10% validity limit, located on a
  0.5-step speed grid by interpolation, is reported in Sec. 5.3
  (script 6d). Four innermost nodes of the R = 0.5 configuration
  reverse direction at higher speed.
- **Energy ratio (Secs. 5.4-5.5).** Leading-order (Southwell) and exact
  transition speeds agree within 3.4%. Fitted exponents of tau_i vs. i
  depend on mode range and speed (modes 5-12: -2.00 to -2.10); the
  trend is explained by the WKB asymptote
  tau_i ~ Gamma^2 J1 I / [pi^2 (i-1/2)^2] for bending-dominated modes.
  The equal spacing of transition speeds is an extrapolation of that
  asymptote and is supported, not derived, by the computed values.
- **Benchmarks (Sec. 4.3-4.4).** Printed to 12 (beam 1, n = 2) and 11
  (beam 2, hub radius) significant figures under an interval rounding
  criterion; estimated truncation-error bounds <= 1.1e-20 and
  <= 3.3e-13. The criterion revealed 25 incorrectly rounded last digits
  in the original Tables 6 and 8; all are corrected.
- **Rayleigh-Ritz (Sec. 4.3).** In 34-digit arithmetic the Ritz values
  agree with the collocation values to ~1e-22 (beam 1) and ~1e-13
  (beam 2, limited by the truncation of the N = 90 collocation value).

## Running

```bash
pip install mpmath gmpy2 numpy scipy matplotlib sympy
cd scripts
python3 9_refinement_and_rounding.py T4 T5 T9 T6 T8   # ~2.5 h
python3 9_refinement_and_rounding.py T6x              # beam 2, Gamma=8, N=110 (reuses N=90/100 from T6)
python3 6_node_shift_magnitude.py                     # ~30 min
python3 6c_node_fine_grid.py B1 B1R B2 BJ21 BJ62      # ~1 h
python3 6b_node_summary.py; python3 6d_node_fine_summary.py
python3 7_tau_scaling_uncertainty.py                  # ~45 min
python3 7b_tau_ratio_decomposition.py
python3 8_compact_verification.py; python3 10_resolved_modes.py
python3 11_cost_matched_accuracy.py                   # run on an idle core
python3 12_ritz_multiprecision.py; python3 13_check_endpoint_formulas.py
python3 14_mode_shape_rates.py; python3 15_make_figures.py
python3 16_version_consistency.py
```

`scripts/` locates the solver via `../beam_code` (overridable with the
`BEAM_CODE_DIR` environment variable).

## Software environment of the reported results

All results were computed with Python 3.13, NumPy 2.5, SciPy 1.18 and
the gmpy2 backend of mpmath, on a two-core cloud virtual machine
(x86_64). Two mpmath versions were used, because an installation of
sympy (needed by script 13) downgraded mpmath during the session:

| mpmath | datasets |
|---|---|
| 1.4.1 | `6_node_shift_magnitude.json`; `9_refinement_raw_T6.json`, `9_refinement_raw_T8.json`, and the N = 90/100 values copied into `9_refinement_raw_T6x.json` (beam-2 and hub-radius benchmarks, including the N = 80/90/100 solve times quoted in Supplementary Section S10) |
| 1.3.0 | all other revision outputs (scripts 6c, 7, 7b, 8, 9 [T4, T5, T9, and the N = 110 value of T6x], 10, 11, 12, 13, 14) |

`16_version_consistency.py` re-solves representative beam-2 cases and
node positions under mpmath 1.3.0 and compares them with the 1.4.1
values (output in `results/16_version_consistency_output.txt`). The
original submission used mpmath 1.3.0 without gmpy2 (3-4x slower).
