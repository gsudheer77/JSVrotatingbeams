"""Sections 4.12/4.15: FEM vs present-solver mode-shape and
error-per-degree-of-freedom comparison for beam 1, Gamma=0."""
import numpy as np
import mpmath as mp
import chebyshev_rotating_beam as cb
from fem_hermite_solver import solve_fem, fem_mode_shape

mp.mp.dps = 34
mu, P = cb.wang_wereley(-0.5, [-1.5, 0.75, -0.125])

print("FEM convergence (beam 1, Gamma=0):")
mp.mp.dps = 50
ref = [float(x) for x in cb.solve(60, 0, mu, P, nmodes=5)]
mp.mp.dps = 34
print("n_el  ndof  err_mode1     err_mode5")
for n_el in (10, 20, 40, 80, 160):
    f, _, _ = solve_fem(n_el, 0.0, [1.0, -0.5], [1.0, -1.5, 0.75, -0.125], nmodes=5)
    ndof = 2 * (n_el + 1) - 2
    e1 = abs(f[0] - ref[0]) / ref[0]
    e5 = abs(f[4] - ref[4]) / ref[4]
    print(f"{n_el:5d} {ndof:5d}  {e1:.3e}   {e5:.3e}")

print("\nMode-shape L2/slope/curvature error, present vs FEM (n_el=80), mode 1:")
freqs, vecs = cb.solve(50, 0, mu, P, nmodes=1, return_vectors=True)
a = vecs[0]
femf, femv, _ = solve_fem(80, 0.0, [1.0, -0.5], [1.0, -1.5, 0.75, -0.125], nmodes=1)

xi_grid = np.linspace(0.02, 0.98, 200)
import per_case_verification as pcv
w_present = np.array([float(pcv.mode_derivs(a, x)[0]) for x in xi_grid])
w_fem = fem_mode_shape(femv[0], 80, xi_grid)
# normalize both to unit tip deflection
w_present /= float(pcv.mode_derivs(a, 0.999999999)[0])
w_fem /= fem_mode_shape(femv[0], 80, np.array([0.999999]))[0]
l2 = np.sqrt(np.mean((w_present - w_fem) ** 2))
print("L2 deflection error (present vs FEM, normalized to unit tip):", l2)
