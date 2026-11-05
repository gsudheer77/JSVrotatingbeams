"""
slope_curvature_error.py
==========================================================================
Section 4.12: the amplitude-optimized slope- and curvature-error tables

    eps_W  = min_a ||W_spec - a*W_FEM||_2 / ||W_spec||_2
    eps_W' = min_a ||W_spec' - a*W_FEM'||_2 / ||W_spec'||_2
    eps_W''= min_a ||W_spec''- a*W_FEM''||_2 / ||W_spec''||_2

evaluated on the same 801-point common grid on the full interval [0,1]
described in Section 4.12, for beam 1 (Gamma=0), modes 1-5, at the FEM
mesh sequence n_el=16,64,256 (matching Table 11's dof sequence
32,128,512), plus fitted convergence rates (least-squares slope of
log(eps) vs log(h)) for each of the three quantities, as referenced by
Section 4.12 ("the full eps_W', eps_W'' tables and their fitted rates").

The present-solution reference is computed at N=30, matching the
manuscript's Table 12 calculation protocol.
==========================================================================
"""
import numpy as np
import mpmath as mp
import chebyshev_rotating_beam as cb
import per_case_verification as pcv
from fem_hermite_solver import solve_fem, fem_mode_shape

mp.mp.dps = 34
mu, P = cb.wang_wereley(-0.5, [-1.5, 0.75, -0.125])  # beam 1


def eps(spec, fem):
    """Amplitude-optimized relative L2 error on a discrete grid."""
    spec = np.asarray(spec, dtype=float)
    fem = np.asarray(fem, dtype=float)
    alpha = np.dot(spec, fem) / np.dot(fem, fem)
    return np.linalg.norm(spec - alpha * fem) / np.linalg.norm(spec)


def main():
    n_el_list = [16, 64, 256]
    nmodes = 5
    xi_grid = np.linspace(0.0, 1.0, 801)  # 801-point common grid on the full
    # interval [0,1] (Section 4.12); the two endpoints are handled by
    # pcv.mode_derivs' exact closed-form endpoint branch, not the interior
    # trig formula (which is singular exactly at xi=0,1).

    freqs, vecs = cb.solve(30, 0, mu, P, nmodes=nmodes, return_vectors=True)

    results = {0: {}, 1: {}, 2: {}}  # order -> n_el -> [eps per mode]
    for n_el in n_el_list:
        femf, femv, _ = solve_fem(n_el, 0.0, [1.0, -0.5], [1.0, -1.5, 0.75, -0.125],
                                   nmodes=nmodes)
        for order in (0, 1, 2):
            row = []
            for m in range(nmodes):
                a = vecs[m]
                spec = np.array([float(pcv.mode_derivs(a, x)[order]) for x in xi_grid])
                fem = fem_mode_shape(femv[m], n_el, xi_grid, order=order)
                row.append(eps(spec, fem))
            results[order][n_el] = row

    labels = {0: "eps_W  (displacement)", 1: "eps_W' (slope)", 2: "eps_W'' (curvature)"}
    for order in (0, 1, 2):
        print(f"\n{labels[order]}: n_el -> mode1..5")
        for n_el in n_el_list:
            print(f"  n_el={n_el:4d}:", [f"{x:.3e}" for x in results[order][n_el]])

        # fitted rate: log(eps) vs log(h), h=1/n_el, least squares over modes 1 & 5
        hs = np.log([1.0 / n for n in n_el_list])
        for mode_idx in (0, 4):
            es = np.log([results[order][n][mode_idx] for n in n_el_list])
            rate = np.polyfit(hs, es, 1)[0]
            print(f"  fitted rate (mode {mode_idx+1}): O(h^{rate:.2f})")


if __name__ == "__main__":
    main()
