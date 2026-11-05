"""
table12_generate.py
==========================================================================
Definitive, single-script generator for manuscript Table 12 (Section
4.12): MAC and amplitude-optimized displacement error eps_W between the
present spectral solution and an independent Hermite-cubic FEM solution,
beam 1, Gamma=0, present solution at N=30 (31 coefficients), for the FEM
mesh sequence n_el = 16, 64, 256 (32/128/512 DOFs), modes 1-5.

The single source of truth for Table 12, callable end to end from the
public solver and FEM code with no manual steps. eps_W' and eps_W''
(slope/curvature) are produced by slope_curvature_error.py, which uses
the identical grid, present solution and FEM sequence, so all three
error quantities are mutually consistent.

MAC is computed exactly as defined in the manuscript (Eq. 30): the
squared cosine between the two shape vectors sampled on the common
801-point grid described in Section 4.12, using the *displacement*
values (order=0) from the same mode_derivs / fem_mode_shape machinery
used by slope_curvature_error.py -- i.e. MAC and eps_W are computed from
one shared pair of sampled vectors per (n_el, mode), not from two
independently-normalized copies.
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


def mac_discrete(spec, fem):
    """Discrete modal assurance criterion, Eq. (30), on the shared grid."""
    spec = np.asarray(spec, dtype=float)
    fem = np.asarray(fem, dtype=float)
    num = np.dot(spec, fem) ** 2
    den = np.dot(spec, spec) * np.dot(fem, fem)
    return num / den


def main():
    n_el_list = [16, 64, 256]
    nmodes = 5
    xi_grid = np.linspace(0.0, 1.0, 801)

    freqs, vecs = cb.solve(30, 0, mu, P, nmodes=nmodes, return_vectors=True)

    print("Table 12 (regenerated) - beam 1, Gamma=0, present N=30 (31 coeff.)")
    print(f"{'n_el':>5} {'DOFs':>5} {'metric':>6}  " + "  ".join(f"mode {m+1}".rjust(11) for m in range(nmodes)))
    for n_el in n_el_list:
        femf, femv, _ = solve_fem(n_el, 0.0, [1.0, -0.5], [1.0, -1.5, 0.75, -0.125],
                                   nmodes=nmodes)
        macs, epss = [], []
        for m in range(nmodes):
            a = vecs[m]
            spec = np.array([float(pcv.mode_derivs(a, x)[0]) for x in xi_grid])
            fem = fem_mode_shape(femv[m], n_el, xi_grid, order=0)
            macs.append(mac_discrete(spec, fem))
            epss.append(eps(spec, fem))
        dofs = 2 * (n_el + 1) - 2
        print(f"{n_el:5d} {dofs:5d} {'MAC':>6}  " + "  ".join(f"{v:.10f}".rjust(11) for v in macs))
        print(f"{n_el:5d} {dofs:5d} {'eps_W':>6}  " + "  ".join(f"{v:.3e}".rjust(11) for v in epss))


if __name__ == "__main__":
    main()
