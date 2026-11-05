"""
14_mode_shape_rates.py
==========================================================================
Mode-shape verification against the independent Hermite-cubic FEM with a
single, explicit error definition (manuscript Section 4.5, Table 10;
Reviewer 2, comment 11).

For mode i of beam 1 (Gamma = 0 and 10) the present shape W (N = 40,
34-digit eigenvector, evaluated analytically) and the FEM shape W_h are
sampled on the 801-point grid G = {0, 1/800, ..., 1}.  ONE scale factor

    s* = argmin_s || W - s W_h ||_G = <W, W_h>_G / <W_h, W_h>_G

is fixed from the displacement and then used for every derivative order:

    eps^(m) = || W^(m) - s* W_h^(m) ||_G / || W^(m) ||_G ,  m = 0, 1, 2.

(eps^(0) is the amplitude-optimized displacement error; note
 (eps^(0))^2 = 1 - MAC exactly, so the MAC adds no information beyond eps^(0).)
The observed order between successive meshes is p = log2(eps(h)/eps(h/2)).
The mesh is refined until double-precision round-off in the FEM
eigenvectors takes over, which separates the discretization-dominated
range (p close to 4, 3, 2 for m = 0, 1, 2) from the round-off range.
==========================================================================
"""
import numpy as np
from analysis_utils import *
from fem_hermite_solver import solve_fem, fem_mode_shape

mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
MESHES = [4, 8, 16, 32, 64, 128, 256, 512]
NM = 5
G = np.linspace(0, 1, 801)

if __name__ == "__main__":
    for Gamma in (0, 10):
        f, v = solve_modes(40, Gamma, mu1, P1, 0.0, NM)
        lam = [float(x) for x in f]
        spec = [[cheb_eval(a, G, m) for m in range(3)] for a in v]
        print(f"\nBeam 1, Gamma={Gamma}: present N=40 vs Hermite-cubic FEM (double precision)")
        res = {}
        for ne in MESHES:
            ff, fv, _ = solve_fem(ne, float(Gamma), [1.0, -0.5], [1.0, -1.5, 0.75, -0.125], nmodes=NM)
            row = []
            for i in range(NM):
                fem = [fem_mode_shape(fv[i], ne, G, order=m) for m in range(3)]
                s = np.dot(spec[i][0], fem[0]) / np.dot(fem[0], fem[0])
                e = [np.linalg.norm(spec[i][m] - s * fem[m]) / np.linalg.norm(spec[i][m]) for m in range(3)]
                mac = np.dot(spec[i][0], fem[0]) ** 2 / (np.dot(spec[i][0], spec[i][0]) * np.dot(fem[0], fem[0]))
                ef = abs(ff[i] - lam[i]) / lam[i]
                row.append((e, 1 - mac, ef))
            res[ne] = row
        for m, name in ((0, "eps_W  (displacement)"), (1, "eps_W' (slope)"), (2, "eps_W''(curvature)")):
            print(f"  {name}")
            print("    n_el  DOF  " + "  ".join(f"   mode {i+1}  (p)  " for i in range(NM)))
            prev = None
            for ne in MESHES:
                cells = []
                for i in range(NM):
                    e = res[ne][i][0][m]
                    p = "" if prev is None else f"{np.log2(prev[i][0][m] / e):5.2f}"
                    cells.append(f"{e:9.2e} ({p:>5})")
                print(f"    {ne:4d} {2*ne:4d}  " + "  ".join(cells))
                prev = res[ne]
        print("  check (eps_W)^2 vs 1-MAC at n_el=16, mode 5: "
              f"{res[16][4][0][0]**2:.4e} vs {res[16][4][1]:.4e}")
        print("  FEM frequency error, modes 1/5:")
        for ne in MESHES:
            print(f"    {ne:4d}  {res[ne][0][2]:.2e}  {res[ne][4][2]:.2e}")
