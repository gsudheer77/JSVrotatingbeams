"""
6c_node_fine_grid.py
==========================================================================
Fine-speed-grid refinement of the nodal-migration study (revision set 2).

1. Node positions for modes 2-8 on a uniform grid Gamma = 0.5, 1.0, ...,
   20.0 (plus 0.05, 0.1, 0.2, 0.25), so that the speed at which the
   first-order magnitude first departs by 10% is bracketed to within
   0.5 and then located by linear interpolation of
       rho(Gamma) = Delta_xi(Gamma) / (Gamma^2 d)
   between the last passing and the first failing grid speed.
2. Stability of the independent finite-difference estimate d of the
   derivative d xi / d(Gamma^2) at Gamma = 0: two-term fits
   Delta = d Gamma^2 + e Gamma^4 from the speed pairs
   (0.05, 0.1), (0.1, 0.2) and (0.25, 0.5).

Usage: python3 6c_node_fine_grid.py KEY [KEY ...]   (KEY in B1 B1R B2 BJ21 BJ62)
Writes ../results/6c_node_fine_grid_<KEY>.json
==========================================================================
"""
import json, time, sys
import numpy as np
from analysis_utils import *

G_SMALL = [0.05, 0.1, 0.2, 0.25]
G_GRID = [round(0.5 * k, 2) for k in range(1, 41)]
GAMMAS = [0.0] + G_SMALL + G_GRID
MODES = range(2, 9)


def run(key, label, mu, P, Rbar, N):
    t0 = time.time()
    xi = {}
    for G in GAMMAS:
        _, v = solve_modes(N, G, mu, P, Rbar, 9)
        for i in MODES:
            r, cnt = interior_nodes(v[i - 1], i - 1)
            assert cnt == i - 1, (key, i, G, cnt)
            xi[(G, i)] = r
    out = dict(key=key, label=label, N=N, mpmath=mp.__version__, backend=mp.libmp.BACKEND,
               gammas=GAMMAS, nodes=[])
    for i in MODES:
        for j in range(i - 1):
            x0 = xi[(0.0, i)][j]
            sh = {G: xi[(G, i)][j] - x0 for G in GAMMAS}
            fits = {}
            for (g1, g2) in ((0.05, 0.1), (0.1, 0.2), (0.25, 0.5)):
                A = np.array([[g1 ** 2, g1 ** 4], [g2 ** 2, g2 ** 4]])
                d, e = np.linalg.solve(A, [sh[g1], sh[g2]])
                fits[f"{g1}-{g2}"] = (d, e)
            d_ref = fits["0.1-0.2"][0]
            last_pass, first_fail, g10 = None, None, None
            prev_g, prev_rho = 0.0, 1.0
            for G in G_GRID:
                rho = sh[G] / (G * G * d_ref)
                if abs(rho - 1) > 0.10:
                    first_fail = G; last_pass = prev_g
                    # linear interpolation of |rho-1| = 0.1 between prev_g and G
                    f0, f1 = abs(prev_rho - 1) - 0.1, abs(rho - 1) - 0.1
                    g10 = prev_g + (G - prev_g) * (-f0) / (f1 - f0) if f1 != f0 else G
                    break
                prev_g, prev_rho = G, rho
            out["nodes"].append(dict(mode=i, node=j + 1, xi0=x0,
                                     shifts={str(G): sh[G] for G in GAMMAS},
                                     d_fits={k: [float(a), float(b)] for k, (a, b) in fits.items()},
                                     last_pass=last_pass, first_fail=first_fail, gamma10=g10))
    json.dump(out, open(os.path.join(HERE, "..", "results", f"6c_node_fine_grid_{key}.json"), "w"), indent=1)
    print(f"{label}: done in {time.time()-t0:.0f}s (mpmath {mp.__version__}, {mp.libmp.BACKEND})", flush=True)


if __name__ == "__main__":
    want = sys.argv[1:]
    for cfg in configs():
        if cfg[0] in want:
            run(*cfg)
