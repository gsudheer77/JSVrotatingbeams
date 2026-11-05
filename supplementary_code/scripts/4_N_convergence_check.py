"""
4_N_convergence_check.py
==========================================================================
Demonstrates that the collocation reference values used elsewhere are
actually converged (not just "N=50 differs from N=80, so use N=80") --
a full N-sweep with the relative change reported at each step, for both
beams, modes 1-3, Gamma=0.

Usage: python3 4_N_convergence_check.py
==========================================================================
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
BEAM_CODE_DIR = os.environ.get("BEAM_CODE_DIR", os.path.join(HERE, "..", "beam_code"))
sys.path.insert(0, BEAM_CODE_DIR)

import mpmath as mp
from chebyshev_rotating_beam import solve, wang_wereley

mp.mp.dps = 34


def sweep(label, mu, P, Ns, nmodes=3, Gamma=0):
    print(f"\n{'='*80}\n{label}, Gamma={Gamma}, modes 1-{nmodes}\n{'='*80}")
    prev = None
    for N in Ns:
        t0 = time.time()
        f = solve(N, Gamma, mu, P, nmodes=nmodes)
        l2 = [float(x ** 2) for x in f]
        dt = time.time() - t0
        if prev is not None:
            rel = [abs(l2[i] - prev[i]) / l2[i] for i in range(nmodes)]
            print(f"N={N:3d}: lam^2={[round(x,9) for x in l2]}  "
                  f"rel.change vs prev={[f'{r:.2e}' for r in rel]}  ({dt:.1f}s)")
        else:
            print(f"N={N:3d}: lam^2={[round(x,9) for x in l2]}  (first row)  ({dt:.1f}s)")
        prev = l2


if __name__ == "__main__":
    mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
    mu2, P2 = wang_wereley(-0.8, [-0.95])

    sweep("Beam 1 (moderate taper)", mu1, P1, [30, 40, 50, 60])
    sweep("Beam 2 (severe taper)", mu2, P2, [50, 60, 70, 80, 90, 100])
