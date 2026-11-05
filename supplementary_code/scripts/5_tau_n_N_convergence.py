"""
5_tau_n_N_convergence.py
==========================================================================
N-refinement check for the fitted tau_n(Gamma=12) ~ n^slope scaling
exponent reported in Section 4.10.2: confirms the exponent is not an
artifact of the N=40-50 resolution used in 2_transition_criterion.py.

Runs the severe-taper beam (beam 2) and the Banerjee c=0.6, n=2 case --
the two configurations the review specifically asked to be checked --
at N = 40, 50, 60, modes 1-8, Gamma=12, and reports the fitted log-log
slope at each resolution.

Usage: python3 5_tau_n_N_convergence.py
==========================================================================
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
BEAM_CODE_DIR = os.environ.get("BEAM_CODE_DIR", os.path.join(HERE, "..", "beam_code"))
sys.path.insert(0, BEAM_CODE_DIR)

import numpy as np
import mpmath as mp
import chebyshev_rotating_beam as cb
import per_case_verification as pcv
from chebyshev_rotating_beam import solve, wang_wereley, banerjee

mp.mp.dps = 34
NMODES = 8
GAMMA = mp.mpf(12)
N_VALUES = [40, 50, 60]


def tau_values(mu, P, Rbar, Gamma, vecs):
    integ = cb.polymul(mu, [Rbar, mp.mpf(1)])
    def Tb1(xi):
        return cb.polyint_def(integ, xi, mp.mpf(1))
    G2 = Gamma ** 2
    out = []
    for a in vecs:
        def num_integ(xi, a=a):
            Wp = pcv.mode_derivs(a, xi)[1]
            return Tb1(xi) * Wp ** 2
        def den_integ(xi, a=a):
            Wpp = pcv.mode_derivs(a, xi)[2]
            return cb.polyval(P, xi) * Wpp ** 2
        num = G2 * pcv._q(num_integ)
        den = pcv._q(den_integ)
        out.append(num / den)
    return out


def fit_slope(mu, P, Rbar, N):
    freqs, vecs = solve(N, GAMMA, mu, P, nmodes=NMODES, return_vectors=True, Rbar=Rbar)
    taus = tau_values(mu, P, Rbar, GAMMA, vecs)
    ns = np.arange(1, NMODES + 1)
    taus_f = np.array([float(t) for t in taus])
    slope, _ = np.polyfit(np.log(ns), np.log(taus_f), 1)
    return slope, taus_f


def run_case(label, mu, P, Rbar):
    print(f"\n{'='*78}\n{label} (Rbar={Rbar}), Gamma=12, modes 1-8\n{'='*78}")
    slopes = []
    for N in N_VALUES:
        slope, taus = fit_slope(mu, P, Rbar, N)
        slopes.append(slope)
        tau_str = ", ".join(f"{t:.5f}" for t in taus)
        print(f"N={N:3d}:  slope = {slope:+.4f}   tau_n = [{tau_str}]")
    spread = max(slopes) - min(slopes)
    print(f"slope spread over N={N_VALUES}: {spread:.2e}  "
          f"(stable to {max(0, -int(np.floor(np.log10(spread + 1e-16))))} decimal places)")
    return slopes


if __name__ == "__main__":
    mu2, P2 = wang_wereley(-0.8, [-0.95])   # beam 2, severe taper
    mu_b6, P_b6 = banerjee(0.6, 2)          # Banerjee c=0.6, n=2

    results = {}
    results["Beam 2 (severe taper)"] = run_case("Beam 2 (severe taper)", mu2, P2, mp.mpf(0))
    results["Banerjee c=0.6, n=2"] = run_case("Banerjee c=0.6, n=2", mu_b6, P_b6, mp.mpf(0))

    print(f"\n\n{'='*60}\nSUMMARY: fitted slope vs N\n{'='*60}")
    print(f"{'Case':<28}" + "".join(f"N={N:<8}" for N in N_VALUES))
    for label, slopes in results.items():
        print(f"{label:<28}" + "".join(f"{s:+.4f}  " for s in slopes))
