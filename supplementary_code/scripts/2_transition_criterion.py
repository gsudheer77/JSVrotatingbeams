"""
2_transition_criterion.py
==========================================================================
Master script for the exact bending-to-tension transition ratio

    tau_n(Gamma) = [Gamma^2 int T1(xi) W_n'(xi;Gamma)^2 dxi]
                   / [int P(xi) W_n''(xi;Gamma)^2 dxi]

evaluated on the true (Gamma-dependent) mode shape returned by solve()
-- no small-Gamma approximation. Runs all five taper/hub-radius cases,
modes 1-8, several Gamma, and fits tau_n vs mode number at Gamma=12 by
least squares in log-log space.

Usage: python3 2_transition_criterion.py
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
N = 40
NMODES = 8


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


def run_case(label, mu, P, Rbar, Gammas):
    print(f"\n{'='*90}\n{label} (Rbar={Rbar})\n{'='*90}")
    header = "mode".ljust(6) + "".join(f"tau(G={g})".rjust(14) for g in Gammas)
    print(header)
    all_rows = {}
    for G in Gammas:
        freqs, vecs = solve(N, G, mu, P, nmodes=NMODES, return_vectors=True, Rbar=Rbar)
        taus = tau_values(mu, P, Rbar, mp.mpf(G), vecs)
        for m in range(NMODES):
            all_rows.setdefault(m, []).append(taus[m])
    for m in range(NMODES):
        row = "".join(mp.nstr(v, 4).rjust(14) for v in all_rows[m])
        print(f"{m+1:<6}{row}")

    ns = np.arange(1, NMODES + 1)
    taus12 = np.array([float(all_rows[m][-1]) for m in range(NMODES)])
    slope, intercept = np.polyfit(np.log(ns), np.log(taus12), 1)
    print(f"log-log fit at Gamma={Gammas[-1]}: tau_n ~ n^({slope:.3f})")
    return all_rows, slope


if __name__ == "__main__":
    mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
    mu2, P2 = wang_wereley(-0.8, [-0.95])
    mu_b2, P_b2 = banerjee(0.2, 1)
    mu_b6, P_b6 = banerjee(0.6, 2)

    Gammas = [2, 4, 6, 8, 10, 12]
    cases = [
        ("Beam 1 (moderate taper)", mu1, P1, mp.mpf(0)),
        ("Beam 1, hub radius Rbar=0.5", mu1, P1, mp.mpf("0.5")),
        ("Beam 2 (severe taper)", mu2, P2, mp.mpf(0)),
        ("Banerjee c=0.2, n=1", mu_b2, P_b2, mp.mpf(0)),
        ("Banerjee c=0.6, n=2", mu_b6, P_b6, mp.mpf(0)),
    ]
    slopes = []
    for label, mu, P, Rbar in cases:
        _, slope = run_case(label, mu, P, Rbar, Gammas)
        slopes.append((label, slope))

    print(f"\n\n{'='*60}\nSUMMARY OF FITTED SCALING EXPONENTS (tau_n ~ n^slope, Gamma=12)\n{'='*60}")
    for label, slope in slopes:
        print(f"{label:<32} slope = {slope:+.3f}")
    mean_slope = np.mean([s for _, s in slopes])
    print(f"\nmean across all five cases: {mean_slope:+.3f}")
