"""
1_node_migration_mechanism.py
==========================================================================
Master, self-contained script for the node-migration mechanism check.

Runs all five taper/hub-radius cases x modes 3-8, predicting the sign of
every interior-node shift via first-order eigenfunction perturbation
theory (see accompanying report) and checking it against the actual
solve() output at Gamma=0 vs Gamma=6. Prints a master results table at
the end (case, modes checked, node-predictions correct/total).

Usage:
    python3 1_node_migration_mechanism.py

Expects the manuscript's own beam_code/ package (chebyshev_rotating_beam.py,
per_case_verification.py) in a sibling directory ../beam_code, or set
BEAM_CODE_DIR to point at it.
==========================================================================
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BEAM_CODE_DIR = os.environ.get("BEAM_CODE_DIR", os.path.join(HERE, "..", "beam_code"))
sys.path.insert(0, BEAM_CODE_DIR)

import mpmath as mp
import chebyshev_rotating_beam as cb
import per_case_verification as pcv
from chebyshev_rotating_beam import solve, wang_wereley, banerjee

mp.mp.dps = 34
N = 40


def mode_val(a, xi):
    return pcv.mode_derivs(a, xi)[0]


def mode_dval(a, xi):
    return pcv.mode_derivs(a, xi)[1]


def tension_unit_shape_fn(mu, Rbar):
    integ = cb.polymul(mu, [Rbar, mp.mpf(1)])
    def Tb1(xi):
        return cb.polyint_def(integ, xi, mp.mpf(1))
    return Tb1


def mass_normalize(vecs, mu):
    out = []
    for a in vecs:
        def integ(xi, a=a):
            W = mode_val(a, xi)
            return cb.polyval(mu, xi) * W ** 2
        nrm2 = pcv._q(integ)
        s = mp.sqrt(nrm2)
        out.append([c / s for c in a])
    return out


def coupling_matrix(phis, Tb1):
    n = len(phis)
    C = [[mp.mpf(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            def integ(xi, i=i, j=j):
                return Tb1(xi) * mode_dval(phis[i], xi) * mode_dval(phis[j], xi)
            val = pcv._q(integ)
            C[i][j] = C[j][i] = val
    return C


def find_interior_nodes(a, n_expected, lo="0.02", hi="0.995", n_scan=250):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    grid = [lo + (hi - lo) * i / (n_scan - 1) for i in range(n_scan)]
    vals = [mode_val(a, x) for x in grid]
    roots = []
    for i in range(len(grid) - 1):
        if vals[i] == 0:
            roots.append(grid[i]); continue
        if vals[i] * vals[i + 1] < 0:
            xlo, xhi = grid[i], grid[i + 1]
            flo = vals[i]
            for _ in range(50):
                mid = (xlo + xhi) / 2
                fmid = mode_val(a, mid)
                if flo * fmid <= 0:
                    xhi = mid
                else:
                    xlo, flo = mid, fmid
            roots.append((xlo + xhi) / 2)
    return roots[:n_expected]


def run_case(label, mu, P, Rbar, modes_to_check, nmodes_trunc, verbose=True):
    t0 = time.time()
    if verbose:
        print(f"\n{'='*78}\n{label}  (Rbar={Rbar}, NMODES truncation={nmodes_trunc})\n{'='*78}")
    freqs0, vecs0 = solve(N, 0, mu, P, nmodes=nmodes_trunc, return_vectors=True, Rbar=Rbar)
    phis = mass_normalize(vecs0, mu)
    Tb1 = tension_unit_shape_fn(mu, Rbar)
    C = coupling_matrix(phis, Tb1)
    lam2 = [f ** 2 for f in freqs0]
    freqs6, vecs6 = solve(N, 6, mu, P, nmodes=nmodes_trunc, return_vectors=True, Rbar=Rbar)

    n_correct, n_total = 0, 0
    for n_idx in modes_to_check:
        mode_number = n_idx + 1
        if mode_number < 3:
            continue
        n_nodes = mode_number - 1
        nodes0 = find_interior_nodes(vecs0[n_idx], n_nodes)
        if len(nodes0) != n_nodes:
            if verbose:
                print(f"  mode {mode_number}: expected {n_nodes} interior nodes, found {len(nodes0)} -- skipping")
            continue

        coefs = [mp.mpf(0)] * len(phis[n_idx])
        for k in range(nmodes_trunc):
            if k == n_idx:
                continue
            denom = lam2[n_idx] - lam2[k]
            if denom == 0:
                continue
            w = C[n_idx][k] / denom
            coefs = [coefs[i] + w * phis[k][i] for i in range(len(coefs))]

        a6 = vecs6[n_idx]
        tipval0 = mode_val(vecs0[n_idx], mp.mpf(1))
        tipval6 = mode_val(a6, mp.mpf(1))
        if tipval0 * tipval6 < 0:
            a6 = [-c for c in a6]
        nodes6 = find_interior_nodes(a6, n_nodes)

        for j, xi0 in enumerate(nodes0):
            deltaW = mode_val(coefs, xi0)
            slope = mode_dval(phis[n_idx], xi0)
            dxi_dG2 = -deltaW / slope
            pred_dir = "root" if dxi_dG2 < 0 else "tip"
            if j < len(nodes6):
                actual_shift = nodes6[j] - xi0
                actual_dir = "root" if actual_shift < 0 else ("tip" if actual_shift > 0 else "none")
            else:
                actual_dir = "?"
            n_total += 1
            match = (pred_dir == actual_dir)
            n_correct += int(match)
            if verbose:
                flag = "OK" if match else ("~" if actual_dir == "none" else "MISMATCH")
                print(f"  mode {mode_number}, node {j+1}/{n_nodes} at xi0={mp.nstr(xi0,5)}: "
                      f"predicted->{pred_dir:5s}  actual(G0->G6)->{actual_dir:5s}  [{flag}]")
    if verbose:
        print(f"  case total: {n_correct}/{n_total} correct  (wall time: {time.time()-t0:.1f} s)")
    return n_correct, n_total


if __name__ == "__main__":
    mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
    mu2, P2 = wang_wereley(-0.8, [-0.95])
    mu_b2, P_b2 = banerjee(0.2, 1)
    mu_b6, P_b6 = banerjee(0.6, 2)

    cases = [
        ("Beam 1 (Wang-Wereley, moderate taper)", mu1, P1, mp.mpf(0), range(2, 8), 12),
        ("Beam 1, hub radius Rbar=0.5",           mu1, P1, mp.mpf("0.5"), range(2, 8), 14),
        ("Beam 2 (severe taper)",                 mu2, P2, mp.mpf(0), range(2, 8), 12),
        ("Banerjee c=0.2, n=1",                   mu_b2, P_b2, mp.mpf(0), range(2, 8), 12),
        ("Banerjee c=0.6, n=2",                   mu_b6, P_b6, mp.mpf(0), range(2, 8), 12),
    ]

    summary = []
    for label, mu, P, Rbar, modes, ntrunc in cases:
        c, t = run_case(label, mu, P, Rbar, modes, ntrunc)
        summary.append((label, c, t))

    print(f"\n\n{'='*78}\nMASTER SUMMARY -- all cases, modes 3-8\n{'='*78}")
    print(f"{'Case':<40}{'Correct/Total':>15}")
    grand_c, grand_t = 0, 0
    for label, c, t in summary:
        print(f"{label:<40}{c:>8}/{t:<6}")
        grand_c += c; grand_t += t
    print(f"\n{'TOTAL':<40}{grand_c:>8}/{grand_t:<6}")
