"""
8_compact_verification.py
==========================================================================
Compact verification diagnostics for representative moderate- and
severe-taper benchmark cases (manuscript Table 7; Reviewer 2, comment 9).

For each case and modes 1-5, at the benchmark resolution N_b:
  * algebraic backward error eta_i (Eq. 27)
  * dense-grid continuous residual eps_cont (Eq. 28; 801 points)
  * Rayleigh-quotient discrepancy |lam_RQ - lam|/lam (Eq. 16)
  * 1 - MAC between N_b and N_b + 10 shapes
  * mass-orthogonality error max_{i!=j} |<mu W_i, W_j>| / (norms)
The refinement change and tail estimate are taken from
9_refinement_and_rounding.py (same N_b).

Uses per_case_verification.py (beam_code) for every diagnostic.
==========================================================================
"""
import time
from analysis_utils import *
import per_case_verification as pcv

mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
mu2, P2 = wang_wereley(-0.8, [-0.95])
mu82, P82 = banerjee(0.8, 2)

CASES = [
    ("Beam 1, Gamma=0", mu1, P1, 0, 0, 50),
    ("Beam 1, Gamma=12", mu1, P1, 0, 12, 50),
    ("Banerjee n=2 c=0.8, Gamma=10", mu82, P82, 0, 10, 50),
    ("Beam 2, Gamma=0", mu2, P2, 0, 0, 80),
    ("Beam 2, Gamma=12", mu2, P2, 0, 12, 80),
]


def diagnostics(label, mu, P, Rbar, G, N, nmodes=5):
    t0 = time.time()
    f, v = cb.solve(N, G, mu, P, Rbar, 0, nmodes, return_vectors=True)
    f2, v2 = cb.solve(N + 10, G, mu, P, Rbar, 0, nmodes, return_vectors=True)
    A, B = cb.build_pencil(N, G, mu, P, Rbar, 0)
    nA = mp.mnorm(A, 1); nB = mp.mnorm(B, 1)
    rows = []
    for i in range(nmodes):
        lam, a = f[i], v[i]
        vv = mp.matrix(a)
        r = A * vv - lam ** 2 * (B * vv)
        eta = mp.norm(r, 2) / ((nA + lam ** 2 * nB) * mp.norm(vv, 2))
        cont = pcv.continuous_residual(a, lam ** 2, mu, P, mp.mpf(G), mp.mpf(Rbar), 0)
        rq = pcv.rayleigh_discrepancy(a, lam, mu, P, mp.mpf(G), mp.mpf(Rbar), 0)
        omac = abs(1 - pcv.mac(a, v2[i]))
        d1 = abs(f2[i] - lam) / f2[i]
        rows.append((i + 1, d1, eta, cont, rq, omac))
    morth = pcv.mass_orthogonality(v, mu, P)
    print(f"\n{label} (N_b={N}; {time.time()-t0:.0f}s)")
    print(" mode  dN(Nb->Nb+10)  backward_err  cont_resid   RQ_discrep   1-MAC")
    for (m, d1, eta, c, rq, om) in rows:
        print(f"  {m}    {mp.nstr(d1,2):>10}   {mp.nstr(eta,2):>10}   {mp.nstr(c,2):>10}   "
              f"{mp.nstr(rq,2):>10}   {mp.nstr(om,2):>10}")
    print(f"  max over modes: dN={mp.nstr(max(r[1] for r in rows),2)} eta={mp.nstr(max(r[2] for r in rows),2)} "
          f"cont={mp.nstr(max(r[3] for r in rows),2)} RQ={mp.nstr(max(r[4] for r in rows),2)} "
          f"1-MAC={mp.nstr(max(r[5] for r in rows),2)}  mass-orth={mp.nstr(morth,2)}", flush=True)


if __name__ == "__main__":
    mp.mp.dps = 34
    sel = sys.argv[1:]
    for k, c in enumerate(CASES):
        if sel and str(k) not in sel:
            continue
        diagnostics(*c)
