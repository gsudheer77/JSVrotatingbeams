"""
10_resolved_modes.py
==========================================================================
Mode acceptance and the number of reliably resolved modes per resolution
(manuscript Section 4.2, Table 3; Reviewer 2, comment 10).

The reduced pencil has N-3 finite eigenvalues. They are classified as
  (a) rejected immediately: complex, |Im mu| >= 1e-15 (1 + |Re mu|),
      or Re mu <= 0   (mu = lambda^2);
  (b) real and positive, then tested by
        - N-refinement against a higher-resolution solution (N_ref),
        - interior nodal count (mode i must have i-1 interior zeros),
        - dense-grid continuous residual (201 points, Eq. 28).
A real eigenvalue is counted as *resolved at tolerance tol* if it and
every lower mode satisfy |lam_i(N) - lam_i(N_ref)|/lam_i(N_ref) < tol and
the nodal count is correct.  Real eigenvalues that fail are real
spurious / under-resolved eigenvalues; they are reported separately
(they are NOT removed by the reality test).
==========================================================================
"""
import time
from analysis_utils import *
import per_case_verification as pcv

mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
mu2, P2 = wang_wereley(-0.8, [-0.95])

CASES = [("Beam 1, Gamma=0", mu1, P1, 0, [20, 30, 40, 50], 70),
         ("Beam 1, Gamma=10", mu1, P1, 10, [20, 30, 40, 50], 70),
         ("Beam 2, Gamma=0", mu2, P2, 0, [40, 60, 80], 100)]
TOLS = [1e-6, 1e-10]


def full_spectrum(N, G, mu, P):
    """All eigenvalues of the reduced standard problem, plus the kept
    real-positive eigenpairs, using the same reduction as cb.solve."""
    A, B = cb.build_pencil(N, G, mu, P)
    n1 = A.rows; m = n1 - 4
    cons = [0, 1, n1 - 2, n1 - 1]
    interior = [r for r in range(n1) if r not in cons]
    elim = [0, 1, n1 - 2, n1 - 1]
    keep = [c for c in range(n1) if c not in elim]
    C2 = mp.matrix([[A[r, c] for c in elim] for r in cons])
    C1 = mp.matrix([[A[r, c] for c in keep] for r in cons])
    Xm = mp.inverse(C2) * C1
    L, R = mp.zeros(m), mp.zeros(m)
    for i, r in enumerate(interior):
        for j, ck in enumerate(keep):
            aL, aR = A[r, ck], B[r, ck]
            for p, ce in enumerate(elim):
                aL -= A[r, ce] * Xm[p, j]; aR -= B[r, ce] * Xm[p, j]
            L[i, j], R[i, j] = aL, aR
    E, V = mp.eig(mp.inverse(R) * L)
    tol = mp.mpf('1e-15')
    real_pos = [i for i in range(len(E)) if abs(mp.im(E[i])) < tol * (1 + abs(mp.re(E[i]))) and mp.re(E[i]) > 0]
    real_pos.sort(key=lambda i: mp.re(E[i]))
    lam = [mp.sqrt(mp.re(E[i])) for i in real_pos]
    vecs = []
    for i in real_pos:
        vred = [V[r, i] for r in range(m)]
        adep = [-sum(Xm[p, j] * vred[j] for j in range(m)) for p in range(4)]
        full = [mp.mpf(0)] * n1
        for j, ck in enumerate(keep): full[ck] = mp.re(vred[j])
        for p, ce in enumerate(elim): full[ce] = mp.re(adep[p])
        vecs.append(full)
    n_complex = sum(1 for i in range(len(E)) if abs(mp.im(E[i])) >= tol * (1 + abs(mp.re(E[i]))))
    n_nonpos = len(E) - n_complex - len(real_pos)
    return len(E), n_complex, n_nonpos, lam, vecs


if __name__ == "__main__":
    for label, mu, P, G, Ns, Nref in CASES:
        t0 = time.time()
        _, _, _, lam_ref, _ = full_spectrum(Nref, G, mu, P)
        print(f"\n{label} (reference N={Nref}); {time.time()-t0:.0f}s", flush=True)
        print("   N  finite  complex  Re<=0  real>0   resolved(1e-6)  resolved(1e-10)  "
              "real-but-unresolved  largest rel.err of real eigs  max cont.resid (resolved)")
        for N in Ns:
            tot, ncx, nnp, lam, vecs = full_spectrum(N, G, mu, P)
            nreal = len(lam)
            ok_nodes, errs, conts = [], [], []
            for i, (l, a) in enumerate(zip(lam, vecs)):
                af = to_float(a)
                _, cnt = interior_nodes(af, 10 ** 6, n_scan=max(4000, 60 * (i + 1)))
                ok_nodes.append(cnt == i)
                errs.append(abs(l - lam_ref[i]) / lam_ref[i] if i < len(lam_ref) else mp.inf)
            res = {}
            for tol in TOLS:
                k = 0
                while k < nreal and errs[k] < tol and ok_nodes[k]:
                    k += 1
                res[tol] = k
            for i in range(res[TOLS[0]]):
                conts.append(pcv.continuous_residual(vecs[i], lam[i] ** 2, mu, P, mp.mpf(G), mp.mpf(0), 0, ngrid=201))
            print(f"  {N:3d}  {tot:5d}  {ncx:6d}  {nnp:5d}  {nreal:6d}   {res[1e-6]:12d}   {res[1e-10]:13d}   "
                  f"{nreal - res[1e-6]:16d}        {mp.nstr(max(errs[res[1e-6]:]) if nreal > res[1e-6] else 0, 2):>10}"
                  f"        {mp.nstr(max(conts) if conts else 0, 2)}", flush=True)
