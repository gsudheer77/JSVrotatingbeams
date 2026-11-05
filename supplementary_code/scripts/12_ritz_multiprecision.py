"""
12_ritz_multiprecision.py
==========================================================================
Rayleigh-Ritz upper bounds computed in 34-digit arithmetic
(manuscript Section 4.3; Reviewer 2, comment 8).

The double-precision Ritz check (3_ritz_upper_bound.py) produced a few
tiny negative gaps (e.g. -1.8e-10) at its finest trial space, i.e. the
floating-point Ritz value fell *below* the collocation value although the
exact Ritz value cannot.  Here the same conforming trial space
  phi_m(xi) = xi^2 P_m(2 xi - 1),  m = 0..M-1   (W(0)=W'(0)=0 exactly)
is assembled with Gauss-Legendre quadrature that is exact for the
polynomial integrands, and the generalized symmetric-definite problem is
solved by Cholesky reduction in mpmath at 34 digits, so that round-off is
~1e-30 and the computed gaps lam_Ritz^2 - lam^2 can be trusted in sign.

Gap reported: (lam_Ritz,M^2 - lam_ref^2) / lam_ref^2, lam_ref from the
collocation benchmark (N=50 beam 1, N=90 beam 2).
==========================================================================
"""
from analysis_utils import *

mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
mu2, P2 = wang_wereley(-0.8, [-0.95])


def legendre_shifted_coeffs(M):
    """ascending-power coefficients (mpf) of P_m(2xi-1), m=0..M-1."""
    out = []
    for m in range(M):
        c = [mp.mpf(0)] * (m + 1)
        for k in range(m + 1):
            c[k] = (-1) ** (m + k) * mp.binomial(m, k) * mp.binomial(m + k, k)
        out.append(c)
    return out


def ritz(mu, P, Rbar, G, M, nmodes=3):
    mu = [mp.mpf(str(x)) for x in mu]; P = [mp.mpf(str(x)) for x in P]
    basis = []
    for c in legendre_shifted_coeffs(M):
        phi = [mp.mpf(0), mp.mpf(0)] + c          # xi^2 * L_m
        basis.append((phi, cb.polyder(phi), cb.polyder(cb.polyder(phi))))
    integ = cb.polymul(mu, [mp.mpf(Rbar), mp.mpf(1)])
    Aint = [mp.mpf(0)] + [integ[i] / (i + 1) for i in range(len(integ))]
    T1c = [cb.polyval(Aint, mp.mpf(1)) - Aint[0]] + [-a for a in Aint[1:]]
    nq = M + 8
    from mpmath.calculus.quadrature import GaussLegendre
    deg = 1
    while 3 * 2 ** (deg - 1) < nq:
        deg += 1
    nodes = GaussLegendre(mp.mp).get_nodes(mp.mpf(0), mp.mpf(1), degree=deg, prec=mp.mp.prec)
    vals = []
    for x, w in nodes:
        vals.append((w, cb.polyval(mu, x), cb.polyval(P, x), cb.polyval(T1c, x),
                     [(cb.polyval(b[0], x), cb.polyval(b[1], x), cb.polyval(b[2], x)) for b in basis]))
    K = mp.zeros(M); Mm = mp.zeros(M)
    G2 = mp.mpf(G) ** 2
    for (w, mv, pv, tv, bv) in vals:
        for i in range(M):
            for j in range(i, M):
                K[i, j] += w * (pv * bv[i][2] * bv[j][2] + G2 * tv * bv[i][1] * bv[j][1])
                Mm[i, j] += w * mv * bv[i][0] * bv[j][0]
    for i in range(M):
        for j in range(i):
            K[i, j] = K[j, i]; Mm[i, j] = Mm[j, i]
    Lc = mp.cholesky(Mm)
    Li = mp.inverse(Lc)
    S = Li * K * Li.T
    S = (S + S.T) / 2
    E = mp.eigsy(S, eigvals_only=True)
    E = sorted([E[i] for i in range(M)])
    return E[:nmodes]


if __name__ == "__main__":
    mp.mp.dps = 34
    for label, mu, P, Nref, Ms in (("Beam 1", mu1, P1, 50, (13, 16, 20, 24, 28)),
                                   ("Beam 2", mu2, P2, 90, (13, 16, 20, 24, 28))):
        print(f"\n{label}  (reference: collocation N={Nref})")
        for G in (0, 6, 12):
            ref = [x ** 2 for x in solve(Nref, G, mu, P, nmodes=3)]
            print(f"  Gamma={G}")
            for M in Ms:
                E = ritz(mu, P, 0, G, M)
                gaps = [(E[i] - ref[i]) / ref[i] for i in range(3)]
                print(f"    M={M:2d}: rel. gap (lam_Ritz^2-lam^2)/lam^2 = " +
                      "  ".join(f"{mp.nstr(g, 3):>10}" for g in gaps) +
                      ("   all positive" if all(g > 0 for g in gaps) else "   NEGATIVE GAP"), flush=True)
