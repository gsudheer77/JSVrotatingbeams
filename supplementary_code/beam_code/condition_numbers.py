"""Boundary-reduction condition-number comparison (Appendix B): kappa2(B_I)
for the well-scaled elimination set {a_0,a_1,a_{N-1},a_N} versus the
ill-conditioned highest-degree set {a_{N-3},...,a_N}, as a function of N."""
import mpmath as mp
import chebyshev_rotating_beam as cb

def kappa_BI(N, mu, P, Gamma, elim_cols):
    A, B = cb.build_pencil(N, Gamma, mu, P)
    n1 = A.rows
    BI = mp.zeros(4, 4)
    cons = [0, 1, n1 - 2, n1 - 1]
    for i, r in enumerate(cons):
        for p, c in enumerate(elim_cols):
            BI[i, p] = A[r, c]
    return mp.mpf(mp.norm(BI, 2)) * mp.mpf(mp.norm(mp.inverse(BI), 2))

mp.mp.dps = 34
mu, P = cb.wang_wereley(-0.5, [-1.5, 0.75, -0.125])

print(f"{'N':>4}  {'kappa2, well-scaled':>22}  {'kappa2, highest-degree':>24}")
for N in [20, 30, 40, 50, 60, 70, 80]:
    n1 = N + 1
    m = n1 - 4
    k_good = kappa_BI(N, mu, P, 0, [0, 1, n1 - 2, n1 - 1])
    k_bad  = kappa_BI(N, mu, P, 0, [m, m + 1, m + 2, m + 3])
    print(f"{N:>4}  {mp.nstr(k_good,4):>22}  {mp.nstr(k_bad,4):>24}")
