"""
13_check_endpoint_formulas.py
==========================================================================
Exact check of the endpoint derivative formulas, Eq. (24) / (A.5)
(Reviewer 2, comment 6):

    T_k''(+-1)  = (+-1)^k     (k^4 - k^2) / 3
    T_k'''(+-1) = (+-1)^(k+1) (k^6 - 5k^4 + 4k^2) / 15

i.e. the COMPLETE numerators divided by 3 and 15.  They are compared, in
exact rational arithmetic, against
  (a) the factored forms k^2(k^2-1)/3 and k^2(k^2-1)(k^2-4)/15 (App. A),
  (b) the general identity T_k^(p)(1) = prod_{m<p}(k^2-m^2)/(2m+1),
  (c) direct symbolic differentiation of the Chebyshev polynomial T_k
      (sympy), evaluated at x = +1 and x = -1,
  (d) the rows actually assembled by chebyshev_rotating_beam.build_pencil,
and the *mis-grouped* reading k^4 - k^2/3 is shown to be wrong.
==========================================================================
"""
from fractions import Fraction as F
import sympy as sp
from analysis_utils import *

x = sp.symbols("x")
bad = 0
for k in range(0, 41):
    a2 = F(k ** 4 - k ** 2, 3); a3 = F(k ** 6 - 5 * k ** 4 + 4 * k ** 2, 15)
    f2 = F(k ** 2 * (k ** 2 - 1), 3); f3 = F(k ** 2 * (k ** 2 - 1) * (k ** 2 - 4), 15)
    g2 = F(k ** 2) * F(k ** 2 - 1, 3); g3 = F(k ** 2) * F(k ** 2 - 1, 3) * F(k ** 2 - 4, 5)
    T = sp.chebyshevt(k, x)
    s2p = sp.Rational(sp.diff(T, x, 2).subs(x, 1)); s3p = sp.Rational(sp.diff(T, x, 3).subs(x, 1))
    s2m = sp.Rational(sp.diff(T, x, 2).subs(x, -1)); s3m = sp.Rational(sp.diff(T, x, 3).subs(x, -1))
    ok = (a2 == f2 == g2 == F(int(s2p.p), int(s2p.q)) and a3 == f3 == g3 == F(int(s3p.p), int(s3p.q))
          and F(int(s2m.p), int(s2m.q)) == (-1) ** k * a2 and F(int(s3m.p), int(s3m.q)) == (-1) ** (k + 1) * a3)
    bad += (not ok)
print(f"k = 0..40: grouped formulas agree with factored form, product identity and sympy at x=+-1: "
      f"{'ALL OK' if bad == 0 else str(bad) + ' FAILURES'}")

# the mis-grouped reading differs for every k >= 2
k = 5
print(f"example k=5: (k^4-k^2)/3 = {F(k**4-k**2,3)}  vs mis-grouped k^4 - k^2/3 = {F(k**4) - F(k**2,3)}")

# rows assembled by the solver
mp.mp.dps = 34
mu, P = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
N = 20
A, _ = cb.build_pencil(N, 0, mu, P)
err = max(max(abs(A[N - 1, k] - mp.mpf(k ** 4 - k ** 2) / 3), abs(A[N, k] - mp.mpf(k ** 6 - 5 * k ** 4 + 4 * k ** 2) / 15))
          for k in range(N + 1))
print(f"build_pencil tip rows vs grouped formulas (N={N}): max abs difference = {mp.nstr(err, 3)}")
