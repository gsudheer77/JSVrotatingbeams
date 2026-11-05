"""
11_cost_matched_accuracy.py
==========================================================================
Elapsed time at matched accuracy and cost per converged mode
(manuscript Section 4.6, Table 10; Reviewer 2, comment 12).

Beam 1, Gamma = 0. Reference: 34-digit collocation at N = 70.
Three solvers, each timed end-to-end (assembly + reduction + full
eigensolve, best of 3 for the fast double-precision solvers):

  (P34) present trigonometric collocation, 34-digit mpmath (gmpy2 backend)
  (P64) the same discretization assembled and solved in IEEE double
        (vectorized NumPy; same rows, same well-scaled reduction)
  (FEM) Hermite-cubic finite elements, IEEE double, scipy.linalg.eigh

For each resolution the number of *converged modes* at tolerance tol is
the number of consecutive modes from the bottom of the spectrum whose
relative frequency error is below tol.  "Cost per converged mode" is the
elapsed time divided by that number.  For each tolerance the cheapest
resolution of each solver that resolves at least the first five modes is
reported (the matched-accuracy comparison).
==========================================================================
"""
import time, platform
from analysis_utils import *
from fem_hermite_solver import solve_fem
import scipy

mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
TOLS = [1e-4, 1e-6, 1e-8, 1e-12]
NMAX_MODES = 40


def solve_present_double(N, Gamma, mu, P, Rbar=0.0):
    mu = np.array([float(c) for c in mu]); P = np.array([float(c) for c in P])
    pp = np.polynomial.polynomial
    Pp, Ppp = pp.polyder(P), pp.polyder(P, 2)
    integ = pp.polymul(mu, [Rbar, 1.0]); Aint = pp.polyint(integ)
    Tb = lambda x: Gamma ** 2 * (pp.polyval(1.0, Aint) - pp.polyval(x, Aint))
    Tbp = lambda x: -Gamma ** 2 * pp.polyval(x, integ)
    n1 = N + 1; k = np.arange(n1, dtype=float)
    A = np.zeros((n1, n1)); B = np.zeros((n1, n1))
    A[0] = (-1.0) ** k; A[1] = (-1.0) ** (k + 1) * k ** 2
    A[N - 1] = (k ** 4 - k ** 2) / 3.0; A[N] = (k ** 6 - 5 * k ** 4 + 4 * k ** 2) / 15.0
    th = np.array([(N - t) * np.pi / N for t in range(1, N - 2)])[:, None]
    s, c = np.sin(th), np.cos(th); sk, ck = np.sin(k * th), np.cos(k * th)
    d0 = ck; d1 = k * sk / s
    d2 = (k * sk * c - k ** 2 * ck * s) / s ** 3
    d3 = -k / s ** 5 * ((k ** 2 - 1) * s ** 2 * sk + 3 * c * (k * s * ck - c * sk))
    d4 = k / s ** 7 * (3 * (5 - 2 * (k ** 2 + 1) * s ** 2) * c * sk + k * ((k ** 2 + 11) * s ** 2 - 15) * s * ck)
    xi = ((np.cos(th) + 1) / 2)
    Pv = pp.polyval(xi, P)
    q3 = 2 * pp.polyval(xi, Pp) / Pv; q2 = (pp.polyval(xi, Ppp) - Tb(xi)) / Pv
    q1 = -Tbp(xi) / Pv; q0 = pp.polyval(xi, mu) / Pv
    A[2:N - 1] = 16 * d4 + 8 * q3 * d3 + 4 * q2 * d2 + 2 * q1 * d1
    B[2:N - 1] = q0 * d0
    cons = [0, 1, N - 1, N]; elim = [0, 1, N - 1, N]
    interior = [r for r in range(n1) if r not in cons]; keep = [c_ for c_ in range(n1) if c_ not in elim]
    Xm = np.linalg.solve(A[np.ix_(cons, elim)], A[np.ix_(cons, keep)])
    L = A[np.ix_(interior, keep)] - A[np.ix_(interior, elim)] @ Xm
    R = B[np.ix_(interior, keep)] - B[np.ix_(interior, elim)] @ Xm
    E = np.linalg.eigvals(np.linalg.solve(R, L))
    E = E[(np.abs(E.imag) < 1e-8 * (1 + np.abs(E.real))) & (E.real > 0)].real
    return np.sqrt(np.sort(E))


def nconv(lam, ref, tol):
    k = 0
    while k < len(lam) and k < len(ref) and abs(lam[k] - ref[k]) / ref[k] < tol:
        k += 1
    return k


def timed(fn, reps):
    best = 1e99
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t)
    return best, out


if __name__ == "__main__":
    print(f"Python {platform.python_version()}, mpmath {mp.__version__} (backend {mp.libmp.BACKEND}), "
          f"NumPy {np.__version__}, SciPy {scipy.__version__}, {platform.processor() or platform.machine()}")
    ref = [float(x) for x in solve(70, 0, mu1, P1, nmodes=NMAX_MODES)]
    rows = []
    for N in (16, 20, 21, 25, 26, 30, 31, 35, 40, 50):
        t, lam = timed(lambda: [float(x) for x in solve(N, 0, mu1, P1, nmodes=NMAX_MODES)], 1)
        rows.append(("P34", N, N + 1, t, lam))
    for N in (16, 20, 21, 25, 26, 30, 35, 40):
        t, lam = timed(lambda: solve_present_double(N, 0.0, mu1, P1), 5)
        rows.append(("P64", N, N + 1, t, list(lam)))
    for ne in (4, 8, 16, 32, 64, 128, 256, 512):
        t, out = timed(lambda: solve_fem(ne, 0.0, [1.0, -0.5], [1.0, -1.5, 0.75, -0.125], nmodes=min(NMAX_MODES, 2 * ne)), 3)
        rows.append(("FEM", ne, 2 * ne, t, list(out[0])))

    print("\nsolver  res   DOF   time[s]    " + "  ".join(f"nconv({t:.0e})" for t in TOLS)
          + "   err mode1   err mode5")
    for (s, n, dof, t, lam) in rows:
        nc = [nconv(lam, ref, tol) for tol in TOLS]
        e1 = abs(lam[0] - ref[0]) / ref[0]; e5 = abs(lam[4] - ref[4]) / ref[4]
        print(f"{s:<6} {n:4d} {dof:5d}  {t:9.3e}   " + "  ".join(f"{c:11d}" for c in nc)
              + f"   {e1:.2e}   {e5:.2e}")

    print("\nMatched accuracy: cheapest resolution resolving >= 5 modes")
    print("tol      solver  res  DOF   time[s]    converged modes   time per converged mode [s]")
    for tol in TOLS:
        for s in ("FEM", "P64", "P34"):
            cand = [(t, n, dof, nconv(lam, ref, tol)) for (ss, n, dof, t, lam) in rows if ss == s]
            cand = [c for c in cand if c[3] >= 5]
            if not cand:
                print(f"{tol:.0e}   {s:<6}   --- not reached at any resolution tested ---")
                continue
            t, n, dof, k = min(cand)
            print(f"{tol:.0e}   {s:<6} {n:4d} {dof:5d}  {t:9.3e}   {k:8d}            {t/k:.3e}")
