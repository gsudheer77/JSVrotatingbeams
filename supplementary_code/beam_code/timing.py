"""
timing.py
==========================================================================
Section 4.13: full timing and memory tables, and fitted scaling
exponents, for the multiprecision trigonometric-collocation solver.

Measures, for beam 1 (moderate taper) and beam 2 (severe taper), at a
sequence of Chebyshev degrees N:
  - pencil assembly time (build_pencil), 34-digit multiprecision
  - full O(N^3) reduction + eigensolve time, 34-digit multiprecision
  - total wall-clock solve time, 34-digit multiprecision
  - peak memory during the full solve (tracemalloc), 34-digit
  - the same reduced-eigenproblem solve in double precision, for the
    "multiprecision vs. double precision" cost ratio quoted in Section
    4.13.

Also fits a power-law scaling exponent (least-squares slope of
log(time) vs log(N)) for the assembly and total-solve timings, over the
range of N tabulated.

Timings are single-threaded (no manual multithreading is invoked, but
BLAS-level threading in the double-precision comparison is not pinned;
see the caveat below) and are, as stated in the manuscript, hardware-
and software-stack dependent -- absolute seconds will differ from the
paper's reported 2.80 GHz-core numbers, but the *scaling* (the fitted
exponents) is expected to reproduce.

Run: python3 timing.py   (the N=80, beam-2 row takes several minutes)
==========================================================================
"""
import time
import tracemalloc
import numpy as np
import mpmath as mp
from chebyshev_rotating_beam import build_pencil, solve, wang_wereley

mp.mp.dps = 34


def time_once(N, Gamma, mu, P, nmodes=5):
    """Return (assembly_s, reduce_eigensolve_s, total_s, peak_kb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    A, B = build_pencil(N, Gamma, mu, P)
    t1 = time.perf_counter()

    n1 = A.rows
    m = n1 - 4
    cons = [0, 1, n1 - 2, n1 - 1]
    interior = [r for r in range(n1) if r not in cons]
    elim_cols = [0, 1, n1 - 2, n1 - 1]
    keep_cols = [c for c in range(n1) if c not in elim_cols]
    C2 = mp.zeros(4, 4)
    C1 = mp.zeros(4, m)
    for i, r in enumerate(cons):
        for j, ck in enumerate(keep_cols):
            C1[i, j] = A[r, ck]
        for p, ce in enumerate(elim_cols):
            C2[i, p] = A[r, ce]
    Xm = mp.inverse(C2) * C1
    L, R = mp.zeros(m), mp.zeros(m)
    for i, r in enumerate(interior):
        for j, ck in enumerate(keep_cols):
            aL, aR = A[r, ck], B[r, ck]
            for p, ce in enumerate(elim_cols):
                aL -= A[r, ce] * Xm[p, j]
                aR -= B[r, ce] * Xm[p, j]
            L[i, j], R[i, j] = aL, aR
    E, V = mp.eig(mp.inverse(R) * L)
    t2 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (t1 - t0), (t2 - t1), (t2 - t0), peak / 1024.0


def time_double(N, Gamma, mu, P):
    """Reduced problem solved in double precision (NumPy), for the
    multiprecision-vs-double cost ratio quoted in Section 4.13."""
    mp.mp.dps = 34
    A, B = build_pencil(N, Gamma, mu, P)
    n1 = A.rows
    m = n1 - 4
    cons = [0, 1, n1 - 2, n1 - 1]
    interior = [r for r in range(n1) if r not in cons]
    elim_cols = [0, 1, n1 - 2, n1 - 1]
    keep_cols = [c for c in range(n1) if c not in elim_cols]

    Anp = np.array([[float(A[i, j]) for j in range(n1)] for i in range(n1)])
    Bnp = np.array([[float(B[i, j]) for j in range(n1)] for i in range(n1)])
    C2 = Anp[np.ix_(cons, elim_cols)]
    C1 = Anp[np.ix_(cons, keep_cols)]
    Xm = np.linalg.solve(C2, C1)
    Lm = (Anp[np.ix_(interior, keep_cols)]
          - Anp[np.ix_(interior, elim_cols)] @ Xm)
    Rm = (Bnp[np.ix_(interior, keep_cols)]
          - Bnp[np.ix_(interior, elim_cols)] @ Xm)

    t0 = time.perf_counter()
    np.linalg.eig(np.linalg.solve(Rm, Lm))
    t1 = time.perf_counter()
    return t1 - t0


def fit_rate(Ns, ts):
    logN = np.log(Ns)
    logt = np.log(ts)
    return np.polyfit(logN, logt, 1)[0]


def main():
    mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])   # beam 1
    mu2, P2 = wang_wereley(-0.8, [-0.95])                # beam 2

    print("=" * 78)
    print("Beam 1 (Gamma=0): assembly / reduce+eigensolve / total time (s), peak mem (KB)")
    Ns1 = [15, 20, 25, 30, 40, 50]
    rows1 = []
    for N in Ns1:
        mp.mp.dps = 34
        asm, eig, tot, peak = time_once(N, 0, mu1, P1)
        rows1.append((N, asm, eig, tot, peak))
        print(f"N={N:3d}  assembly={asm:8.4f}  eig={eig:8.4f}  total={tot:8.4f}  peak={peak:9.1f} KB")

    print()
    print("Beam 2 (Gamma=0), severe taper, N=80 (converged reference row):")
    mp.mp.dps = 34
    asm2, eig2, tot2, peak2 = time_once(80, 0, mu2, P2)
    print(f"N=80  assembly={asm2:8.4f}  eig={eig2:8.4f}  total={tot2:8.4f}  peak={peak2:9.1f} KB "
          f"({tot2/60:.2f} min)")

    print()
    print("Multiprecision (34-digit) vs. double-precision cost, reduced eigenproblem, beam 1, N=50:")
    mp.mp.dps = 34
    _, _, tot_mp, _ = time_once(50, 0, mu1, P1)
    t_dbl = time_double(50, 0, mu1, P1)
    print(f"  34-digit total : {tot_mp:.4f} s")
    print(f"  double total   : {t_dbl*1000:.4f} ms")
    print(f"  ratio (mp/dbl) : {tot_mp / t_dbl:.3e}")

    Ns_arr = np.array([r[0] for r in rows1], dtype=float)
    asm_arr = np.array([r[1] for r in rows1], dtype=float)
    tot_arr = np.array([r[3] for r in rows1], dtype=float)
    print()
    print("Fitted scaling exponents (beam 1, N=15..50, least-squares log-log slope):")
    print(f"  assembly time  ~ N^{fit_rate(Ns_arr, asm_arr):.2f}")
    print(f"  total solve    ~ N^{fit_rate(Ns_arr, tot_arr):.2f}")
    print()
    print("Hardware/software: single core, as reported by platform.processor() / "
          "sys.version below; see Appendix B for the mpmath/NumPy/SciPy versions "
          "used for the paper's own timings.")
    import platform, sys
    print(f"  processor: {platform.processor() or platform.machine()}")
    print(f"  python: {sys.version.split()[0]}, mpmath: {mp.__version__}")


if __name__ == "__main__":
    main()
