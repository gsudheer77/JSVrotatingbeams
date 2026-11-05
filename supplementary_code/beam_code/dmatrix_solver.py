"""
dmatrix_solver.py
==========================================================================
Standard nodal Chebyshev differentiation-matrix collocation for the same
rotating-beam problem (Section 4.14 cross-check / Tables 12-13).

Uses the SAME taper-law convention as chebyshev_rotating_beam.py
(mu, P given as ascending-power coefficient lists) and the SAME
well-scaled boundary reduction by default -- here eliminating against
the four boundary NODES {x_0, x_1, x_{N-1}, x_N} rather than the
trigonometric formulation's four Chebyshev COEFFICIENTS
{a_0, a_1, a_{N-1}, a_N}. Both are the well-scaled choice for their own
representation (nodal values vs. coefficients).
==========================================================================
"""
import mpmath as mp
import numpy as np
from chebyshev_rotating_beam import polyder, polyval, _tension, _mpf_list


def cheb_D_mp(N):
    """Classical Chebyshev differentiation matrix on CGL points x_j =
    cos(j*pi/N), j=0..N (negative-sum-trick diagonal), at working
    precision mp.mp.dps."""
    x = [mp.cos(mp.pi * j / N) for j in range(N + 1)]
    n1 = N + 1
    c = [mp.mpf(2)] + [mp.mpf(1)] * (N - 1) + [mp.mpf(2)]
    c = [c[i] * (-1) ** i for i in range(n1)]
    D = mp.zeros(n1, n1)
    for i in range(n1):
        for j in range(n1):
            if i != j:
                D[i, j] = c[i] / (c[j] * (x[i] - x[j]))
    for i in range(n1):
        D[i, i] = -sum(D[i, j] for j in range(n1) if j != i)
    return D, x


def build_pencil_dmatrix(N, Gamma, mu, P, Rbar=0, etaF=0, collocation="tip-border"):
    """Assemble the nodal-representation pencil A W = lambda^2 B W (W the
    vector of nodal values), matching the monic equation and boundary
    conditions of chebyshev_rotating_beam.build_pencil exactly."""
    mu = _mpf_list(mu); P = _mpf_list(P)
    Gamma, Rbar, etaF = mp.mpf(str(Gamma)), mp.mpf(str(Rbar)), mp.mpf(str(etaF))
    Pp, Ppp = polyder(P), polyder(polyder(P))
    Tb, Tbp = _tension(mu, Gamma, Rbar, etaF)

    D, x = cheb_D_mp(N)
    D = 2 * D  # d/dxi = 2 d/dx  (xi = (x+1)/2)
    D2 = D * D
    D3 = D * D2
    D4 = D2 * D2
    n1 = N + 1
    A, B = mp.zeros(n1, n1), mp.zeros(n1, n1)

    # clamped root at xi=0 <-> x=-1 <-> node index N
    for k in range(n1):
        A[0, k] = mp.mpf(1) if k == N else mp.mpf(0)  # W(root)=0
        A[1, k] = D[N, k]                               # W'(root)=0

    # free tip at xi=1 <-> x=+1 <-> node index 0
    P1 = polyval(P, mp.mpf(1))
    for k in range(n1):
        A[n1 - 2, k] = D2[0, k]                          # W''(tip)=0
        A[n1 - 1, k] = P1 * D3[0, k] - etaF * D[0, k]     # shear (etaF term)

    if collocation == "symmetric":
        jlist = list(range(2, N - 1))
    else:  # tip-border (paper default): drop the two interior points nearest the tip
        jlist = list(range(N - 1, 2, -1))  # j = N-1, N-2, ..., 3

    for i, j in enumerate(jlist):
        xi = (x[j] + 1) / 2
        Pv = polyval(P, xi)
        q3 = 2 * polyval(Pp, xi) / Pv
        q2 = (polyval(Ppp, xi) - Tb(xi)) / Pv
        q1 = -Tbp(xi) / Pv
        q0 = polyval(mu, xi) / Pv
        for k in range(n1):
            A[2 + i, k] = D4[j, k] + q3 * D3[j, k] + q2 * D2[j, k] + q1 * D[j, k]
        B[2 + i, j] = q0

    return A, B


def solve_dmatrix_mp(N, Gamma, mu, P, Rbar=0, etaF=0, nmodes=5):
    """First nmodes frequencies via the nodal D-matrix formulation, at
    working precision mp.mp.dps, well-scaled reduction (boundary nodes)."""
    A, B = build_pencil_dmatrix(N, Gamma, mu, P, Rbar, etaF)
    n1 = A.rows
    m = n1 - 4
    cons = [0, 1, n1 - 2, n1 - 1]
    interior = [r for r in range(n1) if r not in cons]
    elim = [0, 1, n1 - 2, n1 - 1]
    keep = [c for c in range(n1) if c not in elim]

    C2 = mp.zeros(4, 4)
    C1 = mp.zeros(4, m)
    for i, r in enumerate(cons):
        for j, ck in enumerate(keep):
            C1[i, j] = A[r, ck]
        for p, ce in enumerate(elim):
            C2[i, p] = A[r, ce]
    Xm = mp.inverse(C2) * C1

    L, R = mp.zeros(m), mp.zeros(m)
    for i, r in enumerate(interior):
        for j, ck in enumerate(keep):
            aL, aR = A[r, ck], B[r, ck]
            for p, ce in enumerate(elim):
                aL -= A[r, ce] * Xm[p, j]
                aR -= B[r, ce] * Xm[p, j]
            L[i, j], R[i, j] = aL, aR

    E, V = mp.eig(mp.inverse(R) * L)
    kp = [i for i in range(len(E))
          if abs(mp.im(E[i])) < mp.mpf('1e-15') * (1 + abs(mp.re(E[i])))
          and mp.re(E[i]) > 0]
    kp.sort(key=lambda i: mp.re(E[i]))
    kp = kp[:nmodes]
    return [mp.sqrt(mp.re(E[i])) for i in kp]


def solve_trig_double(N, Gamma, mu, P, Rbar=0, etaF=0, nmodes=5):
    """Double-precision run of the trigonometric (coefficient-space)
    formulation (mp.mp.dps=16), well-scaled reduction."""
    import chebyshev_rotating_beam as cb
    old = mp.mp.dps
    mp.mp.dps = 16
    try:
        freqs = cb.solve(N, Gamma, mu, P, Rbar, etaF, nmodes)
    finally:
        mp.mp.dps = old
    return freqs


def solve_dmatrix_double(N, Gamma, mu, P, Rbar=0, etaF=0, nmodes=5):
    """Double-precision run of the nodal D-matrix formulation using
    genuine IEEE-754 double arithmetic (NumPy/SciPy)."""
    from scipy.linalg import eig
    old = mp.mp.dps
    mp.mp.dps = 34
    A, B = build_pencil_dmatrix(N, Gamma, mu, P, Rbar, etaF)
    mp.mp.dps = old
    n1 = A.rows
    An = np.array([[float(A[i, j]) for j in range(n1)] for i in range(n1)])
    Bn = np.array([[float(B[i, j]) for j in range(n1)] for i in range(n1)])
    cons = [0, 1, n1 - 2, n1 - 1]
    interior = [r for r in range(n1) if r not in cons]
    elim = [0, 1, n1 - 2, n1 - 1]
    keep = [c for c in range(n1) if c not in elim]
    C2 = An[np.ix_(cons, elim)]
    C1 = An[np.ix_(cons, keep)]
    X = np.linalg.solve(C2, C1)
    L = An[np.ix_(interior, keep)] - An[np.ix_(interior, elim)] @ X
    R = Bn[np.ix_(interior, keep)] - Bn[np.ix_(interior, elim)] @ X
    w = eig(L, R, right=False)
    w = w[np.isfinite(w)]
    w = w[(np.abs(w.imag) < 1e-8 * (1 + np.abs(w.real))) & (w.real > 0)]
    return np.sort(np.sqrt(w.real))[:nmodes]
