"""
chebyshev_rotating_beam.py
==========================================================================
Multiprecision trigonometric Chebyshev-collocation solver for the flapwise
free vibration of rotating tapered Euler-Bernoulli beams.

Companion code for:
  "Modal Reorganization of Rotating Tapered Beams under Centrifugal
   Stiffening: Physical Insights from High-Accuracy Spectral
   Computation."

Governing (non-dimensional, monic) equation on xi in [0,1]:

    W'''' + q3 W''' + q2 W'' + q1 W'  =  lambda^2 q0 W ,

with  q3 = 2P'/P ,  q2 = (P'' - Tbar)/P ,  q1 = -Tbar'/P ,  q0 = mu/P ,
and centrifugal tension
    Tbar(xi)  = Gamma^2 * Integral_xi^1 mu(s)(Rbar+s) ds  + etaF ,
    Tbar'(xi) = -Gamma^2 * mu(xi)(Rbar+xi) .

mu(xi) = m(xi)/m0 and P(xi) = EI(xi)/EI0 are polynomials given as ascending
power-coefficient lists.  The mode shape is expanded in Chebyshev polynomials
of the first kind, T_k(x)=cos(k*arccos x), x = 2*xi-1, and the equation is
collocated at a subset of interior Chebyshev-Gauss-Lobatto points.  All
derivatives are evaluated analytically in theta = arccos x; no differentiation
matrices are formed.

Clamped root at xi=0 (x=-1):   W = W' = 0
Free tip at   xi=1 (x=+1):     W'' = 0 ,  P(1) W''' = 0.

Scope note on etaF (non-zero outboard tip force): the released solver
implements only the F_t = 0 benchmark problem used throughout the
manuscript. build_pencil()/solve() accept an etaF argument for the
tension Tbar(xi), and the governing equation and root/W''=0 tip rows
are correct for any etaF, but the released fourth boundary row (shear)
imposes the etaF=0 condition, W'''(1)=0 -- the general
trigonometric shear row P(1) W''' - etaF W' = 0 is NOT implemented
(see the comment at that row in build_pencil). Accordingly, passing
etaF != 0 raises NotImplementedError rather than silently solving the
wrong (etaF=0) problem: the general non-zero end-force shear boundary
condition is outside the scope of the numerical results reported in
the manuscript.

Requires: mpmath.  Set precision with mp.mp.dps (default 34 digits here).
==========================================================================
"""

import mpmath as mp

mp.mp.dps = 34  # working precision (significant decimal digits)

# --------------------------------------------------------------------------
# Polynomial helpers (ascending-power coefficient lists of mpf)
# --------------------------------------------------------------------------

def polymul(a, b):
    r = [mp.mpf(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            r[i + j] += ai * bj
    return r

def polyval(a, x):
    s = mp.mpf(0)
    for c in reversed(a):
        s = s * x + c
    return s

def polyder(a):
    return [a[i] * i for i in range(1, len(a))] if len(a) > 1 else [mp.mpf(0)]

def polyint_def(a, lo, hi):
    """Definite integral of polynomial a from lo to hi."""
    A = [mp.mpf(0)] + [a[i] / (i + 1) for i in range(len(a))]
    return polyval(A, hi) - polyval(A, lo)

def _mpf_list(v):
    return [mp.mpf(str(x)) for x in v]

# --------------------------------------------------------------------------
# Analytic derivatives of T_k in the angular variable (Appendix A)
# --------------------------------------------------------------------------

def trig_derivs(N, th):
    """Return [d0,d1,d2,d3,d4], each a length-(N+1) list of T_k^(m)(x) at
    x = cos(th), for m = 0..4, using the closed forms of Appendix A."""
    s, c = mp.sin(th), mp.cos(th)
    s3, s5, s7 = s**3, s**5, s**7
    rows = [[mp.mpf(0)] * (N + 1) for _ in range(5)]
    for k in range(N + 1):
        kk = mp.mpf(k)
        sk, ck = mp.sin(kk * th), mp.cos(kk * th)
        rows[0][k] = ck
        rows[1][k] = kk * sk / s
        rows[2][k] = (kk * sk * c - kk**2 * ck * s) / s3
        rows[3][k] = -kk / s5 * ((kk**2 - 1) * s**2 * sk
                                 + 3 * c * (kk * s * ck - c * sk))
        rows[4][k] = kk / s7 * (3 * (5 - 2 * (kk**2 + 1) * s**2) * c * sk
                                + kk * ((kk**2 + 11) * s**2 - 15) * s * ck)
    return rows

# --------------------------------------------------------------------------
# Pencil assembly
# --------------------------------------------------------------------------

def _tension(mu, Gamma, Rbar, etaF):
    """Return closures Tbar(xi), Tbar'(xi) for the non-dimensional tension."""
    G2 = Gamma**2
    integ = polymul(mu, [Rbar, mp.mpf(1)])          # mu(s)*(Rbar+s)
    def Tb(xi):  return G2 * polyint_def(integ, xi, mp.mpf(1)) + etaF
    def Tbp(xi): return -G2 * polyval(integ, xi)
    return Tb, Tbp

def build_pencil(N, Gamma, mu, P, Rbar=0, etaF=0, collocation="tip-border"):
    """Assemble (A, B) for the clamped-free beam, A a = lambda^2 B a.

    mu, P     : ascending-power coefficient lists for mass and rigidity.
    Gamma     : non-dimensional rotational speed.
    Rbar,etaF : hub radius and outboard axial force (etaF=0 for all paper results).
    collocation : 'tip-border' (paper default, drops the two interior CGL
                  points nearest the tip) or 'symmetric' (drops one at each end).
    """
    if etaF != 0:
        raise NotImplementedError(
            "build_pencil()/solve() implement only the etaF=0 (F_t=0, no "
            "outboard axial force) benchmark problem used throughout the "
            "manuscript. The general non-zero end-force tip-shear row is "
            "not implemented in this released code; see the module "
            "docstring's 'Scope note on etaF'.")
    mu = _mpf_list(mu); P = _mpf_list(P)
    Gamma, Rbar, etaF = mp.mpf(str(Gamma)), mp.mpf(str(Rbar)), mp.mpf(str(etaF))
    Pp, Ppp = polyder(P), polyder(polyder(P))
    Tb, Tbp = _tension(mu, Gamma, Rbar, etaF)
    n1 = N + 1
    A, B = mp.zeros(n1), mp.zeros(n1)

    # boundary rows (exact endpoint derivatives, Appendix A)
    for k in range(n1):
        kk = mp.mpf(k)
        A[0, k]   = (-1)**k                          # W(-1)=0
        A[1, k]   = (-1)**(k + 1) * kk**2            # W'(-1)=0
        A[N-1, k] = (kk**4 - kk**2) / 3              # W''(1)=0
        A[N, k]   = (kk**6 - 5*kk**4 + 4*kk**2) / 15 # W'''(1)=0 (etaF=0 only)
    # NOT IMPLEMENTED: the general trigonometric shear condition
    # P(1)*(2/1)^3 T''' - etaF*(2/1) T' evaluated at x=+1 (Section 3.3,
    # Eq. 5) for etaF != 0. The row above only ever imposes the etaF=0
    # special case; build_pencil() already rejects etaF != 0 above with
    # NotImplementedError, so this row is unreachable for non-zero etaF.
    # See the module docstring's "Scope note on etaF".

    if collocation == "symmetric":
        idx = [(mp.mpf(j) * mp.pi / N, j) for j in range(2, N - 1)]
    else:  # tip-border: theta_t = (N-t)pi/N, t=1..N-3  ==  j=3..N-1
        idx = [(mp.mpf(N - t) * mp.pi / N, N - t) for t in range(1, N - 2)]

    for i, (th, _) in enumerate(idx):
        x = mp.cos(th); xi = (x + 1) / 2
        d = trig_derivs(N, th)
        Pv = polyval(P, xi)
        q3 = 2 * polyval(Pp, xi) / Pv
        q2 = (polyval(Ppp, xi) - Tb(xi)) / Pv
        q1 = -Tbp(xi) / Pv
        q0 = polyval(mu, xi) / Pv
        for k in range(n1):
            # chain-rule factor (2/h)^m with h=1 here => 2^m
            A[2 + i, k] = (16 * d[4][k] + 8 * q3 * d[3][k]
                           + 4 * q2 * d[2][k] + 2 * q1 * d[1][k])
            B[2 + i, k] = q0 * d[0][k]
    return A, B

# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------

def solve(N, Gamma, mu, P, Rbar=0, etaF=0, nmodes=5,
          collocation="tip-border", return_vectors=False, elim="well-scaled"):
    """Return the first `nmodes` non-dimensional frequencies lambda_i.

    If return_vectors=True, also return the full Chebyshev-coefficient
    eigenvectors (length N+1) for mode-shape reconstruction.

    elim : which 4 Chebyshev coefficients the boundary rows are
        eliminated against (Section 4.2 / Appendix B of the manuscript).
          "well-scaled"    (default, used for every reported result):
                            {a_0, a_1, a_{N-1}, a_N} -- keeps the
                            eliminated 4x4 block well conditioned.
          "highest-degree"  (legacy/comparison only -- NOT used for any
                            reported value): {a_{N-3},...,a_N}. The
                            boundary rows' k^2,k^4,k^6 entries are
                            largest in these columns, which concentrates
                            them in the inverted 4x4 block and raises
                            the finite-precision floor of Section 4.2 by
                            several orders of magnitude at fixed
                            precision, with the converged digits
                            unchanged. Retained only to reproduce the
                            Appendix B reduction-choice comparison.
    """
    A, B = build_pencil(N, Gamma, mu, P, Rbar, etaF, collocation)
    n1 = A.rows; m = n1 - 4
    cons = [0, 1, n1 - 2, n1 - 1]
    interior = [r for r in range(n1) if r not in cons]

    if elim == "well-scaled":
        elim_cols = [0, 1, n1 - 2, n1 - 1]
    elif elim == "highest-degree":
        elim_cols = [m, m + 1, m + 2, m + 3]
    else:
        elim_cols = list(elim)
    keep_cols = [c for c in range(n1) if c not in elim_cols]

    C2 = mp.zeros(4, 4); C1 = mp.zeros(4, m)
    for i, r in enumerate(cons):
        for j, ck in enumerate(keep_cols): C1[i, j] = A[r, ck]
        for p, ce in enumerate(elim_cols): C2[i, p] = A[r, ce]
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

    # keep real, positive lambda^2 (reality filter, Section 4.4)
    tol = mp.mpf('1e-15')
    keep = [i for i in range(len(E))
            if abs(mp.im(E[i])) < tol * (1 + abs(mp.re(E[i]))) and mp.re(E[i]) > 0]
    keep.sort(key=lambda i: mp.re(E[i]))
    keep = keep[:nmodes]
    freqs = [mp.sqrt(mp.re(E[i])) for i in keep]
    if not return_vectors:
        return freqs

    vecs = []
    for i in keep:
        vred = [V[r, i] for r in range(m)]
        adep = [-sum(Xm[p, j] * vred[j] for j in range(m)) for p in range(4)]
        full = [mp.mpf(0)] * n1
        for j, ck in enumerate(keep_cols): full[ck] = vred[j]
        for p, ce in enumerate(elim_cols): full[ce] = adep[p]
        vecs.append([mp.re(z) for z in full])
    return freqs, vecs

def mode_value(a, xi):
    """Evaluate W(xi) = sum_k a_k T_k(2xi-1) from coefficient vector a."""
    x = 2 * mp.mpf(xi) - 1
    th = mp.acos(x)
    return sum(a[k] * mp.cos(k * th) for k in range(len(a)))

# --------------------------------------------------------------------------
# Convenience: taper-family coefficient builders
# --------------------------------------------------------------------------

def wang_wereley(alpha, betas):
    """Linear mass 1+alpha*xi, quartic rigidity 1+b1 xi+...+b4 xi^4."""
    mu = [1.0, alpha]
    P = [1.0] + list(betas) + [0.0] * (4 - len(betas))
    return mu, P[:5]

def banerjee(c, n):
    """Power-law class: mu=(1-c xi)^n, P=(1-c xi)^(n+2)."""
    one = [mp.mpf(1), mp.mpf(-c)]
    mu = [mp.mpf(1)]
    for _ in range(n): mu = polymul(mu, one)
    P = [mp.mpf(1)]
    for _ in range(n + 2): P = polymul(P, one)
    return [float(v) for v in mu], [float(v) for v in P]

# --------------------------------------------------------------------------
if __name__ == "__main__":
    # quick self-test: moderate beam (beam 1), Gamma = 0
    mu, P = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
    print("beam 1, Gamma=0, N=25:",
          [mp.nstr(x, 10) for x in solve(25, 0, mu, P, nmodes=3)])
    # severe beam (beam 2), converged
    mu, P = wang_wereley(-0.8, [-0.95])
    print("beam 2, Gamma=0, N=50:",
          [mp.nstr(x, 10) for x in solve(50, 0, mu, P, nmodes=3)])
