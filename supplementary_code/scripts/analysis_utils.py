"""
analysis_utils.py
==========================================================================
Shared helpers for the revision-stage analysis scripts (6_* to 13_*).

* configs()          -- the five taper/hub-radius configurations used for
                        the physical findings (Section 5 of the manuscript).
* cheb_eval(a, xi, m) -- fast double-precision evaluation of the m-th
                        xi-derivative of W(xi) = sum_k a_k T_k(2 xi - 1)
                        from the Chebyshev coefficients returned by
                        chebyshev_rotating_beam.solve(..., return_vectors=True).
* gl(n)              -- Gauss-Legendre nodes/weights on [0,1] (double).
* interior_nodes(a, n_expected) -- interior zeros of a mode shape.

The eigenpairs themselves are always computed by the 34-digit solver;
only post-processing of the converged coefficient vectors (node finding,
energy integrals, coupling coefficients) is done in double precision,
which is far below the accuracy needed for node positions (~1e-14) and
energy ratios (~1e-13).
==========================================================================
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
BEAM_CODE_DIR = os.environ.get("BEAM_CODE_DIR", os.path.join(HERE, "..", "beam_code"))
sys.path.insert(0, BEAM_CODE_DIR)

import numpy as np
from numpy.polynomial import chebyshev as C
import mpmath as mp
import chebyshev_rotating_beam as cb
from chebyshev_rotating_beam import solve, wang_wereley, banerjee

mp.mp.dps = 34


def configs():
    """(key, label, mu, P, Rbar, N) for the five configurations.  N is the
    Chebyshev degree used for the physical-findings computations (beam 2,
    whose rigidity zero lies just outside the domain, needs a higher N)."""
    mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
    mu2, P2 = wang_wereley(-0.8, [-0.95])
    mub2, Pb2 = banerjee(0.2, 1)
    mub6, Pb6 = banerjee(0.6, 2)
    return [
        ("B1",   "Beam 1 (WW moderate taper), R=0",   mu1, P1, 0.0, 40),
        ("B1R",  "Beam 1, hub radius R=0.5",          mu1, P1, 0.5, 40),
        ("B2",   "Beam 2 (WW severe taper), R=0",     mu2, P2, 0.0, 60),
        ("BJ21", "Banerjee c=0.2, n=1, R=0",          mub2, Pb2, 0.0, 40),
        ("BJ62", "Banerjee c=0.6, n=2, R=0",          mub6, Pb6, 0.0, 40),
    ]


def to_float(a):
    return np.array([float(x) for x in a], dtype=float)


def cheb_eval(a, xi, m=0):
    """m-th derivative w.r.t. xi of sum a_k T_k(2xi-1) (double precision)."""
    a = np.asarray(a, dtype=float)
    x = 2.0 * np.asarray(xi, dtype=float) - 1.0
    if m == 0:
        return C.chebval(x, a)
    return (2.0 ** m) * C.chebval(x, C.chebder(a, m))


_GL = {}
def gl(n=160):
    if n not in _GL:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL[n] = (0.5 * (x + 1.0), 0.5 * w)
    return _GL[n]


def poly(coefs, xi):
    return np.polynomial.polynomial.polyval(xi, np.asarray([float(c) for c in coefs]))


def tension_unit(mu, Rbar, xi):
    """T1(xi) = int_xi^1 mu(s)(Rbar+s) ds   (tension at Gamma = 1)."""
    m = np.asarray([float(c) for c in mu])
    integ = np.polynomial.polynomial.polymul(m, [Rbar, 1.0])
    A = np.polynomial.polynomial.polyint(integ)
    return np.polynomial.polynomial.polyval(1.0, A) - np.polynomial.polynomial.polyval(xi, A)


def mass_normalize(a, mu):
    x, w = gl()
    W = cheb_eval(a, x)
    s = np.sqrt(np.sum(w * poly(mu, x) * W * W))
    return np.asarray(a) / s


def interior_nodes(a, n_expected, n_scan=4000):
    """Interior zeros of W on (0,1) by sign-change scan + Newton polish."""
    grid = np.linspace(1e-4, 1 - 1e-6, n_scan)
    v = cheb_eval(a, grid)
    roots = []
    for i in range(len(grid) - 1):
        if v[i] == 0.0 or v[i] * v[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if cheb_eval(a, lo) * cheb_eval(a, mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            r = 0.5 * (lo + hi)
            for _ in range(3):  # Newton polish
                f, fp = cheb_eval(a, r), cheb_eval(a, r, 1)
                if fp != 0:
                    r = r - f / fp
            roots.append(float(r))
    return roots[:n_expected], len(roots)


def solve_modes(N, Gamma, mu, P, Rbar, nmodes):
    """34-digit solve; returns (freqs as mpf list, coefficient vectors as
    float arrays, sign-fixed to positive tip deflection)."""
    f, v = solve(N, Gamma, mu, P, Rbar=Rbar, nmodes=nmodes, return_vectors=True)
    vecs = []
    for a in v:
        af = to_float(a)
        if cheb_eval(af, 1.0) < 0:
            af = -af
        vecs.append(af)
    return f, vecs
