"""
fem_hermite_solver.py
==========================================================================
Independent Hermite-cubic (Euler-Bernoulli) finite element solver for the
rotating tapered beam, used for the mode-shape and error-per-DOF
comparisons of Sections 4.12 and 4.15. Standard 2-node, 4-DOF-per-element
cubic Hermite beam element (w, theta at each node); mass, bending
stiffness, and centrifugal (geometric) stiffness matrices are assembled
with 6-point Gauss-Legendre quadrature per element, matching the
manuscript's stated FEM formulation (Appendix B / Section 4.14). Runs in
double precision (NumPy/SciPy) -- this is the "unscaled double-precision
Hermite implementation" whose error floor near 1e-8-1e-9 the manuscript
explicitly attributes to this particular assembly, not to finite
elements in general (Section 4.15).

Non-dimensional element quantities (xi = y/L in [0,1], element length
h = 1/n_el):
  Local cubic Hermite shape functions on s in [0,1] (s = local xi/h):
    N1 = 1-3s^2+2s^3,       N2 = h(s-2s^2+s^3)
    N3 = 3s^2-2s^3,         N4 = h(-s^2+s^3)
  K_e[i,j] = Int_0^h  P(xi) N_i''(xi) N_j''(xi) dxi   (bending)
  Kg_e[i,j] = Int_0^h Tbar(xi) N_i'(xi) N_j'(xi) dxi   (centrifugal, geometric stiffness)
  M_e[i,j] = Int_0^h  mu(xi) N_i(xi) N_j(xi) dxi       (consistent mass)
Global assembly is the standard overlap-by-shared-node sum; clamped root
removes (w_0, theta_0); free tip keeps all remaining DOFs (natural BCs).
==========================================================================
"""
import numpy as np


def _gauss6():
    """6-point Gauss-Legendre nodes/weights on [0,1]."""
    x, w = np.polynomial.legendre.leggauss(6)
    return 0.5 * (x + 1), 0.5 * w


def _hermite_shape(s, h):
    """N (4,), N' (4,) w.r.t. local physical coordinate, N'' (4,), at
    local parametric coordinate s in [0,1], element length h."""
    N = np.array([1 - 3*s**2 + 2*s**3,
                  h*(s - 2*s**2 + s**3),
                  3*s**2 - 2*s**3,
                  h*(-s**2 + s**3)])
    dN = np.array([(-6*s + 6*s**2) / h,
                   (1 - 4*s + 3*s**2),
                   (6*s - 6*s**2) / h,
                   (-2*s + 3*s**2)])
    d2N = np.array([(-6 + 12*s) / h**2,
                    (-4 + 6*s) / h,
                    (6 - 12*s) / h**2,
                    (-2 + 6*s) / h])
    return N, dN, d2N


def _tension_fn(mu_coef, Gamma, Rbar, etaF):
    """Tbar(xi) = Gamma^2 * Int_xi^1 mu(s)(Rbar+s) ds + etaF, via exact
    polynomial integration (mu is a short polynomial)."""
    from numpy.polynomial import polynomial as npoly
    mu_shift = npoly.polymul(mu_coef, [Rbar, 1.0])  # mu(s)*(Rbar+s)
    integ = npoly.polyint(mu_shift)  # antiderivative, integ(0)=0

    def Tb(xi):
        total = npoly.polyval(1.0, integ) - npoly.polyval(xi, integ)
        return Gamma**2 * total + etaF
    return Tb


def solve_fem(n_el, Gamma, mu_coef, P_coef, Rbar=0.0, etaF=0.0, nmodes=5):
    """First `nmodes` non-dimensional frequencies via cubic Hermite FEM.

    mu_coef, P_coef : ascending-power coefficient lists/arrays (as used
        by chebyshev_rotating_beam.wang_wereley / banerjee), evaluated
        with numpy.polynomial.polynomial.polyval.
    """
    from numpy.polynomial import polynomial as npoly
    n_nodes = n_el + 1
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    h = 1.0 / n_el
    gx, gw = _gauss6()
    Tb = _tension_fn(np.asarray(mu_coef, dtype=float), Gamma, Rbar, etaF)

    for e in range(n_el):
        x0 = e * h
        Ke = np.zeros((4, 4))
        Me = np.zeros((4, 4))
        Kge = np.zeros((4, 4))
        for s, w in zip(gx, gw):
            xi = x0 + s * h
            N, dN, d2N = _hermite_shape(s, h)
            Pv = npoly.polyval(xi, P_coef)
            muv = npoly.polyval(xi, mu_coef)
            Tv = Tb(xi)
            jac = h  # dxi = h ds
            Ke += Pv * np.outer(d2N, d2N) * w * jac
            Kge += Tv * np.outer(dN, dN) * w * jac
            Me += muv * np.outer(N, N) * w * jac
        dof = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
        for a in range(4):
            for b in range(4):
                K[dof[a], dof[b]] += Ke[a, b] + Kge[a, b]
                M[dof[a], dof[b]] += Me[a, b]

    # clamped root: remove w_0, theta_0 (dof 0, 1); free tip: natural BCs
    free = list(range(2, ndof))
    Kff = K[np.ix_(free, free)]
    Mff = M[np.ix_(free, free)]

    from scipy.linalg import eigh
    w2, V = eigh(Kff, Mff)
    w2 = np.clip(w2, 0, None)
    freqs = np.sqrt(w2)[:nmodes]

    vecs = []
    for i in range(nmodes):
        full = np.zeros(ndof)
        full[free] = V[:, i]
        vecs.append(full)
    return freqs, vecs, free


def fem_mode_shape(vec, n_el, xi_eval, order=0):
    """Evaluate the FEM mode shape at physical points xi_eval (array in
    [0,1]) from the global DOF vector `vec` (length 2*(n_el+1):
    w_0,theta_0,w_1,theta_1,...). order=0: deflection W; order=1: slope
    W'; order=2: curvature W''."""
    h = 1.0 / n_el
    xi_eval = np.asarray(xi_eval, dtype=float)
    out = np.zeros_like(xi_eval)
    for idx, xi in enumerate(xi_eval):
        e = min(int(xi / h), n_el - 1)
        s = (xi - e * h) / h
        N, dN, d2N = _hermite_shape(s, h)
        shape = (N, dN, d2N)[order]
        dof = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
        out[idx] = shape @ vec[dof]
    return out


if __name__ == "__main__":
    # self-test: beam 1 (Wang-Wereley), Gamma=0, converged FEM frequencies
    mu_coef = [1.0, -0.5]
    P_coef = [1.0, -1.5, 0.75, -0.125]
    for n_el in (10, 20, 40, 80):
        f, _, _ = solve_fem(n_el, 0.0, mu_coef, P_coef, nmodes=3)
        print(f"n_el={n_el:4d}  {f}")
