"""
per_case_verification.py
==========================================================================
Section 4.16 supplementary per-case verification diagnostics for a
benchmark case: N-refinement relative change, algebraic backward error,
dense-grid continuous residual, boundary-condition residual, Rayleigh-
quotient discrepancy, cross-resolution MAC, and mass-orthogonality error.

All diagnostics are computed purely from chebyshev_rotating_beam.py (no
FEM needed) using the reconstructed Chebyshev-coefficient eigenvectors.

Quadrature note: the Rayleigh-quotient discrepancy and mass-orthogonality
error are integrals of a mode shape (a degree-N Chebyshev series) against
polynomial weights (mu, P, Tbar), so the integrand is itself a polynomial
of bounded degree (up to roughly degree 100-110 at N=50, depending on the
taper-law degree). A single fixed-order Gauss-Legendre rule is used
rather than adaptive refinement; the order below (_GL_DEGREE=6, a
96-point rule, exact through degree 191) is chosen with a comfortable
margin above that estimated integrand degree, so the quadrature error is
negligible relative to the reported continuum-diagnostic floor for the
benchmark cases considered here -- see _gauss_legendre_rule / _q below.
==========================================================================
"""
import mpmath as mp
import chebyshev_rotating_beam as cb
from mpmath.calculus.quadrature import GaussLegendre

_GL_DEGREE = 6  # 96-point fixed Gauss-Legendre rule, exact through
                # degree 191 -- comfortably above the ~100-110 degree
                # integrands arising here for N up to 50.
_gl_cache = {}


def _gauss_legendre_rule(a=0, b=1, degree=_GL_DEGREE):
    key = (a, b, degree, mp.mp.dps)
    if key not in _gl_cache:
        gl = GaussLegendre(mp.mp)
        _gl_cache[key] = gl.get_nodes(mp.mpf(a), mp.mpf(b), degree=degree,
                                       prec=mp.mp.prec)
    return _gl_cache[key]


def _q(f, a=0, b=1):
    """Integral of f over [a,b] via a fixed high-order Gauss-Legendre
    rule -- see module docstring."""
    nodes = _gauss_legendre_rule(a, b)
    return sum(w * f(x) for x, w in nodes)


def _Tk_endpoint(N, sign, max_order=3):
    """Exact T_k^(p)(+-1), p=0..max_order, for k=0..N, via the closed-form
    product T_k^(p)(1) = prod_{m=0}^{p-1}(k^2-m^2)/(2p-1)!! -- avoids the
    sin(theta)->0 singularity of the interior trig_derivs formula at the
    endpoints. max_order=3 covers the boundary-condition rows (W,W',W'',
    W'''); max_order=4 additionally gives the closed-form W'''', needed to
    evaluate the governing-equation residual exactly at xi=0,1 (see
    mode_derivs below)."""
    rows = [[mp.mpf(0)] * (N + 1) for _ in range(max_order + 1)]
    for k in range(N + 1):
        kk = mp.mpf(k)
        k2 = kk * kk
        rows[0][k] = mp.mpf(1)
        prod = mp.mpf(1)
        dfact = mp.mpf(1)
        for p in range(1, max_order + 1):
            prod *= (k2 - (p - 1) ** 2)
            dfact *= (2 * p - 1)
            rows[p][k] = prod / dfact
    if sign < 0:
        for p in range(max_order + 1):
            for k in range(N + 1):
                rows[p][k] *= (-1) ** (k + p)
    return rows


def endpoint_derivs(a, sign):
    """W, W', W'', W''' at xi=0 (sign=-1) or xi=1 (sign=+1), chain-ruled
    (d/dxi = 2 d/dx)."""
    N = len(a) - 1
    d = _Tk_endpoint(N, sign, max_order=3)
    scale = [1, 2, 4, 8]
    return [sum(a[k] * scale[m] * d[m][k] for k in range(N + 1)) for m in range(4)]


def mode_derivs(a, xi):
    """W, W', W'', W''', W'''' at physical coordinate xi, from Chebyshev
    coefficient vector a (length N+1). Uses the same closed forms as
    build_pencil (trig_derivs) in the open interval, chain-ruled to xi.

    At the exact endpoints xi=0,1 the trig_derivs formula is genuinely
    singular (it involves 1/sin(theta)^7 for the fourth derivative), so a
    naive small epsilon offset there returns numerical garbage rather than
    the true (finite) boundary limit. Instead, at (or numerically at) the
    endpoints this dispatches to the exact closed-form endpoint derivative
    (_Tk_endpoint, extended here to fourth order), which has no such
    singularity."""
    N = len(a) - 1
    xi = mp.mpf(xi)
    if xi <= 0 or xi >= 1:
        sign = -1 if xi <= 0 else 1
        d = _Tk_endpoint(N, sign, max_order=4)
        scale = [1, 2, 4, 8, 16]
        return [sum(a[k] * scale[m] * d[m][k] for k in range(N + 1)) for m in range(5)]
    x = 2 * xi - 1
    th = mp.acos(x)
    d = cb.trig_derivs(N, th)
    scale = [1, 2, 4, 8, 16]
    return [sum(a[k] * scale[m] * d[m][k] for k in range(N + 1)) for m in range(5)]


def continuous_residual(a, lam2, mu, P, Gamma, Rbar, etaF, ngrid=801):
    """Maximum-norm dense-grid residual of the governing equation in its
    conservative form, evaluated *before* division by P(xi) (unlike the
    monic q3,q2,q1,q0 form used in build_pencil):

        P W'''' + 2P' W''' + (P'' - Tbar) W'' - Tbar' W' - lambda^2 mu W

    computed over ngrid equispaced points on the closed interval [0,1]
    (Section 4.16; default 801 points, matching the manuscript's stated
    diagnostic), and reported as the maximum absolute residual over the
    grid, normalized by the maximum absolute value of the dominant
    P*W'''' term over the same grid.

    The two grid endpoints xi=0,1 are evaluated via mode_derivs' exact
    closed-form endpoint branch (see above), not the interior trig
    formula, which is genuinely singular there."""
    Pp, Ppp = cb.polyder(P), cb.polyder(cb.polyder(P))
    Tb, Tbp = cb._tension(mu, Gamma, Rbar, etaF)
    grid = [mp.mpf(i) / (ngrid - 1) for i in range(ngrid)]
    res, dom = [], []
    for xi in grid:
        W, Wp, Wpp, Wppp, Wpppp = mode_derivs(a, xi)
        Pv = cb.polyval(P, xi)
        Ppv = cb.polyval(Pp, xi)
        Pppv = cb.polyval(Ppp, xi)
        muv = cb.polyval(mu, xi)
        r = (Pv * Wpppp + 2 * Ppv * Wppp + (Pppv - Tb(xi)) * Wpp
             - Tbp(xi) * Wp - lam2 * muv * W)
        res.append(abs(r))
        dom.append(abs(Pv * Wpppp))
    scale = max(dom) + mp.mpf('1e-300')
    return max(res) / scale


def boundary_residual(a, mu, P, Gamma, Rbar, etaF):
    W0, Wp0, _, _ = endpoint_derivs(a, -1)          # root, xi=0
    _, Wp1, Wpp1, Wppp1 = endpoint_derivs(a, +1)     # tip,  xi=1
    P1 = cb.polyval(P, mp.mpf(1))
    shear = P1 * Wppp1 - etaF * Wp1
    return max(abs(W0), abs(Wp0), abs(Wpp1), abs(shear))


def rayleigh_discrepancy(a, lam, mu, P, Gamma, Rbar, etaF):
    Tb, _ = cb._tension(mu, Gamma, Rbar, etaF)

    def num_integrand(xi):
        W, Wp, Wpp, _, _ = mode_derivs(a, xi)
        Pv = cb.polyval(P, xi)
        return Pv * Wpp ** 2 + Tb(xi) * Wp ** 2

    def den_integrand(xi):
        W, _, _, _, _ = mode_derivs(a, xi)
        return cb.polyval(mu, xi) * W ** 2

    num = _q(num_integrand)
    den = _q(den_integrand)
    lam_rq = mp.sqrt(num / den)
    return abs(lam_rq - lam) / lam


def mac(a1, a2):
    def w1w2(xi):
        return mode_derivs(a1, xi)[0] * mode_derivs(a2, xi)[0]
    def w1w1(xi):
        return mode_derivs(a1, xi)[0] ** 2
    def w2w2(xi):
        return mode_derivs(a2, xi)[0] ** 2
    num = _q(w1w2) ** 2
    den = _q(w1w1) * _q(w2w2)
    return num / den


def mass_orthogonality(vecs, mu, P):
    n = len(vecs)
    worst = mp.mpf(0)
    norms = []
    for a in vecs:
        def integ(xi, a=a):
            W = mode_derivs(a, xi)[0]
            return cb.polyval(mu, xi) * W ** 2
        norms.append(_q(integ))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            def cross(xi, i=i, j=j):
                Wi = mode_derivs(vecs[i], xi)[0]
                Wj = mode_derivs(vecs[j], xi)[0]
                return cb.polyval(mu, xi) * Wi * Wj
            g = _q(cross)
            gnorm = abs(g) / mp.sqrt(norms[i] * norms[j])
            worst = max(worst, gnorm)
    return worst


def certification_table(N, N2, Gamma, mu, P, Rbar=0, etaF=0, nmodes=5):
    freqs, vecs = cb.solve(N, Gamma, mu, P, Rbar, etaF, nmodes, return_vectors=True)
    freqs2 = cb.solve(N2, Gamma, mu, P, Rbar, etaF, nmodes)
    A, B = cb.build_pencil(N, Gamma, mu, P, Rbar, etaF)
    normA = max(abs(A[i, j]) for i in range(A.rows) for j in range(A.cols))
    normB = max(abs(B[i, j]) for i in range(B.rows) for j in range(B.cols))

    freqs2_N2, vecs2 = cb.solve(N2, Gamma, mu, P, Rbar, etaF, nmodes, return_vectors=True)

    rows = []
    for i in range(nmodes):
        lam, a = freqs[i], vecs[i]
        rel_change = abs(freqs2[i] - lam) / freqs2[i]

        v = mp.matrix(a)
        r = A * v - lam ** 2 * (B * v)
        eta = mp.norm(r, 2) / ((normA + lam ** 2 * normB) * mp.norm(v, 2))

        cont = continuous_residual(a, lam ** 2, mu, P, Gamma, Rbar, etaF)
        bres = boundary_residual(a, mu, P, Gamma, Rbar, etaF)
        rq = rayleigh_discrepancy(a, lam, mu, P, Gamma, Rbar, etaF)
        macv = abs(1 - mac(a, vecs2[i]))

        rows.append(dict(mode=i + 1, rel_change=rel_change, backward_error=eta,
                          cont_residual=cont, boundary_residual=bres,
                          rayleigh=rq, one_minus_mac=macv))
    morth = mass_orthogonality(vecs, mu, P)
    return rows, morth


if __name__ == "__main__":
    mp.mp.dps = 34
    mu, P = cb.wang_wereley(-0.5, [-1.5, 0.75, -0.125])  # beam 1
    rows, morth = certification_table(50, 60, 0, mu, P, nmodes=5)
    print("Certification table: beam 1, Rbar=0, Gamma=0, N=50 (N2=60)")
    for r in rows:
        print(f"mode {r['mode']}: relchange={mp.nstr(r['rel_change'],4)} "
              f"eta={mp.nstr(r['backward_error'],4)} "
              f"cont_res={mp.nstr(r['cont_residual'],4)} "
              f"bnd_res={mp.nstr(r['boundary_residual'],4)} "
              f"rayleigh={mp.nstr(r['rayleigh'],4)} "
              f"1-MAC={mp.nstr(r['one_minus_mac'],4)}")
    print("mass-orthogonality max|G_ij| (i!=j):", mp.nstr(morth, 4))
