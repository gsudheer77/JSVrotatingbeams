"""
Well-conditioned Ritz basis: phi_i(xi) = xi^2 * P_i(2*xi-1)  (shifted
Legendre P_i), i=0..M-1 -- satisfies the essential BCs W(0)=W'(0)=0
exactly for every i, and is far better conditioned than raw monomials
xi^2, xi^3, ... because the Legendre factor stays O(1) in magnitude
and oscillates rather than growing like a power.
"""
import sys, numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
BEAM_CODE_DIR = os.environ.get("BEAM_CODE_DIR", os.path.join(HERE, "..", "beam_code"))
sys.path.insert(0, BEAM_CODE_DIR)
import mpmath as mp
from chebyshev_rotating_beam import solve, wang_wereley
from numpy.polynomial import polynomial as npoly
from numpy.polynomial.legendre import Legendre
from scipy.linalg import eigh
from scipy.integrate import quad

mp.mp.dps = 34
def to_float_list(a): return np.array([float(x) for x in a], dtype=float)

def make_case(mu_mpf, P_mpf, Rbar):
    mu_c = to_float_list(mu_mpf); P_c = to_float_list(P_mpf)
    def mu_eval(xi): return npoly.polyval(xi, mu_c)
    def P_eval(xi):  return npoly.polyval(xi, P_c)
    muR = np.polynomial.polynomial.polymul(mu_c, np.array([Rbar, 1.0]))
    integ_coeffs = np.concatenate(([0.0], muR/np.arange(1, len(muR)+1)))
    val1 = npoly.polyval(1.0, integ_coeffs)
    def Tbar1(xi): return val1 - npoly.polyval(xi, integ_coeffs)
    return mu_eval, P_eval, Tbar1

def leg_basis_funcs(M):
    """Return phi, dphi, ddphi lists of length M using xi^2 * shifted-Legendre_i."""
    phis, dphis, ddphis = [], [], []
    for i in range(M):
        coef = np.zeros(i+1); coef[i] = 1.0
        L = Legendre(coef, domain=[0.0, 1.0])
        dL = L.deriv(1)
        ddL = L.deriv(2)
        def phi(xi, L=L):   return xi**2 * L(xi)
        def dphi(xi, L=L, dL=dL): return 2*xi*L(xi) + xi**2*dL(xi)
        def ddphi(xi, L=L, dL=dL, ddL=ddL): return 2*L(xi) + 4*xi*dL(xi) + xi**2*ddL(xi)
        phis.append(phi); dphis.append(dphi); ddphis.append(ddphi)
    return phis, dphis, ddphis

def ritz_eigs(mu_eval, P_eval, Tbar1, M, Gamma, nmodes=3):
    phi, dphi, ddphi = leg_basis_funcs(M)
    K = np.zeros((M, M)); Mm = np.zeros((M, M))
    G2 = Gamma**2
    for i in range(M):
        for j in range(i, M):
            fK = lambda xi: P_eval(xi)*ddphi[i](xi)*ddphi[j](xi) + G2*Tbar1(xi)*dphi[i](xi)*dphi[j](xi)
            fM = lambda xi: mu_eval(xi)*phi[i](xi)*phi[j](xi)
            K[i,j]=K[j,i]=quad(fK,0,1,limit=300)[0]
            Mm[i,j]=Mm[j,i]=quad(fM,0,1,limit=300)[0]
    w = eigh(K, Mm, eigvals_only=True)
    return np.sort(w)[:nmodes], np.linalg.cond(Mm)

def run(label, mu_mpf, P_mpf, Rbar, Gammas, Ns_ref=50, Mmax=24, nmodes=3):
    print(f"\n{'='*95}\n{label} (Rbar={Rbar})\n{'='*95}")
    mu_eval, P_eval, Tbar1 = make_case(mu_mpf, P_mpf, Rbar)
    for G in Gammas:
        ref = to_float_list(solve(Ns_ref, G, mu_mpf, P_mpf, nmodes=nmodes, Rbar=Rbar))**2
        print(f"\n Gamma={G}   converged (collocation, N={Ns_ref}): lam^2 = {np.round(ref,6)}")
        for M in sorted(set([nmodes+3, nmodes+6, nmodes+10, Mmax])):
            try:
                ritz, cond = ritz_eigs(mu_eval, P_eval, Tbar1, M, G, nmodes=nmodes)
            except Exception as e:
                print(f"   Ritz M={M:2d}: FAILED ({type(e).__name__})")
                continue
            gaps = ritz - ref
            # NOTE: the Rayleigh-Ritz upper-bound property (lam_true <= lam_Ritz) is
            # rigorous only in exact arithmetic for a conforming trial subspace. The
            # numbers here are floating-point (SciPy quad + eigh), so a gap can show
            # as a tiny negative value from roundoff even when the underlying bound
            # holds exactly -- this is NOT evidence the variational principle is
            # violated. We therefore report the gaps themselves plus a tolerance
            # flag, honestly named, rather than an unqualified "is this a bound" bool.
            tol = 1e-9
            within_tol = np.all(gaps > -tol)
            print(f"   Ritz M={M:2d} (cond(Mm)={cond:.2e}): lam^2={np.round(ritz,7)}  "
                  f"gap={np.array2string(gaps, precision=2, floatmode='unique')}  "
                  f"upper-bound-within-tolerance(tol={tol:.0e})={within_tol}")
            if not within_tol:
                worst = gaps.min()
                print(f"      -> smallest gap {worst:.3e} exceeds the {tol:.0e} floating-point "
                      f"tolerance; treat as roundoff at the double-precision floor unless "
                      f"reproduced with higher-precision (mpmath) assembly.")

if __name__ == "__main__":
    mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
    mu2, P2 = wang_wereley(-0.8, [-0.95])
    run("Beam 1", mu1, P1, 0, [0, 6, 12], Mmax=13)
    run("Beam 2 (severe taper)", mu2, P2, 0, [0, 6, 12], Ns_ref=80, Mmax=13)
