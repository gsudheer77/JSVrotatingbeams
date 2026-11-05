"""
7b_tau_ratio_decomposition.py
Why the leading-order energy ratio (non-rotating shapes) differs from the
exact one (rotating shapes): decomposition of tau_ex / tau_LO into the
change of the mass-normalized tension integral A = int T1 W'^2 / int mu W^2
and of the mass-normalized bending integral B = int P W''^2 / int mu W^2:
    tau_ex / tau_LO = [A(Gamma)/A(0)] / [B(Gamma)/B(0)],  A(0)=c_ii, B(0)=lambda_i(0)^2.
Beam 1, modes 1-3, Gamma = 0, 4, 8, 12, 20 (N = 50).
"""
from analysis_utils import *
mu, P = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
x, w = gl()
T1 = tension_unit(mu, 0.0, x); m = poly(mu, x); p = poly(P, x)
res = {}
for G in (0, 4, 8, 12, 20):
    _, v = solve_modes(50, G, mu, P, 0.0, 3)
    for i, a in enumerate(v):
        W, Wp, Wpp = cheb_eval(a, x), cheb_eval(a, x, 1), cheb_eval(a, x, 2)
        M = np.sum(w * m * W * W)
        res[(G, i)] = (np.sum(w * T1 * Wp ** 2) / M, np.sum(w * p * Wpp ** 2) / M)
print(f"mpmath {mp.__version__}")
for i in range(3):
    A0, B0 = res[(0, i)]
    for G in (4, 8, 12, 20):
        A, B = res[(G, i)]
        print(f"mode {i+1} Gamma={G:2d}: A/A0={A/A0:.4f}  B/B0={B/B0:.4f}  tau_ex/tau_LO={(A/A0)/(B/B0):.4f}")
