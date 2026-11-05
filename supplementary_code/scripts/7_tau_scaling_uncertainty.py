"""
7_tau_scaling_uncertainty.py
==========================================================================
Energy ratio tau_i(Gamma) = U_c/U_b (manuscript Section 5.4-5.5;
Reviewer 2, comments 4 and 5; Reviewer 3, comment 4).

For the five configurations, modes i = 1..12, and
Gamma in {2,4,6,8,10,12,16,20}:

  tau_ex  : exact ratio evaluated on the rotating (Gamma-dependent) mode
            shape, Eq. (36);
  tau_LO  : leading-order estimate Gamma^2 c_ii / lambda_i(0)^2, Eq. (33),
            evaluated with the NON-rotating shapes (c_ii is the classical
            Southwell coefficient of mode i);
  tau_WKB : high-mode asymptote (Gamma^2/lambda_i) J1/J0 with
            J1 = int T1 mu^(-1/4) P^(-3/4),  J0 = int mu^(1/4) P^(-1/4),
            and its fully explicit form using
            lambda_i ~ [(i-1/2) pi / I]^2,  I = int (mu/P)^(1/4).

Power-law fits  log tau_ex = a + s log i  are made over several mode
ranges at every Gamma, reporting the slope s, its 95% confidence
interval (Student t, n-2 dof), R^2 and the maximum absolute residual in
log10 tau.  The same fits are repeated against (i - 1/2), the variable
suggested by the asymptotic argument.

Transition speeds Gamma*_i at which tau_i = 1 are found for modes 1-4 by
secant iteration in (log Gamma, log tau) on the exact ratio, and
compared with the leading-order prediction Gamma*_LO = lambda_i(0)/sqrt(c_ii).

Writes ../results/7_tau_scaling.json and a printed report.
==========================================================================
"""
import json, time
from scipy import stats
from analysis_utils import *

GAMMAS = [2, 4, 6, 8, 10, 12, 16, 20]
NMODES = 12
OUT = os.path.join(HERE, "..", "results", "7_tau_scaling.json")
RANGES = [(1, 8), (2, 8), (3, 8), (1, 12), (3, 12), (5, 12)]


def energies(a, mu, P, Rbar):
    x, w = gl()
    T1 = tension_unit(mu, Rbar, x)
    Wp, Wpp = cheb_eval(a, x, 1), cheb_eval(a, x, 2)
    Uc1 = np.sum(w * T1 * Wp ** 2)        # per unit Gamma^2
    Ub = np.sum(w * poly(P, x) * Wpp ** 2)
    return Uc1, Ub


def tau_exact(N, G, mu, P, Rbar, nm):
    f, v = solve_modes(N, G, mu, P, Rbar, nm)
    out = []
    for a in v:
        Uc1, Ub = energies(a, mu, P, Rbar)
        out.append(G ** 2 * Uc1 / Ub)
    return [float(x) for x in f], out


def wkb_integrals(mu, P, Rbar):
    x, w = gl(400)
    m, p = poly(mu, x), poly(P, x)
    T1 = tension_unit(mu, Rbar, x)
    J1 = np.sum(w * T1 * m ** -0.25 * p ** -0.75)
    J0 = np.sum(w * m ** 0.25 * p ** -0.25)
    I = np.sum(w * (m / p) ** 0.25)
    return J1, J0, I


def fit(i, t):
    X, Y = np.log(i), np.log(t)
    r = stats.linregress(X, Y)
    n = len(X)
    tcrit = stats.t.ppf(0.975, n - 2) if n > 2 else np.nan
    res = Y - (r.intercept + r.slope * X)
    return dict(slope=r.slope, ci95=tcrit * r.stderr, R2=r.rvalue ** 2,
                maxres_log10=float(np.max(np.abs(res)) / np.log(10)),
                prefactor=float(np.exp(r.intercept)))


def gamma_star(N, mu, P, Rbar, mode, g_lo, t_lo, g_hi, t_hi, tol=1e-6):
    """secant on log tau vs log Gamma for tau_mode(Gamma) = 1."""
    xa, ya = np.log(g_lo), np.log(t_lo)
    xb, yb = np.log(g_hi), np.log(t_hi)
    for _ in range(8):
        xc = xb - yb * (xb - xa) / (yb - ya)
        _, tc = tau_exact(N, float(np.exp(xc)), mu, P, Rbar, mode)
        yc = np.log(tc[mode - 1])
        xa, ya, xb, yb = xb, yb, xc, yc
        if abs(yc) < tol:
            break
    return float(np.exp(xb))


def run_case(key, label, mu, P, Rbar, N0):
    N = N0 + 10           # 12 modes: one resolution step above the node study
    t0 = time.time()
    f0, v0 = solve_modes(N, 0, mu, P, Rbar, NMODES)
    lam0 = np.array([float(x) for x in f0])
    phis = [mass_normalize(a, mu) for a in v0]
    x, w = gl()
    T1 = tension_unit(mu, Rbar, x)
    cii = np.array([np.sum(w * T1 * cheb_eval(p, x, 1) ** 2) for p in phis])
    J1, J0, I = wkb_integrals(mu, P, Rbar)
    rec = dict(key=key, label=label, N=N, Rbar=Rbar, lam0=lam0.tolist(), southwell=cii.tolist(),
               J1=J1, J0=J0, I=I, tau={}, lam={}, fits={}, fits_half={})
    for G in GAMMAS:
        lamG, tG = tau_exact(N, G, mu, P, Rbar, NMODES)
        rec["tau"][str(G)] = tG
        rec["lam"][str(G)] = lamG
        ii = np.arange(1, NMODES + 1)
        tG = np.array(tG)
        rec["fits"][str(G)] = {f"{a}-{b}": fit(ii[a - 1:b], tG[a - 1:b]) for a, b in RANGES}
        rec["fits_half"][str(G)] = {f"{a}-{b}": fit(ii[a - 1:b] - 0.5, tG[a - 1:b]) for a, b in RANGES}
    # transition speeds
    rec["Gstar"] = {}
    for mode in (1, 2, 3, 4):
        gs = [0] + GAMMAS
        ts = [0] + [rec["tau"][str(G)][mode - 1] for G in GAMMAS]
        GLO = lam0[mode - 1] / np.sqrt(cii[mode - 1])
        k = next((j for j in range(1, len(gs)) if ts[j] >= 1), None)
        if k is None:      # tau_i < 1 throughout the grid: extrapolate beyond Gamma = 20
            Gs = gamma_star(N, mu, P, Rbar, mode, gs[-2], ts[-2], gs[-1], ts[-1])
        elif k == 1:       # crossing below Gamma = 2: start from tau ~ Gamma^2 scaling
            Gs = gamma_star(N, mu, P, Rbar, mode, 1.0, ts[1] / 4, gs[1], ts[1])
        else:
            Gs = gamma_star(N, mu, P, Rbar, mode, gs[k - 1], ts[k - 1], gs[k], ts[k])
        rec["Gstar"][str(mode)] = dict(exact=Gs, LO=float(GLO),
                                       WKB=float(np.sqrt(lam0[mode - 1] * J0 / J1)))
    print(f"{label}: done in {time.time()-t0:.0f}s", flush=True)
    return rec


if __name__ == "__main__":
    allres = []
    for cfg in configs():
        allres.append(run_case(*cfg))
        with open(OUT, "w") as f:
            json.dump(allres, f, indent=1)

    for R in allres:
        print("\n" + "=" * 96 + f"\n{R['label']}  (N={R['N']})\n" + "=" * 96)
        print("mode  " + "".join(f"G={G:<9}" for G in GAMMAS) + "  Southwell c_ii")
        for i in range(NMODES):
            print(f"{i+1:<5} " + "".join(f"{R['tau'][str(G)][i]:<11.4g}" for G in GAMMAS)
                  + f"  {R['southwell'][i]:.6g}")
        print("tau_ex / tau_LO:")
        for i in (0, 1, 2, 4, 7, 11):
            print(f"  mode {i+1:<3}" + "".join(
                f"{R['tau'][str(G)][i] / (G**2 * R['southwell'][i] / R['lam0'][i]**2):<11.4f}" for G in GAMMAS))
        print("tau_ex / tau_WKB (with exact lambda_i(Gamma)):")
        for i in (0, 1, 2, 4, 7, 11):
            print(f"  mode {i+1:<3}" + "".join(
                f"{R['tau'][str(G)][i] / (G**2 / R['lam'][str(G)][i] * R['J1'] / R['J0']):<11.4f}" for G in GAMMAS))
        print("fits (slope +- 95% CI, R^2, max|res| in log10):")
        for G in GAMMAS:
            s = "  G=%-3d " % G
            for rng, F in R["fits"][str(G)].items():
                s += f"[{rng}] {F['slope']:+.3f}+-{F['ci95']:.3f} R2={F['R2']:.4f} r={F['maxres_log10']:.3f}  "
            print(s)
        print("fits against (i-1/2):")
        for G in (4, 12, 20):
            s = "  G=%-3d " % G
            for rng, F in R["fits_half"][str(G)].items():
                s += f"[{rng}] {F['slope']:+.3f}+-{F['ci95']:.3f}  "
            print(s)
        print("transition speeds Gamma* (tau_i = 1): exact / leading-order / WKB")
        for m, g in R["Gstar"].items():
            ex = f"{g['exact']:.4f}"
            print(f"  mode {m}: {ex} / {g['LO']:.4f} / {g['WKB']:.4f}")
