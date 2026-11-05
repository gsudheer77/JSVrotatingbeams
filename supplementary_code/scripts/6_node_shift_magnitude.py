"""
6_node_shift_magnitude.py
==========================================================================
Quantitative test of the first-order nodal-migration formula
(manuscript Section 5.2-5.3; Reviewer 2, comments 2 and 3).

For each of the five configurations and modes 2-8 (every interior node):

  * xi_j(Gamma): node positions computed directly from the 34-digit
    eigenvectors on the speed grid Gamma = 0 ... 20;
  * d_K = d xi_j / d(Gamma^2) at Gamma = 0 predicted by the first-order
    formula (Eqs. 30-31) with the coupling sum truncated to the first K
    modes, K = i+1 ... KMAX;
  * d_FD: the same derivative obtained independently from the computed
    node positions at Gamma = 0.25 and 0.5 (two-term Richardson fit
    Delta = d Gamma^2 + e Gamma^4), i.e. without any perturbation theory;
  * rho(Gamma) = Delta_computed(Gamma) / (Gamma^2 d_KMAX): ratio of the
    computed shift to the first-order prediction; Gamma_10 is the speed
    at which |rho - 1| first exceeds 0.10;
  * monotonicity of xi_j(Gamma) over the grid (direction reversals);
  * numerical uncertainty of node positions and shifts from an N+10
    re-solve at Gamma = 0 and 6.

Writes ../results/6_node_shift_magnitude.json and a printed summary.
==========================================================================
"""
import json, time
from analysis_utils import *

GAMMAS = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]
KMAX = 16
MODES = range(2, 9)          # mode numbers i (1-based); mode i has i-1 nodes
OUT = os.path.join(HERE, "..", "results", "6_node_shift_magnitude.json")


def coupling(phis, mu, Rbar):
    x, w = gl()
    T1 = tension_unit(mu, Rbar, x)
    D = [cheb_eval(p, x, 1) for p in phis]
    K = len(phis)
    Cm = np.zeros((K, K))
    for a in range(K):
        for b in range(a, K):
            Cm[a, b] = Cm[b, a] = np.sum(w * T1 * D[a] * D[b])
    return Cm


def run_case(key, label, mu, P, Rbar, N):
    t0 = time.time()
    sol = {}
    for G in GAMMAS:
        f, v = solve_modes(N, G, mu, P, Rbar, KMAX)
        sol[G] = (f, v)
    fN, vN = {}, {}
    for G in (0, 6):
        fN[G], vN[G] = solve_modes(N + 10, G, mu, P, Rbar, 9)

    f0, v0 = sol[0]
    phis = [mass_normalize(a, mu) for a in v0]
    lam2 = np.array([float(x) ** 2 for x in f0])
    Cm = coupling(phis, mu, Rbar)

    nodes_out = []
    for i in MODES:
        n = i - 1                       # 0-based mode index
        xi_G = {}
        for G in GAMMAS:
            r, cnt = interior_nodes(sol[G][1][n], i - 1)
            assert cnt == i - 1, (label, i, G, cnt)
            xi_G[G] = r
        xiN0, _ = interior_nodes(vN[0][n], i - 1)
        xiN6, _ = interior_nodes(vN[6][n], i - 1)
        for j in range(i - 1):
            x0 = xi_G[0][j]
            # truncated first-order predictions d_K
            dK = {}
            for K in range(i + 1, KMAX + 1):
                s = 0.0
                for k in range(K):
                    if k == n:
                        continue
                    s += Cm[k, n] / (lam2[n] - lam2[k]) * cheb_eval(phis[k], x0)
                dK[K] = -s / cheb_eval(phis[n], x0, 1)
            d = dK[KMAX]
            # sign-stable truncation order
            Ksign = KMAX
            for K in range(KMAX, i, -1):
                if np.sign(dK[K]) == np.sign(d):
                    Ksign = K
                else:
                    break
            # independent finite-difference derivative
            g1, g2 = 0.25 ** 2, 0.5 ** 2
            D1, D2 = xi_G[0.25][j] - x0, xi_G[0.5][j] - x0
            A = np.array([[g1, g1 ** 2], [g2, g2 ** 2]])
            dFD, eFD = np.linalg.solve(A, [D1, D2])
            traj = [xi_G[G][j] for G in GAMMAS]
            shifts = {G: xi_G[G][j] - x0 for G in GAMMAS}
            rho = {G: (shifts[G] / (G ** 2 * d) if G > 0 else 1.0) for G in GAMMAS}
            G10 = None
            for G in GAMMAS[1:]:
                if abs(rho[G] - 1) > 0.10:
                    G10 = G
                    break
            inc = np.diff(traj)
            sgn = np.sign(inc[np.abs(inc) > 1e-13])
            reversals = int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0
            rev_at = [GAMMAS[k + 1] for k in range(1, len(inc)) if np.sign(inc[k]) != np.sign(inc[k - 1]) and abs(inc[k]) > 1e-13]
            unc0 = abs(xiN0[j] - x0)
            unc6 = abs(xiN6[j] - xi_G[6][j])
            nodes_out.append(dict(
                mode=i, node=j + 1, xi0=x0, d_pert=d, d_FD=dFD, e_FD=eFD,
                dK={str(k): v for k, v in dK.items()}, K_sign=Ksign,
                pred_dir="tip" if d > 0 else "root",
                shift6=shifts[6], comp_dir="tip" if shifts[6] > 0 else "root",
                shifts={str(G): shifts[G] for G in GAMMAS},
                rho={str(G): rho[G] for G in GAMMAS},
                Gamma10=G10, reversals=reversals, reversal_at=rev_at,
                unc_xi0=unc0, unc_xi6=unc6, unc_shift6=unc0 + unc6))
    print(f"{label}: done in {time.time()-t0:.0f}s", flush=True)
    return dict(key=key, label=label, N=N, Rbar=Rbar, KMAX=KMAX,
                lam0=[float(x) for x in f0[:9]], nodes=nodes_out)


if __name__ == "__main__":
    allres = []
    for cfg in configs():
        allres.append(run_case(*cfg))
        with open(OUT, "w") as f:
            json.dump(allres, f, indent=1)

    print("\n" + "=" * 100)
    print("SUMMARY per configuration (modes 3-8 = original 135-node set; mode 2 added)")
    print("=" * 100)
    for R in allres:
        nd = R["nodes"]
        for sel, name in ((lambda z: z["mode"] >= 3, "modes 3-8"), (lambda z: z["mode"] == 2, "mode 2")):
            s = [z for z in nd if sel(z)]
            ok = sum(z["pred_dir"] == z["comp_dir"] for z in s)
            nroot = sum(z["comp_dir"] == "root" for z in s)
            mins = min(abs(z["shift6"]) for z in s)
            maxunc = max(z["unc_shift6"] for z in s)
            relFD = max(abs(z["d_FD"] - z["d_pert"]) / abs(z["d_pert"]) for z in s)
            Ks = max(z["K_sign"] for z in s)
            G10 = [z["Gamma10"] for z in s]
            G10min = min([g for g in G10 if g is not None], default=None)
            rev = sum(z["reversals"] > 0 for z in s)
            print(f"{R['label']:<36} {name:<9} agree {ok}/{len(s)}  rootward {nroot}  "
                  f"min|shift(6)|={mins:.2e}  max unc={maxunc:.1e}  max|dFD-dpert|/|d|={relFD:.1e}  "
                  f"K_sign<={Ks}  min Gamma10={G10min}  nodes w/ reversal={rev}")
