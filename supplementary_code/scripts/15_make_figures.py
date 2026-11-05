"""
15_make_figures.py
==========================================================================
Regenerates every figure of the revised manuscript (Figures 2-7) and of
the Supplementary Material (Figures S1-S3) with enlarged labels
(Reviewer 2, comment 14).  Data sources:

  Fig. 2  convergence            ../beam_code/figdata_conv2.json
  Fig. 3  lambda_i vs Gamma       ../results/9_refinement_raw_T4.json, _T6.json
  Fig. 4  mode shapes            solved here (beam 1 N=50, beam 2 N=80)
  Fig. 5  node shifts            ../results/6_node_shift_magnitude.json
  Fig. 6  first-order validity   ../results/6_node_shift_magnitude.json, 7_tau_scaling.json
  Fig. 7  energy ratio           ../results/7_tau_scaling.json
  Fig. S1 precision floor        ../beam_code/figdata_floor3.json
  Fig. S2 error vs DOF           ../beam_code/figdata_fig8.json
  Fig. S3 fit residuals          ../results/7_tau_scaling.json

Output: ../figures/Figure_<k>.pdf/.png and Figure_S<k>.pdf/.png
==========================================================================
"""
import json
from analysis_utils import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(HERE, "..", "results")
BC = BEAM_CODE_DIR
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 12.5,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 10.5,
    "axes.linewidth": 0.9, "mathtext.fontset": "cm", "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 5.5,
})
COL = ["#1f3b73", "#c0392b", "#1e8449", "#8e44ad", "#d68910", "#17a2b8", "#7f8c8d", "#e84393"]
CFG_COL = {"B1": COL[0], "B1R": COL[1], "B2": COL[2], "BJ21": COL[3], "BJ62": COL[4]}
CFG_LBL = {"B1": r"beam 1, $\bar R=0$", "B1R": r"beam 1, $\bar R=0.5$", "B2": "beam 2 (severe)",
           "BJ21": r"$c=0.2$, $n=1$", "BJ62": r"$c=0.6$, $n=2$"}


def save(fig, stem):
    fig.savefig(os.path.join(FIG, stem + ".pdf"))
    fig.savefig(os.path.join(FIG, stem + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", stem)


def panel_label(ax, s):
    t = ax.get_title()
    ax.set_title("")
    ax.set_title(s + " " + t, loc="left")


def fig2():
    d = json.load(open(os.path.join(BC, "figdata_conv2.json")))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for a, key, ttl, lab in [(ax[0], "conv_beam1_G0", r"beam 1, $\Gamma=0$", "(a)"),
                             (ax[1], "conv_beam1_G10", r"beam 1, $\Gamma=10$", "(b)"),
                             (ax[2], "conv_beam2_G0", r"beam 2, $\Gamma=0$", "(c)")]:
        c = d[key]
        for m, mk, cl in (("m1", "o-", COL[0]), ("m3", "s-", COL[1]), ("m5", "^-", COL[2])):
            a.semilogy(c["N"], c[m], mk, color=cl, label=f"mode {m[1]}")
        a.axhspan(1e-24, 1e-22, color="0.85", alpha=0.6, zorder=0)
        a.set_title(ttl); a.set_xlabel(r"Chebyshev degree $N$"); a.set_ylim(1e-24, 5)
        panel_label(a, lab)
    ax[0].set_ylabel("relative frequency error")
    ax[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "Figure_2")


def figS1():
    d = json.load(open(os.path.join(BC, "figdata_floor3.json")))["floor_beam1"]
    fig, a = plt.subplots(figsize=(6, 4.3))
    a.semilogy(d["N"], d["d34"], "o-", color=COL[1], label="34-digit arithmetic")
    a.semilogy(d["N"], d["d50"], "s-", color=COL[0], label="50-digit arithmetic")
    a.set_xlabel(r"Chebyshev degree $N$"); a.set_ylabel(r"relative error, mode 1")
    a.set_title(r"beam 1, $\Gamma=0$"); a.legend(loc="lower left"); a.set_ylim(1e-29, 1e-3)
    save(fig, "Figure_S1")


def figS2():
    d = json.load(open(os.path.join(BC, "figdata_fig8.json")))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    dof_p = [n + 1 for n in d["N"]]; dof_f = [2 * n for n in d["nel"]]
    for a, m, lab in ((ax[0], "m1", "(a) mode 1"), (ax[1], "m5", "(b) mode 5")):
        a.loglog(dof_p, d["present_" + m], "o-", color=COL[0], label="present, 34 digits")
        a.loglog(dof_f, d["fem_" + m], "s--", color=COL[1], label="Hermite-cubic FEM, double")
        a.axhline(1e-6, color="0.5", ls=":", lw=1)
        a.set_xlabel("degrees of freedom"); a.set_title(lab)
    ax[0].set_ylabel("relative frequency error"); ax[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "Figure_S2")


def fig3():
    T4 = json.load(open(os.path.join(RES, "9_refinement_raw_T4.json")))
    T6 = json.load(open(os.path.join(RES, "9_refinement_raw_T6.json")))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    G1 = sorted({float(k.split("G=")[1]) for k in T4})
    lam1 = np.array([[float(x) for x in T4[f"T4|B1 G={g:g}"]["vals"]["60"][:5]] for g in G1])
    G2 = list(range(13))
    lam2 = np.array([[float(x) for x in T6[f"T6|B2 G={g}"]["vals"]["90"][:5]] for g in G2])
    for a, G, lam, ttl, lab in ((ax[0], G1, lam1, "beam 1", "(a)"), (ax[1], G2, lam2, "beam 2", "(b)")):
        for i in range(3):
            a.plot(G, lam[:, i] / lam[0, i], "o-", color=COL[i], label=f"mode {i+1}")
        a.set_xlabel(r"rotational speed $\Gamma$"); a.set_title(ttl); panel_label(a, lab)
    ax[0].set_ylabel(r"$\lambda_i(\Gamma)/\lambda_i(0)$")
    ax[0].legend(loc="upper left", bbox_to_anchor=(0.06, 0.92))
    fig.tight_layout()
    save(fig, "Figure_3")


def fig4():
    mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
    mu2, P2 = wang_wereley(-0.8, [-0.95])
    x = np.linspace(0, 1, 801)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for a, (mu, P, N), ttl, lab in ((ax[0], (mu1, P1, 50), "beam 1", "(a)"),
                                    (ax[1], (mu2, P2, 80), "beam 2", "(b)")):
        _, v = solve_modes(N, 0, mu, P, 0.0, 4)
        for i in range(4):
            w = cheb_eval(v[i], x); a.plot(x, w / w[-1], color=COL[i], label=f"mode {i+1}")
        a.axhline(0, color="0.5", lw=0.8); a.set_xlabel(r"$\xi=y/L$"); a.set_title(ttl)
        panel_label(a, lab)
    ax[0].set_ylabel(r"$W(\xi)/W(1)$"); ax[1].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "Figure_4")


def load_nodes():
    return {R["key"]: R for R in json.load(open(os.path.join(RES, "6_node_shift_magnitude.json")))}


def load_fine():
    return {k: json.load(open(os.path.join(RES, f"6c_node_fine_grid_{k}.json")))
            for k in ("B1", "B1R", "B2", "BJ21", "BJ62")}


def fig5():
    R = load_nodes(); F = load_fine()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    a = ax[0]
    Gf = np.linspace(0, 12, 200)
    k = 0
    for z in F["B1"]["nodes"]:
        if z["mode"] not in (3, 4):
            continue
        G = np.array([g for g in F["B1"]["gammas"] if 0 < g <= 12 and (g >= 0.5)])
        sh = np.array([z["shifts"][str(g)] for g in G])
        d = z["d_fits"]["0.1-0.2"][0]
        a.plot(G[::2], 1e3 * sh[::2], "o", color=COL[k], ms=5,
               label=rf"mode {z['mode']}, node {z['node']} ($\xi_0$={z['xi0']:.3f})")
        a.plot(Gf, 1e3 * Gf ** 2 * d, "--", color=COL[k], lw=1.3)
        k += 1
    a.axhline(0, color="0.5", lw=0.8)
    a.set_xlabel(r"rotational speed $\Gamma$"); a.set_ylabel(r"node shift $\Delta\xi_{i,j}\times10^{3}$")
    a.set_title("beam 1: computed (symbols) vs first order (dashed)")
    a.legend(loc="lower left"); panel_label(a, "(a)")
    a = ax[1]
    for key, mk in (("BJ21", "o"), ("BJ62", "s")):
        zs = [z for z in R[key]["nodes"] if z["mode"] == 8]
        xs = [z["xi0"] for z in zs]; ys = [1e3 * z["shift6"] for z in zs]
        mk_, st, bl = a.stem(xs, ys, linefmt="-", markerfmt=mk, basefmt=" ")
        plt.setp(st, color=CFG_COL[key], lw=1.4); plt.setp(mk_, color=CFG_COL[key], ms=7)
        mk_.set_label(CFG_LBL[key] + f" ({sum(y < 0 for y in ys)} rootward)")
    a.axhline(0, color="0.3", lw=0.9)
    a.set_xlabel(r"node position at $\Gamma=0$, $\xi_{8,j}$"); a.set_ylabel(r"$\Delta\xi_{8,j}(\Gamma=6)\times10^{3}$")
    a.set_title("mode 8: contrasting migration patterns"); a.legend(loc="lower right")
    panel_label(a, "(b)")
    fig.tight_layout()
    save(fig, "Figure_5")


def fig6():
    F = load_fine()
    T = {r["key"]: r for r in json.load(open(os.path.join(RES, "7_tau_scaling.json")))}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    a = ax[0]
    for key, Rk in F.items():
        G = [g for g in Rk["gammas"] if g >= 0.5]
        for z in Rk["nodes"]:
            d = z["d_fits"]["0.1-0.2"][0]
            a.plot(G, [z["shifts"][str(g)] / (g * g * d) for g in G], "-", color=CFG_COL[key], lw=0.9, alpha=0.75)
        a.plot([], [], "-", color=CFG_COL[key], lw=2.2, label=CFG_LBL[key])
    a.axhspan(0.9, 1.1, color="0.8", alpha=0.6, zorder=0)
    a.set_ylim(-0.5, 1.6); a.set_xlim(0, 20)
    a.set_xlabel(r"rotational speed $\Gamma$")
    a.set_ylabel(r"$\rho=\Delta\xi_{\rm computed}/(\Gamma^2 d^{\rm FD})$")
    a.set_title("all 140 nodes, modes 2-8"); a.legend(loc="lower left"); panel_label(a, "(a)")
    a = ax[1]
    for n, (key, Rk) in enumerate(F.items()):
        R = T[key]
        Gt = np.array([int(k) for k in R["tau"]], float)
        xs, ys = [], []
        for z in Rk["nodes"]:
            g = z["gamma10"]
            if g is None:
                continue
            t = np.array([R["tau"][k][z["mode"] - 1] for k in R["tau"]])
            tg = t[0] * (g / Gt[0]) ** 2 if g <= Gt[0] else float(np.exp(np.interp(np.log(g), np.log(Gt), np.log(t))))
            xs.append(z["mode"] + 0.08 * (n - 2)); ys.append(tg)
        a.semilogy(xs, ys, "o", color=CFG_COL[key], label=CFG_LBL[key], ms=5, alpha=0.85)
    a.axhspan(TAU_BAND[0], TAU_BAND[1], color="0.85", alpha=0.6, zorder=0)
    a.set_xlabel("mode number $i$"); a.set_ylabel(r"$\tau_i$ at interpolated $\Gamma_{10}$")
    a.set_title(r"energy ratio where first-order error reaches 10%"); panel_label(a, "(b)")
    fig.tight_layout()
    save(fig, "Figure_6")


TAU_BAND = (0.127, 0.262)   # 10-90% range of tau at interpolated Gamma_10 (6d output)


def fig7():
    T = {r["key"]: r for r in json.load(open(os.path.join(RES, "7_tau_scaling.json")))}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    for key, R in T.items():
        i = np.arange(1, 13)
        t = np.array(R["tau"]["12"])
        a.loglog(i - 0.5, t, "o", color=CFG_COL[key], label=CFG_LBL[key])
        wkb = 144 * R["J1"] * R["I"] ** 2 / (R["J0"] * ((i - 0.5) * np.pi) ** 2)
        a.loglog(i - 0.5, wkb, "--", color=CFG_COL[key], lw=1.1)
    a.set_xlabel(r"$i-\frac{1}{2}$"); a.set_ylabel(r"$\tau_i(\Gamma=12)$")
    a.set_title("symbols: exact; dashed: WKB asymptote"); a.legend(loc="lower left")
    panel_label(a, "(a)")
    a = ax[1]
    R = T["B1"]
    G = np.array([int(k) for k in R["tau"]])
    Gf = np.linspace(0, 20, 200)
    for i in range(4):
        t = np.array([R["tau"][k][i] for k in R["tau"]])
        a.plot(G, t, "o-", color=COL[i], label=f"mode {i+1}")
        a.plot(Gf, Gf ** 2 * R["southwell"][i] / R["lam0"][i] ** 2, ":", color=COL[i], lw=1.4)
        gs = R["Gstar"][str(i + 1)]["exact"]
        if gs and gs <= 20:
            a.plot([gs], [1], "k*", ms=11, zorder=5)
    a.axhline(1, color="0.3", lw=0.9)
    a.set_yscale("log"); a.set_ylim(1e-2, 30); a.set_xlim(0, 20.5)
    a.set_xlabel(r"rotational speed $\Gamma$"); a.set_ylabel(r"$\tau_i=U_{c,i}/U_{b,i}$")
    a.set_title(r"beam 1: exact (solid), leading order (dotted), $\Gamma_i^*$ ($\star$)")
    a.legend(loc="lower right"); panel_label(a, "(b)")
    fig.tight_layout()
    save(fig, "Figure_7")


def figS3():
    T = json.load(open(os.path.join(RES, "7_tau_scaling.json")))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for R in T:
        Gs = [int(k) for k in R["fits"]]
        for a, rng, ttl in ((ax[0], "1-8", "fit over modes 1-8"), (ax[1], "5-12", "fit over modes 5-12")):
            s = [R["fits"][str(g)][rng]["slope"] for g in Gs]
            e = [R["fits"][str(g)][rng]["ci95"] for g in Gs]
            a.errorbar(Gs, s, yerr=e, fmt="o-", color=CFG_COL[R["key"]], capsize=3, label=CFG_LBL[R["key"]])
            a.set_title(ttl); a.set_xlabel(r"rotational speed $\Gamma$")
    for a in ax:
        a.axhline(-2, color="0.3", ls="--", lw=1)
    ax[0].set_ylabel(r"fitted exponent $s$ in $\tau_i\propto i^{s}$ (95% CI)")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "Figure_S3")


if __name__ == "__main__":
    which = sys.argv[1:] or ["2", "3", "4", "5", "6", "7", "S1", "S2", "S3"]
    for w in which:
        globals()["fig" + w]()
