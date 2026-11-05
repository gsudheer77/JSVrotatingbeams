"""
make_figures_4_6.py
==========================================================================
Regenerates Figures 4-6 directly from the numerical solver, with no
cached figure data required.

Figure 4: first three non-dimensional frequencies of tapered beam 1
          versus rotational speed Gamma (cantilever, N=50, 34-digit
          precision) -- Table 4's data, plotted.
Figure 5: cantilever mode shapes at Gamma=0, first four modes,
          normalized to unit tip amplitude: beam 1 and beam 2.
Figure 6: interior node positions of modes 3 and 4 versus Gamma,
          beam 1, found as sign changes of the reconstructed mode
          shape on a fine grid (bisection-refined).

Run: python3 make_figures_4_6.py
==========================================================================
"""
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chebyshev_rotating_beam import solve, wang_wereley
import per_case_verification as pcv

mp.mp.dps = 34

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
    "mathtext.fontset": "cm", "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 9,
})
C = {"m1": "#1f3b73", "m2": "#c0392b", "m3": "#1e8449", "m4": "#8e44ad"}


def save(fig, stem):
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=200)
    plt.close(fig)
    print("wrote", stem + ".pdf/.png")


def mode_shape_grid(a, xi_grid):
    return np.array([float(pcv.mode_derivs(a, x)[0]) for x in xi_grid])


def interior_nodes(a, n_expected, xi_lo=0.02, xi_hi=0.995, n_scan=400):
    """Interior zero crossings of the reconstructed mode shape a, found
    by scanning for sign changes then bisection-refining each one."""
    grid = np.linspace(xi_lo, xi_hi, n_scan)
    vals = mode_shape_grid(a, grid)
    roots = []
    for i in range(len(grid) - 1):
        if vals[i] == 0:
            roots.append(grid[i])
        elif vals[i] * vals[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            flo = vals[i]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                fmid = mode_shape_grid(a, [mid])[0]
                if flo * fmid <= 0:
                    hi = mid
                else:
                    lo, flo = mid, fmid
            roots.append(0.5 * (lo + hi))
    return roots[:n_expected]


def main():
    mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])   # beam 1
    mu2, P2 = wang_wereley(-0.8, [-0.95])                # beam 2
    N = 50

    # ---- Figure 4: frequency vs. rotational speed, beam 1, modes 1-3 --
    Gammas = list(range(0, 13, 2))
    freqs_by_mode = [[], [], []]
    for G in Gammas:
        f = solve(N, G, mu1, P1, nmodes=3)
        for m in range(3):
            freqs_by_mode[m].append(float(f[m]))

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for m, lbl, col in zip(range(3), ["mode 1", "mode 2", "mode 3"],
                            [C["m1"], C["m2"], C["m3"]]):
        ax.plot(Gammas, freqs_by_mode[m], "o-", ms=4, lw=1.2, color=col, label=lbl)
    ax.set_xlabel(r"rotational speed $\Gamma$")
    ax.set_ylabel(r"non-dimensional frequency $\lambda$")
    ax.set_title("Figure 4: frequency vs. rotational speed\n"
                  r"(tapered beam 1, cantilever, $N=50$, 34-digit precision)",
                  fontsize=10)
    ax.legend(loc="upper left")
    save(fig, "fig4_frequency_speed")

    # ---- Figure 5: mode shapes, Gamma=0, beam 1 and beam 2, modes 1-4 -
    xi_grid = np.linspace(0.0, 1.0, 801)
    _, vecs1 = solve(N, 0, mu1, P1, nmodes=4, return_vectors=True)
    _, vecs2 = solve(N, 0, mu2, P2, nmodes=4, return_vectors=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True)
    for beam_ax, vecs, ttl in [(axes[0], vecs1, "beam 1"),
                                (axes[1], vecs2, "beam 2")]:
        for m in range(4):
            w = mode_shape_grid(vecs[m], xi_grid)
            w = w / w[-1]  # unit tip amplitude
            beam_ax.plot(xi_grid, w, lw=1.2, label=f"mode {m+1}")
        beam_ax.set_title(ttl, fontsize=10)
        beam_ax.set_xlabel(r"$\xi$")
        beam_ax.axhline(0, color="0.6", lw=0.6)
    axes[0].set_ylabel(r"$W(\xi)$ (unit tip amplitude)")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure 5: cantilever mode shapes at " r"$\Gamma=0$" ", first four modes",
                 fontsize=10, y=1.03)
    save(fig, "fig5_modeshapes")

    # ---- Figure 6: interior node positions, modes 3 and 4, beam 1 -----
    Gammas_nodes = list(range(0, 13, 2))
    nodes3, nodes4 = [], []
    for G in Gammas_nodes:
        _, vecs = solve(N, G, mu1, P1, nmodes=4, return_vectors=True)
        n3 = interior_nodes(vecs[2], 2)
        n4 = interior_nodes(vecs[3], 3)
        nodes3.append(n3)
        nodes4.append(n4)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    n3 = np.array(nodes3)
    n4 = np.array(nodes4)
    for k in range(n3.shape[1]):
        ax.plot(Gammas_nodes, n3[:, k], "o-", ms=4, lw=1.1, color=C["m3"],
                label="mode 3" if k == 0 else None)
    for k in range(n4.shape[1]):
        ax.plot(Gammas_nodes, n4[:, k], "s-", ms=4, lw=1.1, color=C["m4"],
                label="mode 4" if k == 0 else None)
    ax.set_xlabel(r"rotational speed $\Gamma$")
    ax.set_ylabel(r"interior node position $\xi$")
    ax.set_title("Figure 6: interior node migration, beam 1\n(modes 3 and 4)",
                 fontsize=10)
    ax.legend(loc="center left")
    save(fig, "fig6_node_migration")


if __name__ == "__main__":
    main()
