"""Generate Figures 2 and 3: exponential convergence and the finite-
precision floor, using the well-scaled boundary reduction of Section 4.2."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
    "mathtext.fontset": "cm", "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 9,
})
C = {"m1": "#1f3b73", "m3": "#c0392b", "m5": "#1e8449", "d34": "#c0392b", "d50": "#1f3b73"}

d = {}
d.update(json.load(open("figdata_conv2.json")))
d.update(json.load(open("figdata_floor3.json")))

def save(fig, stem):
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=200)
    plt.close(fig); print("wrote", stem + ".pdf/.png")

# ---- Figure 2: convergence (corrected: floor near 1e-22, not 1e-16) -------
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
for a, key, ttl in [
    (ax[0], "conv_beam1_G0",  r"Beam 1,  $\Gamma=0$"),
    (ax[1], "conv_beam1_G10", r"Beam 1,  $\Gamma=10$"),
    (ax[2], "conv_beam2_G0",  r"Beam 2,  $\Gamma=0$")]:
    c = d[key]
    a.semilogy(c["N"], c["m1"], "o-", ms=3.5, lw=1.1, color=C["m1"], label="mode 1")
    a.semilogy(c["N"], c["m3"], "s-", ms=3.5, lw=1.1, color=C["m3"], label="mode 3")
    a.semilogy(c["N"], c["m5"], "^-", ms=3.5, lw=1.1, color=C["m5"], label="mode 5")
    a.set_title(ttl, fontsize=10); a.set_xlabel(r"Chebyshev degree $N$")
    a.set_ylim(1e-24, 5)
ax[0].set_ylabel(r"relative frequency error")
ax[0].legend(loc="upper right")
# well-scaled-reduction floor band: below ~1e-22 (Section 4.2), not ~1e-16
for a in ax:
    a.axhspan(1e-24, 1e-22, color="0.85", alpha=0.5, zorder=0)
fig.suptitle("Exponential convergence of the collocation frequencies "
             "(shaded band: well-scaled-reduction finite-precision floor, "
             r"below $\approx 10^{-22}$)", fontsize=9.5, y=1.03)
save(fig, "fig2_convergence")

# ---- Figure 3: precision floor (corrected) ---------------------------------
fig, a = plt.subplots(figsize=(5.2, 3.8))
f = d["floor_beam1"]
a.semilogy(f["N"], f["d34"], "o-", ms=4, lw=1.2, color=C["d34"], label="34-digit arithmetic")
a.semilogy(f["N"], f["d50"], "s-", ms=4, lw=1.2, color=C["d50"], label="50-digit arithmetic")
a.set_xlabel(r"Chebyshev degree $N$")
a.set_ylabel(r"relative error, mode 1 (beam 1, $\Gamma=0$)")
a.set_title("Finite-precision conditioning and the convergence floor:\n"
            "raising precision lowers it (well-scaled reduction)", fontsize=10)
a.legend(loc="lower left")
a.set_ylim(1e-29, 1e-3)
save(fig, "fig3_precision_floor")
