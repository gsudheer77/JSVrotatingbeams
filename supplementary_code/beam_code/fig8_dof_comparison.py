"""
fig8_dof_comparison.py
==========================================================================
Regenerates Figure 8 (Section 4.15): relative frequency error versus
degrees of freedom, mode 1 and mode 5, beam 1 (Gamma=0), comparing the
present trigonometric collocation method (DOF = N+1 Chebyshev
coefficients) against the unscaled double-precision Hermite-cubic FEM
of fem_hermite_solver.py (DOF = 2*n_el after the clamped root is
imposed). Uses the same solvers as fem_comparison.py and Table 14.
==========================================================================
"""
import json
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chebyshev_rotating_beam as cb
from fem_hermite_solver import solve_fem

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
    "mathtext.fontset": "cm", "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 9,
})
C = {"present": "#1f3b73", "fem": "#c0392b"}

mp.mp.dps = 34
mu, P = cb.wang_wereley(-0.5, [-1.5, 0.75, -0.125])  # beam 1

# converged reference (mode indices 0 and 4 for modes 1 and 5)
mp.mp.dps = 50
ref = cb.solve(60, 0, mu, P, nmodes=5)
ref = [float(x) for x in ref]
mp.mp.dps = 34

N_list = [8, 12, 16, 20, 26, 31, 36, 40]
nel_list = [4, 8, 16, 32, 64, 128, 256, 512]

data = {"N": N_list, "nel": nel_list,
        "present_m1": [], "present_m5": [], "fem_m1": [], "fem_m5": []}

for N in N_list:
    f = cb.solve(N, 0, mu, P, nmodes=5)
    data["present_m1"].append(float(abs(f[0] - ref[0]) / ref[0]))
    data["present_m5"].append(float(abs(f[4] - ref[4]) / ref[4]))

for nel in nel_list:
    f, _, _ = solve_fem(nel, 0.0, [1.0, -0.5], [1.0, -1.5, 0.75, -0.125], nmodes=5)
    data["fem_m1"].append(abs(f[0] - ref[0]) / ref[0])
    data["fem_m5"].append(abs(f[4] - ref[4]) / ref[4])

json.dump(data, open("figdata_fig8.json", "w"))

dof_present = [N + 1 for N in N_list]
dof_fem = [2 * n for n in nel_list]

fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
titles = [r"mode 1", r"mode 5"]
keys_present = ["present_m1", "present_m5"]
keys_fem = ["fem_m1", "fem_m5"]

for a, ttl, kp, kf in zip(ax, titles, keys_present, keys_fem):
    a.loglog(dof_present, data[kp], "o-", ms=4, lw=1.2, color=C["present"],
              label="present (Chebyshev coeff.)")
    a.loglog(dof_fem, data[kf], "s-", ms=4, lw=1.2, color=C["fem"],
              label="FEM (Hermite, double precision)")
    a.axhline(1e-6, color="0.5", lw=0.8, ls=":")
    a.set_title(ttl, fontsize=10)
    a.set_xlabel("degrees of freedom")
ax[0].set_ylabel("relative frequency error")
ax[0].legend(loc="lower left")
fig.suptitle("Figure 8. Frequency error versus degrees of freedom "
             "(beam 1, " + r"$\Gamma=0$" + ")", fontsize=10, y=1.03)
fig.savefig("fig8_dof_comparison.pdf")
fig.savefig("fig8_dof_comparison.png", dpi=200)
plt.close(fig)
print("wrote fig8_dof_comparison.pdf/.png")
