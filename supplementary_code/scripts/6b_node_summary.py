"""
6b_node_summary.py
==========================================================================
Post-processing of ../results/6_node_shift_magnitude.json into the
compact per-configuration table of the manuscript (Table 12) and the
validity / reversal statistics quoted in Section 5.3.  No solves.

Range-of-validity statistics use the *exact* first-order derivative
d_FD (independent of the coupling-sum truncation), so that they measure
only the Gamma^4 (nonlinear) departure:
    rho(Gamma) = Delta_xi(Gamma) / (Gamma^2 d_FD),
    Gamma_10   = first grid speed with |rho - 1| > 0.10.
The two-term estimate of the speed at which the net shift changes sign
is Gamma_c = sqrt(|d/e|) when d and e (the Gamma^4 coefficient) have
opposite signs.
==========================================================================
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "results", "6_node_shift_magnitude.json")))
G = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]


def g10(z):
    for g in G[1:]:
        if abs(z["shifts"][str(g)] / (g * g * z["d_FD"]) - 1) > 0.10:
            return g
    return None


tot_ok = tot = 0
print("Table 12 data: config | modes | Gamma interval | K | nodes | agree | rootward (per mode 2..8) | "
      "min|dxi(6)| | max unc | ratio | max |dK-dFD|/|dFD|")
for r in R:
    nd = r["nodes"]
    ok = sum(z["pred_dir"] == z["comp_dir"] for z in nd)
    tot_ok += ok; tot += len(nd)
    rw = [sum(z["comp_dir"] == "root" for z in nd if z["mode"] == m) for m in range(2, 9)]
    mins = min(abs(z["shift6"]) for z in nd); mu = max(z["unc_shift6"] for z in nd)
    rel = [abs(z["d_FD"] - z["d_pert"]) / abs(z["d_FD"]) for z in nd]
    print(f"{r['label']:<34} 2-8  [0,6]  K={r['KMAX']} (sign-stable from K<={max(z['K_sign'] for z in nd)})  "
          f"{len(nd)}  {ok}/{len(nd)}  {rw}  {mins:.1e}  {mu:.0e}  {mins/mu:.0e}  "
          f"median {np.median(rel):.1e} max {max(rel):.1e}")
print(f"TOTAL agree {tot_ok}/{tot}  (modes 3-8 subset: {tot_ok - len(R)}/{tot - len(R)} if all mode-2 agree)")

print("\nGamma_10 (first-order magnitude within 10%) by mode, min-max over nodes and configurations:")
for m in range(2, 9):
    vals = [g10(z) for r in R for z in r["nodes"] if z["mode"] == m]
    v = [x if x is not None else 99 for x in vals]
    print(f"  mode {m}: min {min(v)}  median {np.median(v)}  max {'>20' if max(v)==99 else max(v)}  "
          f"(nodes beyond 20: {sum(x==99 for x in v)})")
allv = [g10(z) for r in R for z in r["nodes"]]
print("  nodes with Gamma_10 <= 2:", [(r['key'], z['mode'], z['node'], g10(z), f"{z['d_FD']:.1e}")
                                   for r in R for z in r["nodes"] if (g10(z) or 99) <= 2])

print("\nDirection reversals over 0 <= Gamma <= 20:")
for r in R:
    for z in r["nodes"]:
        sh = [z["shifts"][str(g)] for g in G]
        flips = [G[k] for k in range(2, len(G)) if np.sign(sh[k]) != np.sign(sh[k - 1])]
        if z["reversals"] or flips:
            gc = (np.sqrt(abs(z["d_FD"] / z["e_FD"])) if np.sign(z["e_FD"]) != np.sign(z["d_FD"]) else None)
            print(f"  {r['key']:<5} mode {z['mode']} node {z['node']} (xi0={z['xi0']:.3f}): trajectory turns near "
                  f"Gamma~{z['reversal_at']}, net shift changes sign near {flips or '>20'}; two-term estimate "
                  f"turn {gc/np.sqrt(2) if gc else float('nan'):.1f}, sign change {gc if gc else float('nan'):.1f}")
