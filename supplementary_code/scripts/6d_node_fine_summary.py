"""
6d_node_fine_summary.py
Post-processing of 6c_node_fine_grid_*.json (no solves):
  * Gamma_10: last passing and first failing grid speed (0.5 spacing) and
    the linearly interpolated 10% crossing of |rho - 1|, with
    rho = Delta_xi / (Gamma^2 d) and d the finite-difference estimate from
    the speed pair (0.1, 0.2);
  * tau_i at the interpolated Gamma_10 (interpolation in log tau vs log
    Gamma of the exact ratios of 7_tau_scaling.json);
  * spread of the finite-difference derivative estimate between the speed
    pairs (0.05,0.1), (0.1,0.2), (0.25,0.5);
  * agreement of the K = 16 perturbation prediction (script 6) with d.
"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
KEYS = ["B1", "B1R", "B2", "BJ21", "BJ62"]
tau = {r["key"]: r for r in json.load(open(os.path.join(RES, "7_tau_scaling.json")))}
old = {r["key"]: r for r in json.load(open(os.path.join(RES, "6_node_shift_magnitude.json")))}


def tau_at(key, mode, g):
    R = tau[key]
    G = np.array([int(k) for k in R["tau"]], float)
    t = np.array([R["tau"][k][mode - 1] for k in R["tau"]])
    if g <= G[0]:
        return t[0] * (g / G[0]) ** 2          # tau ~ Gamma^2 at low speed
    return float(np.exp(np.interp(np.log(g), np.log(G), np.log(t))))


rows, allg, alltau, spread_small, spread_large, relK = [], {}, [], [], [], []
print("config mode node  last_pass first_fail Gamma10(interp)  tau(Gamma10)")
for key in KEYS:
    R = json.load(open(os.path.join(RES, f"6c_node_fine_grid_{key}.json")))
    for z, zo in zip(R["nodes"], old[key]["nodes"]):
        d = z["d_fits"]["0.1-0.2"][0]
        ds = z["d_fits"]["0.05-0.1"][0]; dl = z["d_fits"]["0.25-0.5"][0]
        spread_small.append(abs(ds - d) / abs(d)); spread_large.append(abs(dl - d) / abs(d))
        relK.append(abs(zo["d_pert"] - d) / abs(d))
        g10 = z["gamma10"]
        allg.setdefault(z["mode"], []).append(g10 if g10 is not None else np.inf)
        tg = tau_at(key, z["mode"], g10) if g10 is not None else None
        if tg is not None:
            alltau.append((tg, key, z["mode"], z["node"]))
        print(f"{key:<5} {z['mode']:>3} {z['node']:>4}  {str(z['last_pass']):>9} {str(z['first_fail']):>10} "
              f"{(f'{g10:.2f}' if g10 is not None else '>20'):>14}  {(f'{tg:.3f}' if tg is not None else '-'):>8}")

print("\nInterpolated Gamma_10 by mode (min / median / max; inf = still within 10% at Gamma=20):")
for m in sorted(allg):
    v = np.array(allg[m])
    print(f"  mode {m}: {np.min(v):.2f} / {np.median(v):.2f} / {np.max(v):.2f}   (beyond 20: {int(np.sum(np.isinf(v)))})")
t = np.array([x[0] for x in alltau])
print(f"\ntau_i at interpolated Gamma_10 (n={len(t)}): median {np.median(t):.3f}, "
      f"10-90% {np.percentile(t,10):.3f}-{np.percentile(t,90):.3f}, min {t.min():.3f}, max {t.max():.3f}")
print("  lowest:", [(round(a, 3), b, c, d) for a, b, c, d in sorted(alltau)[:6]])
print(f"\nFinite-difference derivative estimate, relative spread vs the (0.1,0.2) estimate:")
print(f"  (0.05,0.1): median {np.median(spread_small):.1e}, max {np.max(spread_small):.1e}")
print(f"  (0.25,0.5): median {np.median(spread_large):.1e}, max {np.max(spread_large):.1e}")
print(f"K=16 perturbation prediction vs (0.1,0.2) estimate: median {np.median(relK):.1e}, max {np.max(relK):.1e}")
for key in KEYS:
    R = json.load(open(os.path.join(RES, f"6c_node_fine_grid_{key}.json")))
    rk = [abs(zo["d_pert"] - z["d_fits"]["0.1-0.2"][0]) / abs(z["d_fits"]["0.1-0.2"][0]) for z, zo in zip(R["nodes"], old[key]["nodes"])]
    print(f"  {key}: median {np.median(rk):.1e}")
