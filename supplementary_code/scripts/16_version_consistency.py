"""
16_version_consistency.py
Checks that results produced under mpmath 1.4.1 (script 6 node positions;
script 9 tables T6/T8) are reproduced under the mpmath version installed
now (1.3.0 in the reported environment).
  (a) beam 2, Gamma = 0 and 12, N = 90: frequencies vs. 9_refinement_raw_T6.json
  (b) beam 1 node shifts at Gamma = 0.25, 0.5, 6, 12, 20 from 6c (this version)
      vs. 6_node_shift_magnitude.json (1.4.1)
"""
import json
from analysis_utils import *
RES = os.path.join(HERE, "..", "results")
print(f"mpmath {mp.__version__} (backend {mp.libmp.BACKEND})")
mu2, P2 = wang_wereley(-0.8, [-0.95])
T6 = json.load(open(os.path.join(RES, "9_refinement_raw_T6.json")))
for G in (0, 12):
    f = solve(90, G, mu2, P2, nmodes=5)
    old = [mp.mpf(x) for x in T6[f"T6|B2 G={G}"]["vals"]["90"]]
    d = max(abs(a - b) / b for a, b in zip(f, old))
    print(f"(a) beam 2, Gamma={G}, N=90: max relative difference vs mpmath-1.4.1 run = {mp.nstr(d, 3)}")
old = {r["key"]: r for r in json.load(open(os.path.join(RES, "6_node_shift_magnitude.json")))}
for key in ("B1", "B1R", "B2", "BJ21", "BJ62"):
    p = os.path.join(RES, f"6c_node_fine_grid_{key}.json")
    if not os.path.exists(p):
        continue
    new = json.load(open(p))
    dm = 0.0
    for zo, zn in zip(old[key]["nodes"], new["nodes"]):
        assert (zo["mode"], zo["node"]) == (zn["mode"], zn["node"])
        dm = max(dm, abs(zo["xi0"] - zn["xi0"]))
        for G in ("0.25", "0.5", "6.0", "12.0", "20.0"):
            Go = G.rstrip("0").rstrip(".") if G.endswith(".0") else G
            dm = max(dm, abs(zo["shifts"][Go] - zn["shifts"][G]))
    print(f"(b) {key}: max |difference| of node positions/shifts between the two runs = {dm:.2e}")
