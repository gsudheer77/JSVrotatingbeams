"""
9_refinement_and_rounding.py
==========================================================================
Refinement evidence and the rounding criterion for every benchmark table
(manuscript Section 4.3; Reviewer 2, comment 7).

For every tabulated entry the frequency is computed at the benchmark
resolution N_b and at N_b+10 and N_b+20 (34-digit arithmetic).  The
script records

  printed value: lam(N_p), N_p = N_b + 10 (finest-but-one resolution)
  delta = |lam(N_p) - lam(N_p+10)| / lam(N_p+10)    (final refinement step)
  r     = delta / delta_prev                         (observed contraction)
  E     = delta (1 + r/(1-r))  -- final step plus geometric-tail estimate of
          the truncation error beyond it.

Rounding criterion used for printing: an entry is printed to d
significant figures only if lam(N_p+10) and every value in the estimated
uncertainty interval [lam(N_p)(1-E), lam(N_p)(1+E)] round to the same
d-figure value as lam(N_p). E is an estimated bound that assumes the
observed geometric contraction continues; it is not a rigorous bound.

All raw values (30 significant digits) are written to
../results/9_refinement_raw_<table>.json so the analysis can be resumed.

Run time: dominated by the beam-2 N=100 solves (~2.5 min each with the
gmpy2 backend). Pass a table name to run a subset:
    python3 9_refinement_and_rounding.py T4 T5 T11 T9 T6 T8
==========================================================================
"""
import json, os, sys, time
from analysis_utils import *

def OUTF(name):
    return os.path.join(HERE, "..", "results", f"9_refinement_raw_{name}.json")

mu1, P1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
mu2, P2 = wang_wereley(-0.8, [-0.95])
mun2, Pn2 = banerjee(0.5, 2)
mu81, P81 = banerjee(0.8, 1)
mu82, P82 = banerjee(0.8, 2)

TABLES = {
    # name: list of (label, mu, P, Rbar, Gamma, nmodes, N_b)
    "T4":  [("B1 G=%g" % g, mu1, P1, 0, g, 10, 50) for g in (0, 2, 4, 5, 6, 8, 10, 12)],
    "T5":  [("n2c05 G=%g" % g, mun2, Pn2, 0, g, 5, 50) for g in (0, 2, 4, 6, 8, 10)],
    "T9":  [("c08n1 G=%g" % g, mu81, P81, 0, g, 5, 50) for g in (5, 10)]
         + [("c08n2 G=%g" % g, mu82, P82, 0, g, 5, 50) for g in (5, 10)],
    "T6":  [("B2 G=%g" % g, mu2, P2, 0, g, 5, 80) for g in range(0, 13)],
    # additional refinement for the one beam-2 entry (Gamma=8, mode 3) that fails the
    # interval criterion at the printed 11 figures with N_p = 90: N_p = 100, check at 110
    "T6x": [("B2 G=8", mu2, P2, 0, 8, 5, 90)],
    "T8":  [("B2 G=1 R=1", mu2, P2, 1, 1, 3, 80),
            ("B2 G=1 R=5", mu2, P2, 5, 1, 3, 80),
            ("B2 G=5 R=1", mu2, P2, 1, 5, 3, 80)],
}


PRINT_FIGS = {"T4": 12, "T5": 12, "T6": 11, "T6x": 11, "T8": 11}


def load(name):
    if os.path.exists(OUTF(name)):
        with open(OUTF(name)) as f:
            return json.load(f)
    return {}


def save(name, d):
    tmp = OUTF(name) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, OUTF(name))


def run(names):
    for name in names:
        data = load(name)
        for (label, mu, P, Rbar, G, nm, Nb) in TABLES[name]:
            key = f"{name}|{label}"
            rec = data.get(key, {"Nb": Nb, "vals": {}})
            for N in (Nb, Nb + 10, Nb + 20):
                if str(N) in rec["vals"]:
                    continue
                t = time.time()
                f = solve(N, G, mu, P, Rbar=Rbar, nmodes=nm)
                rec["vals"][str(N)] = [mp.nstr(x, 30) for x in f]
                rec["t_" + str(N)] = time.time() - t
                data[key] = rec
                save(name, data)
                print(f"{key} N={N} done in {time.time()-t:.1f}s", flush=True)


def sig_round(x, d):
    return mp.nstr(mp.mpf(x), d, strip_zeros=False)


def analyse(names):
    """Certification policy (manuscript Section 4.3): the printed value is
    lambda(N_p), N_p = N_b + 10 (the finest-but-one resolution computed);
    delta = |lambda(N_p) - lambda(N_p+10)|/lambda(N_p+10), the contraction
    r = delta / delta_prev is taken from the three resolutions, and the
    estimated truncation-error bound (assuming continued geometric
    contraction) is E = delta (1 + r/(1-r)); when both changes are already at
    the finite-precision floor (delta_prev <= 1e-19, no contraction) E is
    the larger of the two changes.  An entry is printed to d
    figures only if lambda(N_p) and lambda(N_p+10) round to the same
    d-figure value as every value in the interval [lam(1-E), lam(1+E)]."""
    for name in names:
        data = load(name)
        print(f"\n=== {name} ===")
        worst = {}
        for (label, mu, P, Rbar, G, nm, Nb) in TABLES[name]:
            rec = data.get(f"{name}|{label}")
            if not rec or len(rec["vals"]) < 3:
                print(label, "incomplete"); continue
            vm, v0, v1 = [[mp.mpf(s) for s in rec["vals"][str(N)]] for N in (Nb, Nb + 10, Nb + 20)]
            print(f"{label}  (printed at N_p={Nb+10}):")
            for i in range(nm):
                dprev = abs(vm[i] - v0[i]) / v0[i]
                d = abs(v0[i] - v1[i]) / v1[i]
                r = d / dprev if dprev > 0 else mp.mpf(0)
                if r < 1 and dprev > mp.mpf("1e-19"):
                    E = d * (1 + r / (1 - r))
                else:   # both changes at the finite-precision floor (~1e-21): use the larger change
                    E = max(d, dprev)
                dmax = 0
                for dd in range(4, 19):
                    ref = sig_round(v0[i], dd)
                    if all(sig_round(x, dd) == ref for x in (v1[i], v0[i] * (1 - E), v0[i] * (1 + E))):
                        dmax = dd
                print(f"   mode {i+1}: delta={mp.nstr(d,2)} r={mp.nstr(r,2)} E={mp.nstr(E,2)} -> {dmax} figs: "
                      f"{sig_round(v0[i], max(dmax, 4))}")
                w = worst.setdefault(i + 1, [0, 0, 99])
                w[0] = max(w[0], d); w[1] = max(w[1], E); w[2] = min(w[2], dmax)
        pf = PRINT_FIGS.get(name)
        if pf:
            fails = []
            for (label, mu, P, Rbar, G, nm, Nb) in TABLES[name]:
                rec = data.get(f"{name}|{label}")
                vm, v0, v1 = [[mp.mpf(s) for s in rec["vals"][str(N)]] for N in (Nb, Nb + 10, Nb + 20)]
                for i in range(nm):
                    dprev = abs(vm[i] - v0[i]) / v0[i]; d = abs(v0[i] - v1[i]) / v1[i]
                    r = d / dprev if dprev > 0 else mp.mpf(0)
                    E = d * (1 + r / (1 - r)) if (r < 1 and dprev > mp.mpf("1e-19")) else max(d, dprev)
                    ref = sig_round(v0[i], pf)
                    if not all(sig_round(x, pf) == ref for x in (v1[i], v0[i] * (1 - E), v0[i] * (1 + E))):
                        fails.append(f"{label} mode {i+1}")
            print(f" test at the PRINTED precision ({pf} figures): "
                  f"{'all entries pass' if not fails else 'FAIL: ' + ', '.join(fails)}")
        print(f" column summary for {name} (mode: max delta, max E, min printable figs):")
        for m, (a, b, c) in sorted(worst.items()):
            print(f"   mode {m}: max delta={mp.nstr(a,2)}  max E={mp.nstr(b,2)}  min figs={c}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["T4", "T5", "T9", "T6", "T8"]
    names = [n for n in names if n in TABLES]
    if os.environ.get("ANALYSE_ONLY") != "1":
        run(names)
    analyse(names)
