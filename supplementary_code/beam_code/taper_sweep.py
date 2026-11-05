"""Section 4.9: taper-ratio sweep (Banerjee power-law classes n=1,2),
c in {0.1,0.2,0.4,0.6,0.8}, required-N convergence, validated against
Banerjee-Su-Jackson published values where quoted in the manuscript."""
import mpmath as mp
import chebyshev_rotating_beam as cb

mp.mp.dps = 34

def check_low_taper():
    print("Validation: c=0.1, n=2, Gamma=10, N=25 (paper quotes 11.30167373, "
          "32.95542371, 71.84051488 vs published 11.3017, 32.9554, 71.8405)")
    mu, P = cb.banerjee(0.1, 2)
    f = cb.solve(25, 10, mu, P, nmodes=3)
    print([mp.nstr(x, 10) for x in f])

def table9():
    print("\nTable 9: convergence at c=0.8")
    cases = [("n=1,G=5,mode1", 1, 5, 0), ("n=1,G=10,mode3", 1, 10, 2),
             ("n=2,G=5,mode1", 2, 5, 0), ("n=2,G=10,mode3", 2, 10, 2)]
    for label, n, G, midx in cases:
        mu, P = cb.banerjee(0.8, n)
        print(label)
        for N in (25, 40, 50, 60):
            f = cb.solve(N, G, mu, P, nmodes=midx + 1)
            print(f"  N={N}: {mp.nstr(f[midx], 9)}")

def required_N_sweep():
    """Required N to reach 1e-9 relative-change convergence for mode 1,
    as a function of taper ratio c, both classes, Gamma=0."""
    print("\nRequired-N sweep (mode 1, Gamma=0, converged to 1e-9 relative change)")
    for n in (1, 2):
        print(f"n={n}")
        for c in (0.1, 0.2, 0.4, 0.6, 0.8):
            mu, P = cb.banerjee(c, n)
            prev = None
            for N in range(15, 90, 5):
                f = cb.solve(N, 0, mu, P, nmodes=1)[0]
                if prev is not None and abs(f - prev) / f < mp.mpf('1e-9'):
                    print(f"  c={c}: N_req={N}, lambda_1={mp.nstr(f,9)}")
                    break
                prev = f
            else:
                print(f"  c={c}: not converged by N=85")

if __name__ == "__main__":
    check_low_taper()
    table9()
    required_N_sweep()
