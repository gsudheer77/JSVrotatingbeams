"""Section 4.7 / Tables 6, 7: beam-2 extended convergence, N=50..90,
Gamma=0,6,12, modes 1,3,5 (and full Table 6 at N=80, Gamma=0..12)."""
import mpmath as mp
import chebyshev_rotating_beam as cb

mp.mp.dps = 34
mu, P = cb.wang_wereley(-0.8, [-0.95])  # beam 2

def table6():
    print("Table 6 (beam 2, N=80, Rbar=0): lambda_1..5")
    for G in range(13):
        f = cb.solve(80, G, mu, P, nmodes=5)
        print(G, [mp.nstr(x, 10) for x in f])

def table7():
    print("\nTable 7: relative change |l(N)-l(N+10)|/l(N+10), modes 1/3/5")
    for G in (0, 6, 12):
        print(f"Gamma={G}")
        prev = None
        for N in (50, 60, 70, 80, 90):
            f = cb.solve(N, G, mu, P, nmodes=5)
            if prev is not None:
                rel = [abs(prev[i] - f[i]) / f[i] for i in (0, 2, 4)]
                print(f"  {N-10}->{N}:", [mp.nstr(x, 3) for x in rel])
            prev = f

if __name__ == "__main__":
    table6()
    table7()
