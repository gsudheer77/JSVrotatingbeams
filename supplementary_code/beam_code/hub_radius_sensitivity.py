"""Section 4.8 / Table 8: hub-radius cases and logarithmic sensitivity
S_Rbar^(i) = (Rbar/lambda_i) d(lambda_i)/d(Rbar), computed by numerical
differentiation of the solver output (mp.diff) rather than finite
difference between two published rows."""
import mpmath as mp
import chebyshev_rotating_beam as cb

mp.mp.dps = 34
mu, P = cb.wang_wereley(-0.8, [-0.95])  # beam 2

def table8():
    print("Table 8: (Gamma,Rbar) -> lambda_1,2,3")
    cases = [(1, 1, 90), (1, 5, 80), (5, 1, 80)]
    for G, R, N in cases:
        f = cb.solve(N, G, mu, P, Rbar=R, nmodes=3)
        print(f"  (G={G},R={R}) N={N}:", [mp.nstr(x, 10) for x in f])

def sensitivity(N=40, nmodes=3):
    print("\nLogarithmic sensitivity S_Rbar^(i) at Gamma=1, Rbar in {1,3,5}")
    for R in (1, 3, 5):
        def li(Rv, i):
            return cb.solve(N, 1, mu, P, Rbar=Rv, nmodes=nmodes)[i]
        row = []
        for i in range(nmodes):
            lam = li(R, i)
            dlam = mp.diff(lambda Rv: li(Rv, i), R)
            S = mp.mpf(R) / lam * dlam
            row.append(S)
        print(f"  Rbar={R}:", [mp.nstr(x, 6) for x in row])

if __name__ == "__main__":
    table8()
    sensitivity()
