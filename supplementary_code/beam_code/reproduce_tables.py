"""
reproduce_tables.py
Regenerate the principal benchmark tables of the paper from the solver.
Run:  python3 reproduce_tables.py
(Severe-beam N=80 rows take ~1 min each; reduce Ns for a quick check.)

Table numbers below match the manuscript's current numbering (Tables 4,
5, 6, 9). Values are printed at 30 significant digits, matching the
34-digit working precision used throughout (mp.mp.dps = 34): the last
2-4 digits of a 30-digit print are not independently meaningful for
every case (they carry residual round-off from the assembly/eigensolve
chain), but they are not truncated away either, consistent with the
manuscript's high-precision (30-significant-digit) output-table statement
The manuscript itself only ever prints as many figures as N-refinement
verifies as converged (see Table 2 and Section 4.2) -- that convergence
bookkeeping, not this script's print width, is what determines how many
of these digits are physically meaningful for a given N.

Tables 4 and 5 are generated at N=50, the resolution used as the
converged reference for beam-1-scale cases elsewhere in this package
(see per_case_verification.py and fem_comparison.py), matching the
manuscript's printed benchmark digits. TABLE_4_5_N below can be
changed to reproduce these tables at other resolutions.
"""
TABLE_4_5_N = 50
import mpmath as mp
from chebyshev_rotating_beam import solve, wang_wereley, banerjee
mp.mp.dps = 34

def row(vals, w=34):
    return "  ".join(mp.nstr(v, 30).ljust(w) for v in vals)

print("=" * 74)
print(f"Table 4  -  Beam 1  m=m0(1-0.5xi), EI=EI0(1-0.5xi)^3,  N={TABLE_4_5_N},  Rbar=0")
print("Gamma   lambda_1        lambda_2        lambda_3")
b1 = wang_wereley(-0.5, [-1.5, 0.75, -0.125])
for G in (0, 2, 4, 6, 8, 10, 12):
    print(f"{G:<5}", row(solve(TABLE_4_5_N, G, *b1, nmodes=3)))

print("=" * 74)
print(f"Table 5  -  Banerjee n=2, c=0.5,  N={TABLE_4_5_N},  Rbar=0")
print("Gamma   lambda_1 .. lambda_5")
mu, P = banerjee(0.5, 2)
for G in (0, 2, 4, 6, 8, 10):
    print(f"{G:<5}", row(solve(TABLE_4_5_N, G, mu, P, nmodes=5)))

print("=" * 74)
print("Table 6  -  Severe beam 2  m=m0(1-0.8xi), EI=EI0(1-0.95xi),  N=80")
print("(converged reference; each row ~1 min)")
print("Gamma   lambda_1 .. lambda_5")
b2 = wang_wereley(-0.8, [-0.95])
for G in (0, 4, 8, 12):          # subset; extend to range(13) for the full table
    print(f"{G:<5}", row(solve(80, G, *b2, nmodes=5)))

print("=" * 74)
print("Table 9  -  Resolution of c=0.8 (n=2, Gamma=10, mode 3)")
mu, P = banerjee(0.8, 2)
for N in (25, 40, 50, 60):
    f = solve(N, 10, mu, P, nmodes=3)
    print(f"  N={N:<3} mode3 = {mp.nstr(f[2], 30)}   (DSM: 50.4956)")
