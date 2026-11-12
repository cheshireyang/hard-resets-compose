"""E30: typical number of pieces of hard-reset segment maps.
(i)  beta = 1, i.i.d. symmetric continuous inputs: E[1 + R_L] = 1 + (2L-1) C(2L-2,L-1) / 4^(L-1) (Sparre Andersen),
     for several input laws, including heavy-tailed ones; checked by Monte Carlo on the record counts and, for small L,
     against piece counts of exact segment maps computed in rational arithmetic.
(ii) asymmetric zero-mean inputs (not covered by the formula) for contrast.
usage: python3 experiments/e30_typical_pieces.py"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from fractions import Fraction as Fr
from lifscan.exact import segment_map

rng = np.random.default_rng(0)
LAWS = {"gauss": lambda s: rng.standard_normal(s), "uniform": lambda s: rng.uniform(-1, 1, s),
        "laplace": lambda s: rng.laplace(0, 1, s), "cauchy": lambda s: rng.standard_cauchy(s),
        "student_t1.5": lambda s: rng.standard_t(1.5, s)}
ASYM = {"exp_minus_1": lambda s: rng.exponential(1.0, s) - 1.0, "lognormal_centred": lambda s: rng.lognormal(0, 1, s) - math.exp(0.5)}


def formula(L):
    """E[1 + R_L] = 1 + sum_{j=0}^{L-1} C(2j,j)/4^j = 1 + (2L-1) C(2L-2,L-1)/4^(L-1)."""
    n = L - 1
    return 1 + (2 * n + 1) * math.exp(math.lgamma(2 * n + 1) - 2 * math.lgamma(n + 1) - 2 * n * math.log(2))


def records(S):
    """S: [n, L] partial sums; number of k = 1..L with S_k > max_{j<k} S_j (k = 1 always counts)."""
    run = np.maximum.accumulate(S, axis=1)
    prev = np.concatenate([np.full((S.shape[0], 1), -np.inf), run[:, :-1]], axis=1)
    return (S > prev).sum(1)


rows = []
LS = (1, 2, 3, 4, 8, 16, 64, 256, 1024, 4096)
for name, draw in list(LAWS.items()) + list(ASYM.items()):
    out = []
    for L in LS:
        n = 200000 if L <= 256 else (40000 if L <= 1024 else 10000)
        S = np.cumsum(draw((n, L)), axis=1)
        pieces = 1 + records(S)
        m, se = pieces.mean(), pieces.std() / math.sqrt(n)
        rows.append(dict(law=name, L=L, n=n, mean=float(m), se=float(se), formula=formula(L)))
        out.append(f"L={L}: {m:.3f}±{se:.3f} (f {formula(L):.3f}, z={(m - formula(L)) / se if se > 0 else 0:+.1f})")
    print(f"{name:18s}", " | ".join(out[:4]), "|", " | ".join(out[-3:]), flush=True)

# exact check: pieces of the exact segment map (rational arithmetic) equal 1 + #records, beta = 1
ok, n_checked = True, 0
for trial in range(300):
    L = int(rng.integers(1, 25))
    I = [Fr(int(x), 64) for x in rng.integers(-200, 201, L)]
    th, ur = Fr(1), Fr(int(rng.integers(-64, 64)), 64)
    M = segment_map(I, Fr(1), th, ur, merge=False)        # one piece per record region (Theorem 1)
    S = np.cumsum([float(x) for x in I])[None, :]
    # ties (S_k equal to an earlier maximum) do not create pieces: strict records only
    ok &= (M.pieces == 1 + int(records(S)[0])); n_checked += 1
print("exact segment maps: pieces == 1 + strict records in", n_checked, "random cases:", ok)
json.dump(dict(rows=rows, exact_check=dict(cases=n_checked, ok=bool(ok))), open("results/e30_typical_pieces.json", "w"), indent=1)
