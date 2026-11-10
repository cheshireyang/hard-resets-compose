"""E13: piece counts of L-step segment maps for different reset rules r(v) (exact arithmetic).
Upper bound theory: constant r -> L+1; r strictly increasing with f_t(r(u)) >= r(f_t(u)) -> 1+L(L+1)/2.
We search (hill-climbing over inputs) for the maximum piece count, inside and outside the conditions."""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fractions import Fraction as Fr
from lifscan.exact import PWA, pwa_segment

TH = Fr(1)
def step(rule, I, beta):
    c = (TH - I) / beta
    g = Fr(1, 2)
    table = {
        "hard (r=0)":                    (Fr(0), Fr(0)),
        "soft (r=v-1)":                  (Fr(1), -TH),
        "proportional (r=v/2), I>=0":    (g, Fr(0)),
        "proportional (r=v/2), any I":   (g, Fr(0)),
        "affine (r=v/2-1/2)":            (g, -g),
        "decreasing (r=-v/2)":           (-g, Fr(0)),
    }
    a, d = table[rule]
    return PWA([c], [(beta, I), (a * beta, a * I + d)])

def pieces(rule, I, beta):
    return pwa_segment([step(rule, x, beta) for x in I]).pieces

def cond_ok(rule, I, beta):
    # check f_t(r(u)) >= r(f_t(u)) for affine pieces: (1-g)I >= (1-beta)d
    if rule.startswith("proportional"):
        return all(x >= 0 for x in I)
    return None

def search(rule, L, betas, lo, hi, reps, iters, rng):
    best, best_cfg = 0, None
    for beta in betas:
        for _ in range(reps):
            I = [Fr(rng.randint(lo, hi), 20) for _ in range(L)]
            c = pieces(rule, I, beta)
            for _ in range(iters):
                J = list(I); k = rng.randrange(L)
                J[k] = min(max(J[k] + Fr(rng.randint(-6, 6), 40), Fr(lo, 20)), Fr(hi, 20))
                cj = pieces(rule, J, beta)
                if cj >= c: I, c = J, cj
            if c > best: best, best_cfg = c, (float(beta), [float(x) for x in I])
    return best, best_cfg

rng = random.Random(5)
rows = []
for rule in ("hard (r=0)", "soft (r=v-1)", "proportional (r=v/2), I>=0", "proportional (r=v/2), any I",
             "affine (r=v/2-1/2)", "decreasing (r=-v/2)"):
    lo, hi = (0, 30) if "I>=0" in rule else (-20, 30)
    out = []
    for L in range(2, 9):
        b, cfg = search(rule, L, (Fr(1, 2), Fr(9, 10), Fr(99, 100)), lo, hi, 30, 40, rng)
        out.append(b)
        rows.append(dict(rule=rule, L=L, max=b, cfg=cfg))
    print(f"{rule:32s}", out, flush=True)
print("L+1        ", [L + 1 for L in range(2, 9)])
print("1+L(L+1)/2 ", [1 + L * (L + 1) // 2 for L in range(2, 9)])
print("2^L        ", [2 ** L for L in range(2, 9)])
json.dump(rows, open("results/e13_reset_classes.json", "w"), indent=1)
