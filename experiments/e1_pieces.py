"""E1: piece counts of segment maps, hard vs soft reset (exact rational arithmetic)."""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fractions import Fraction as Fr
from lifscan.exact import *

random.seed(0)
theta = Fr(1); out = {}

# (a) tightness of the hard bound: constant input I in ((1-b)th, (1-b)th/(1-b^L))
tight = {}
for beta in (Fr(1, 2), Fr(9, 10), Fr(99, 100)):
    for L in (4, 16, 64):
        lo = (1 - beta) * theta; hi = lo / (1 - beta ** L)
        I = [(lo + hi) / 2] * L
        M = segment_map(I, beta, theta, Fr(0))
        tight[f"beta={beta},L={L}"] = (M.pieces, L + 1)
        assert M.pieces == L + 1
out["hard_tightness"] = tight

# (b) random inputs: hard vs soft pieces vs L for several drives
rows = []
def q(x): return Fr(round(x * 1000), 1000)
for beta in (Fr(1, 2), Fr(9, 10), Fr(99, 100)):
    for mu in (-0.05, 0.05, 0.15, 0.3):
        for L in (4, 8, 16, 32, 64):
            hs, ss, rec = [], [], []
            for rep in range(20):
                I = [q(random.gauss(mu, 0.3)) for _ in range(L)]
                hs.append(segment_map(I, beta, theta, Fr(0)).pieces)
                if L <= 32:
                    ss.append(pwa_segment([pwa_step_soft(x, beta, theta) for x in I]).pieces)
                rec.append(1 + sum(1 for x in I if x > (1 - beta) * theta))
            rows.append(dict(beta=float(beta), mu=mu, L=L,
                             hard_mean=sum(hs)/len(hs), hard_max=max(hs),
                             soft_mean=(sum(ss)/len(ss) if ss else None), soft_max=(max(ss) if ss else None),
                             drive_count_mean=sum(rec)/len(rec)))
            r = rows[-1]
            print(f"beta={r['beta']:<5} mu={mu:<5} L={L:<3} hard {r['hard_mean']:6.2f} (max {r['hard_max']:3d}, bound {L+1:3d})  "
                  f"soft {r['soft_mean'] if r['soft_mean'] is None else round(r['soft_mean'],2)} (max {r['soft_max']}, bound {1+L*(L+1)//2})  "
                  f"1+#[I>(1-b)th] {r['drive_count_mean']:.2f}", flush=True)
out["random"] = rows
json.dump(out, open("results/e1_pieces.json", "w"), indent=1, default=str)
