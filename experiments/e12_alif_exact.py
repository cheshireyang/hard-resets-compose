"""E12: exact (rational) analysis of the ALIF segment map after a reset.
After the first spike, u = u_r and the remaining dynamics depend only on the adaptation w, so the
map is a 1-D piecewise-affine map Psi(w).  We track the w-intervals with constant spike pattern
exactly and count pieces; we also test the monotonicity lemma
   w < w'  =>  N_t(w) >= N_t(w')  and  (N_t(w) = N_t(w') => w_t(w) < w_t(w')).
ALIF:  v = beta*u + I_t ; s = [v >= theta0 + w_{t-1}] ; u = u_r if s else v ; w = rho*w + q*s."""
import json, random, sys
from fractions import Fraction as Fr

def psi_pieces(I, beta, rho, q, theta0=Fr(1), u_r=Fr(0), wlo=Fr(0), whi=Fr(4)):
    # intervals: (lo, hi, a, b, u, n)  with w_t = a*w + b on [lo, hi), u constant, n = spike count
    iv = [(wlo, whi, Fr(1), Fr(0), u_r, 0)]
    for x in I:
        new = []
        for lo, hi, a, b, u, n in iv:
            v = beta * u + x
            # spike iff a*w + b <= v - theta0  <=>  w <= (v - theta0 - b)/a
            z = (v - theta0 - b) / a
            if z >= lo:          # spiking part [lo, min(hi, z')] (closed at z: equality fires)
                top = min(hi, z)
                if top > lo or (top == lo and z >= lo):
                    new.append((lo, min(hi, z), rho * a, rho * b + q, u_r, n + 1))
            if z < hi:
                start = max(lo, z)
                new.append((start, hi, rho * a, rho * b, v, n))
        # merge adjacent identical
        new.sort(key=lambda r: (r[0], r[1]))
        merged = []
        for r in new:
            if r[1] <= r[0] and not (r[0] == r[1]):
                continue
            if merged and merged[-1][2:] == r[2:] and merged[-1][1] >= r[0]:
                m = merged[-1]; merged[-1] = (m[0], max(m[1], r[1])) + m[2:]
            else:
                merged.append(r)
        iv = [r for r in merged if r[1] > r[0]]
    return iv

def simulate(I, beta, rho, q, w, theta0=Fr(1), u_r=Fr(0)):
    u, n = u_r, 0
    for x in I:
        v = beta * u + x
        s = v >= theta0 + w
        u = u_r if s else v
        w = rho * w + q * (1 if s else 0); n += s
    return n, w, u

random.seed(0)
rows = []
viol = 0; checks = 0
for beta, rho, q in ((Fr(9, 10), Fr(95, 100), Fr(1, 2)), (Fr(9, 10), Fr(95, 100), Fr(1)), (Fr(99, 100), Fr(99, 100), Fr(1, 4))):
    for L in (4, 8, 16, 32, 64, 128):
        cnts = []; tots = []
        for rep in range(20):
            I = [Fr(round(random.gauss(0.3, 0.4) * 100), 100) for _ in range(L)]
            iv = psi_pieces(I, beta, rho, q)
            cnts.append(len(iv))
            if L <= 64:   # exact bound on the 2-D segment map: 1 + sum_k |Psi_k| over w in [0, 4]
                tot = 1 + sum(len(psi_pieces(I[k:], beta, rho, q)) for k in range(1, L + 1))
                tots.append(tot)
            if L <= 32:   # monotonicity lemma on a grid
                ws = sorted(set(Fr(random.randint(0, 4000), 1000) for _ in range(60)))
                prev = None
                for w in ws:
                    n, wt, _ = simulate(I, beta, rho, q, w)
                    if prev is not None:
                        checks += 1
                        if n > prev[0] or (n == prev[0] and wt <= prev[1]):
                            viol += 1
                    prev = (n, wt)
        rows.append(dict(beta=float(beta), rho=float(rho), q=float(q), L=L, mean=sum(cnts) / len(cnts), max=max(cnts),
                         bound2d_mean=(sum(tots) / len(tots) if tots else None)))
        print(json.dumps(rows[-1]), flush=True)
print("monotonicity checks", checks, "violations", viol)
json.dump(dict(rows=rows, mono_checks=checks, mono_violations=viol), open("results/e12_alif_exact.json", "w"), indent=1)
