"""E16: size of soft-reset chunk maps pruned to the reachable entry interval [l, h]
(l_t = min(0, beta*l + I_t), h_t = max(theta, beta*h + I_t - theta)), float64."""
import json, sys
import numpy as np

def soft_chunk_pieces(I, beta, lo, hi, theta=1.0):
    # map u -> A*u + c on pieces [x_i, x_{i+1}); start: one piece (-inf, inf), offset 0, slope 1
    xs = [-np.inf]; cs = [0.0]; A = 1.0
    for x in I:
        ct = (theta - x) / beta
        nx, nc = [], []
        for i, (xl, c) in enumerate(zip(xs, cs)):
            xr = xs[i + 1] if i + 1 < len(xs) else np.inf
            z = (ct - c) / A                       # spike iff u >= z on this piece
            if z <= xl:
                nx.append(xl); nc.append(beta * c + x - theta)
            elif z >= xr:
                nx.append(xl); nc.append(beta * c + x)
            else:
                nx += [xl, z]; nc += [beta * c + x, beta * c + x - theta]
        A *= beta
        # prune to [lo, hi]
        keep_x, keep_c = [], []
        for i, (xl, c) in enumerate(zip(nx, nc)):
            xr = nx[i + 1] if i + 1 < len(nx) else np.inf
            if xr <= lo or xl > hi:
                continue
            keep_x.append(max(xl, lo) if not keep_x else xl); keep_c.append(c)
        xs, cs = (keep_x, keep_c) if keep_x else ([-np.inf], [0.0])
    return len(xs)

rng = np.random.default_rng(0)
rows = []
for beta in (0.9, 0.95, 0.99):
    for mu, sd in ((0.15, 0.3), (0.05, 0.3), (0.3, 0.4)):
        T = 8192
        out = []
        for L in (16, 64, 256):
            cnt = []
            for rep in range(4):
                I = mu + sd * rng.standard_normal(T)
                l = h = 0.0; ls = [l]; hs = [h]
                for x in I:
                    l = min(0.0, beta * l + x); h = max(1.0, beta * h + x - 1.0); ls.append(l); hs.append(h)
                for m in range(0, T, L):
                    cnt.append(soft_chunk_pieces(I[m:m + L], beta, ls[m], hs[m]))
            cnt = np.array(cnt)
            rows.append(dict(beta=beta, mu=mu, sd=sd, L=L, mean=float(cnt.mean()), p99=float(np.quantile(cnt, 0.99)), max=int(cnt.max())))
            out.append(f"L={L}: {cnt.mean():.1f} (p99 {np.quantile(cnt, .99):.0f}, max {cnt.max()})")
        print(f"beta={beta} mu={mu} sd={sd}: ", " | ".join(out), flush=True)
json.dump(rows, open("results/e16_soft_pieces.json", "w"), indent=1)
