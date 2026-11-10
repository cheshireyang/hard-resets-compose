"""E8: how many affine pieces does the L-step map of an adaptive-threshold (ALIF) neuron have on a
2-D window of initial states (u0, w0)?  Pieces = distinct spike patterns (the map is affine on each).
Estimated by dense sampling (a lower bound: thin regions can be missed).  Hard-reset LIF (q = 0,
threshold independent of w) is the 1-D reference.
  v_t = beta*u + I_t ;  s_t = [v_t >= theta0 + w_{t-1}] ;  u_t = u_r if s_t else v_t ;  w_t = rho*w + q*s_t"""
import json, os, sys
import numpy as np
rng = np.random.default_rng(0)
G = 700
u0 = np.linspace(-4, 1.0, G)[:, None] * np.ones((1, G))
w0 = np.ones((G, 1)) * np.linspace(0.0, 2.0, G)[None, :]
rows = []
for q, rho, name in ((0.0, 0.0, "LIF"), (0.5, 0.95, "ALIF q=0.5"), (1.0, 0.95, "ALIF q=1.0")):
    for L in (4, 8, 16, 32, 64):
        res = []
        for rep in range(5):
            I = 0.15 + 0.4 * rng.standard_normal(L)
            u, w = u0.copy(), w0.copy() if q > 0 else np.zeros_like(w0)
            key = np.zeros((G, G), dtype=np.uint64); h = np.zeros((G, G), dtype=np.uint64)
            pats = []
            for t in range(L):
                v = 0.9 * u + I[t]
                s = v >= 1.0 + w
                u = np.where(s, 0.0, v); w = rho * w + q * s
                pats.append(s)
            P = np.stack(pats, -1)
            n = len(np.unique(P.reshape(-1, L), axis=0))
            res.append(n)
        rows.append(dict(model=name, L=L, mean=float(np.mean(res)), max=int(np.max(res))))
        print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open("results/e8_alif_regions.json", "w"), indent=1)
