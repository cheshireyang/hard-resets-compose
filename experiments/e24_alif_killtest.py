"""E24: go/no-go test for an exact two-level scan of adaptive-threshold (ALIF) neurons.  CPU, float64.

ALIF:  v = beta*u + I_t ;  s = [v >= theta0 + w] ;  u <- u_r if s else v ;  w <- rho*w + q*s.

A chunk map (u0, w0) -> (u_C, w_C) is represented as in App. D of the paper:
  * the no-spike region (a convex polygon in the entry box), on which the map is affine;
  * for every first-spike step k, a list of intervals of x = w_k (the adaptation right after that spike); on each
    interval the final u is a constant and the final w is affine in x.
The size of this representation ("pieces") is what a GPU kernel would have to store and update at every step, so
it decides whether a scan can beat the serial kernel.  It depends on the entry box the map is pruned to.

We measure, for chunks of real trajectories,
  (1) pieces as a function of the half-width dw of the box in w (box centred on the true entry state; u-range =
      everything allowed by the lower envelope and the threshold),
  (2) pieces for the only rigorous box we can compute by a scan (w between "no spike so far" and "a spike at
      every step"),
  (3) how well the entry w can be predicted without the serial pass: (a) from the previous iterate of an
      optimisation loop (same input scaled by 1+eps), (b) from a warm-up of W chunks started at w = 0 or at the
      mean w.
Every map whose box contains the true entry state is checked against the serial end state.
usage: python3 experiments/e24_alif_killtest.py [T] [nproc]"""
import sys, os, json, math, time
import numpy as np
from multiprocessing import Pool

TH0, UR = 1.0, 0.0
DW = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)          # box half-widths in units of q


def serial(I, beta, rho, q, u=UR, w=0.0):
    T = len(I); U = np.empty(T + 1); W = np.empty(T + 1); S = np.zeros(T, dtype=bool)
    U[0], W[0] = u, w
    for t in range(T):
        v = beta * u + I[t]
        if v >= TH0 + w:
            u = UR; w = rho * w + q; S[t] = True
        else:
            u = v; w = rho * w
        U[t + 1], W[t + 1] = u, w
    return U, W, S


def clip(poly, A, B, Cc, keep_pos):
    """Convex polygon [(u, w)] cut by the half-plane A*u + B*w + Cc >= 0 (or <= 0)."""
    n = len(poly)
    if n == 0: return []
    sg = 1.0 if keep_pos else -1.0
    vals = [sg * (A * p[0] + B * p[1] + Cc) for p in poly]
    out = []
    for i in range(n):
        p, r = poly[i], poly[(i + 1) % n]; a, b = vals[i], vals[(i + 1) % n]
        if a >= 0: out.append(p)
        if (a > 0 and b < 0) or (a < 0 and b > 0):
            t = a / (a - b); out.append((p[0] + t * (r[0] - p[0]), p[1] + t * (r[1] - p[1])))
    return out


def area(poly):
    s = 0.0
    for i in range(len(poly)):
        p, r = poly[i], poly[(i + 1) % len(poly)]
        s += p[0] * r[1] - r[0] * p[1]
    return abs(s) / 2


def build(Ic, beta, rho, q, lu, hu, lw, hw):
    """Chunk map pruned to the entry box [lu,hu] x [lw,hw].  Returns (pieces, has_nospike_region, peak size)."""
    poly = [(lu, lw), (hu, lw), (hu, hw), (lu, hw)]
    eps = 1e-13 * max(1.0, (hu - lu) * (hw - lw))
    pieces = []                                   # (k, x_lo, x_hi, u, a, off):  w_now = a*x + off
    A, b, R = 1.0, 0.0, 1.0                       # beta^j, b_j, rho^(j-1)
    peak = 1
    for j, I in enumerate(Ic, start=1):
        new = []
        for (k, xl, xr, u, a, off) in pieces:
            v = beta * u + I
            z = (v - TH0 - off) / a               # spike iff x <= z
            if z >= xr:   new.append((k, xl, xr, UR, rho * a, rho * off + q))
            elif z < xl:  new.append((k, xl, xr, v, rho * a, rho * off))
            else:
                new.append((k, xl, z, UR, rho * a, rho * off + q)); new.append((k, z, xr, v, rho * a, rho * off))
        pieces = new
        A *= beta; b = beta * b + I               # free trajectory v_j = A*u0 + b, threshold theta0 + R*w0
        if poly:
            sp = clip(poly, A, -R, b - TH0, True)
            if len(sp) >= 3 and area(sp) > eps:
                ws = [p[1] for p in sp]
                pieces.append((j, R * rho * min(ws) + q, R * rho * max(ws) + q, UR, 1.0, 0.0))
            poly = clip(poly, A, -R, b - TH0, False)
            if len(poly) < 3 or area(poly) <= eps: poly = []
        R *= rho
        peak = max(peak, len(pieces) + (1 if poly else 0))
    return pieces, bool(poly), peak


def evalmap(pieces, Ic, beta, rho, q, u0, w0):
    A, b, R = 1.0, 0.0, 1.0
    for j, I in enumerate(Ic, start=1):
        A *= beta; b = beta * b + I
        if A * u0 + b >= TH0 + R * w0:
            x = R * rho * w0 + q
            for (k, xl, xr, u, a, off) in pieces:
                if k == j and xl - 1e-9 <= x <= xr + 1e-9: return u, a * x + off
            return None
        R *= rho
    return A * u0 + b, R * w0


def run(job):
    name, beta, rho, q, m_free, seed, C, T = job
    rng = np.random.default_rng(seed)
    sig = 0.5
    WARM = 2048
    I = (1 - beta) * m_free + sig * math.sqrt(1 - beta * beta) * rng.standard_normal(T + WARM)
    U, W, S = serial(I, beta, rho, q)
    # lower envelope on u (valid for ALIF as for LIF) and predictors of the entry w
    l = np.empty(T + WARM + 1); l[0] = UR
    for t in range(T + WARM): l[t + 1] = min(UR, beta * l[t] + I[t])
    pred = {}
    for e in (0.005, 0.02):
        pred[f"prev_iter_{e}"] = serial((1 + e) * I, beta, rho, q)[1]
    wbar = float(W[WARM:].mean())
    starts = list(range(WARM, WARM + T, C))
    for tag, w_init in (("cold", 0.0), ("mean", wbar)):
        for Wn in (1, 2, 4):
            arr = np.full(T + WARM + 1, np.nan)
            for t0 in starts:
                arr[t0] = serial(I[t0 - Wn * C:t0], beta, rho, q, UR, w_init)[1][-1]
            pred[f"warm_{tag}_W{Wn}"] = arr
    out = dict(name=name, beta=beta, rho=rho, q=q, m_free=m_free, seed=seed, C=C, T=T, rate=float(S[WARM:].mean()),
               w_mean=wbar, w_std=float(W[WARM:].std()), n_chunks=len(starts), boxes={}, pred={})
    stats = {}
    def add(key, P, peak, n1):
        stats.setdefault(key, []).append((P, peak, n1))
    bad = 0; checked = 0
    for t0 in starts:
        Ic = I[t0:t0 + C].tolist(); us, ws = float(U[t0]), float(W[t0])
        specs = [(f"oracle_{d}", ws, d * q) for d in DW] + [(f"previter_{d}", float(pred["prev_iter_0.005"][t0]), d * q) for d in DW]
        w_hi_rig = q * (1 - rho ** t0) / (1 - rho)
        specs.append(("rigorous", w_hi_rig / 2, w_hi_rig / 2))
        for key, cw, dw in specs:
            lw, hw = max(cw - dw, 0.0), cw + dw
            lu, hu = float(l[t0]) - 1e-9, max(UR, TH0 + hw / rho) + 1e-9
            pieces, has, peak = build(Ic, beta, rho, q, lu, hu, lw, hw)
            add(key, len(pieces) + has, peak, len({p[0] for p in pieces}))
            if lw <= ws <= hw:
                r = evalmap(pieces, Ic, beta, rho, q, us, ws); checked += 1
                if r is None or abs(r[0] - U[t0 + C]) > 1e-6 or abs(r[1] - W[t0 + C]) > 1e-6: bad += 1
    for key, v in stats.items():
        a = np.array(v, dtype=float)
        out["boxes"][key] = dict(mean=float(a[:, 0].mean()), p95=float(np.percentile(a[:, 0], 95)), max=float(a[:, 0].max()),
                                 peak_mean=float(a[:, 1].mean()), peak_max=float(a[:, 1].max()), first_spike_mean=float(a[:, 2].mean()))
    idx = np.array(starts)
    for key, arr in pred.items():
        err = np.abs(arr[idx] - W[idx]) / q
        out["pred"][key] = dict(q50=float(np.percentile(err, 50)), q90=float(np.percentile(err, 90)), q95=float(np.percentile(err, 95)),
                                q99=float(np.percentile(err, 99)), hit={str(d): float((err <= d).mean()) for d in DW})
    out["map_checks"] = checked; out["map_mismatch"] = bad
    return out


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    # (name, beta, rho, q): LSNN-like slow strong adaptation, GLIF2-like fast weak adaptation, and the paper's App. D sets
    neurons = [("lsnn", 0.95, 0.995, 0.9), ("glif2", 0.99, 0.96, 0.2), ("appD-a", 0.9, 0.95, 0.5), ("appD-b", 0.99, 0.99, 0.25)]
    jobs = [(n, b, r, q, m, seed, C, T) for (n, b, r, q) in neurons for m in (0.5, 1.0, 2.0) for seed in (0, 1) for C in (64, 128, 256)]
    t0 = time.time(); rows = []
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(run, jobs):
            rows.append(r)
            b = r["boxes"]
            print(f"{r['name']:7s} m={r['m_free']:<4} C={r['C']:<3} seed={r['seed']} rate={100 * r['rate']:5.2f}% | pieces oracle dw/q=0.01:{b['oracle_0.01']['mean']:6.1f} "
                  f"0.1:{b['oracle_0.1']['mean']:6.1f} 1:{b['oracle_1.0']['mean']:7.1f} rigorous:{b['rigorous']['mean']:8.1f} | prev-iter err/q q95={r['pred']['prev_iter_0.005']['q95']:.3f} "
                  f"| mismatches {r['map_mismatch']}/{r['map_checks']}", flush=True)
            json.dump(sorted(rows, key=lambda r: (r["name"], r["m_free"], r["C"], r["seed"])), open("results/e24_alif_killtest.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")
