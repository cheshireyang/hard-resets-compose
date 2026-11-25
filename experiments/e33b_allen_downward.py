"""E33b: is the reset 'downward' (hypothesis of Theorem alifdom) in the fitted Allen ALIF models?

For every cell of E28all, on the Noise 1 (train) and Noise 2 (test) sweeps at dt = 0.2 ms, with the fitted
(beta, gain, bias, u_r):
  * frac_suff: fraction of steps where the sufficient condition I_t >= (1-beta) u_r fails (I_t = gain*X_t + bias);
  * exact check of the hypothesis itself: a subthreshold trajectory started at u_r at any step never goes below u_r
    before it spikes.  The lowest such trajectory is the one that never spikes, and its minimum over all start times
    is the running minimum of a backward recursion, so we check  m_t = min(u_r, beta*m_{t-1} + I_t)  ... implemented
    directly as: g_t = min over s<t of the free state at t started from u_r at s, g_t = min(beta*g_{t-1} + I_t,
    beta*u_r + I_t) = beta*min(g_{t-1}, u_r) + I_t; the hypothesis holds on a sweep iff g_t >= u_r for all t.
    (Ignoring spikes only makes trajectories lower, so 'holds' is exact and 'fails' is conservative.)
CPU only; needs data/allen_all (experiments/allen_fetch_all.py).  Output: results/e33b_allen_downward.json
"""
import glob, json, os, sys
import numpy as np

try:
    from numba import njit
except ImportError:                       # plain Python fallback (slow but exact)
    def njit(f): return f


@njit
def min_gap(I, beta, ur):
    """min over t of g_t - u_r, with g_t = beta*min(g_{t-1}, u_r) + I_t (lowest free trajectory started at u_r)."""
    g = 1e300; m = 1e300
    for t in range(I.shape[0]):
        g = beta * min(g, ur) + I[t]
        if g - ur < m:
            m = g - ur
    return m


DT_MS = 0.2
DATA = os.environ.get("E33B_DATA", "data/allen_all")


def load_cell(path):
    z = np.load(path, allow_pickle=False)
    sweeps = sorted({k.split("__")[0] for k in z.files if "__" in k})
    out = {"Noise 1": [], "Noise 2": []}
    for s in sweeps:
        name = str(z[f"{s}__name"]); rate = float(z[f"{s}__rate"])
        i_pa = z[f"{s}__i_pa"]
        step = int(round(rate * DT_MS / 1000.0)); n = len(i_pa) // step
        if name in out:
            out[name].append(i_pa[: n * step].reshape(n, step).mean(1) / 100.0)     # 100 pA units, as in E15/E28
    return out


def main():
    fits = {}
    for f in sorted(glob.glob("results/e28all_allen_alif_w*.json")):
        for r in json.load(open(f)):
            if "beta" in r:
                fits[r["cell"]["specimen"]] = r
    rows = []
    for sid, r in sorted(fits.items()):
        path = f"{DATA}/{sid}.npz"
        if not os.path.exists(path):
            continue
        b, g, bias, ur = r["beta"], r["gain"], r["bias"], r["ur"]
        sw = load_cell(path)
        n_steps = n_suff_fail = 0; holds = True; min_pa = np.inf; worst_gap = np.inf
        for name in ("Noise 1", "Noise 2"):
            for x in sw[name]:
                I = g * x + bias
                n_steps += len(I); n_suff_fail += int((I < (1 - b) * ur).sum())
                min_pa = min(min_pa, float(x.min() * 100.0))
                m = min_gap(np.ascontiguousarray(I, dtype=np.float64), float(b), float(ur))
                worst_gap = min(worst_gap, m)
                holds &= m >= 0
        rows.append(dict(specimen=sid, steps=n_steps, frac_suff_fail=n_suff_fail / max(n_steps, 1), downward=bool(holds),
                         min_gap=float(worst_gap), min_stimulus_pA=min_pa, ur=ur, beta=b))
        print(json.dumps(rows[-1]), flush=True)
    summ = dict(cells=len(rows), downward=sum(r["downward"] for r in rows),
                suff_everywhere=sum(r["frac_suff_fail"] == 0 for r in rows),
                min_stimulus_pA=min((r["min_stimulus_pA"] for r in rows), default=None))
    print("summary", summ)
    json.dump(dict(summary=summ, rows=rows), open("results/e33b_allen_downward.json", "w"), indent=1)


if __name__ == "__main__":
    main()
