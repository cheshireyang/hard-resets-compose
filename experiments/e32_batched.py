"""E32: batched serial fitting of the whole Allen database -- the throughput baseline for Sec. 'fit'.

Instead of one fit per process (E15all / E28all, scans or serial kernels, several processes under MPS), all cells are
stacked into one [T, N] batch per recording length, every column carrying the parameters of its (cell, restart) unit,
and fitted with the per-column serial kernels of lifscan.triton_batch.  Adam is elementwise, so this is the same
optimisation as the independent per-cell fits (same initialisation, loss, learning rate and iterations); only the
rounding differs.  The van Rossum filter is a fused serial exponential filter (the per-cell runs use an FFT; same
filter in exact arithmetic).

  MODEL=hard : E15 protocol -- hard-reset LIF (beta, gain, bias, u_r), 3 restarts, 600 Adam steps, rate-matched init.
  MODEL=alif : E28 protocol -- ALIF warm-started from the per-cell hard fits (E15all), 4 (q, rho) inits, 400 steps.
Outputs per-cell held-out F1 (Noise 2), the wall-clock of the whole database, the time per batched iteration, and,
(Per-cell iteration times with scan and serial kernels: E15t / E28t.)
usage: MODEL=hard python3 experiments/e32_batched.py"""
import sys, os, json, glob, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import triton, triton.language as tl
_argv = sys.argv[:]
sys.argv = [sys.argv[0], "0.2", "1", "1"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e15_allen_fit.py")).read().split("rows = []")[0])
sys.argv = _argv
from lifscan import triton_batch as B
from lifscan import triton_alif as A

MODEL = os.environ.get("MODEL", "hard")
DATA = os.environ.get("E32_DATA", "data/allen_all")
GROUP = int(os.environ.get("E32_GROUP", "2048"))          # columns per kernel call (memory)
ITERS = int(os.environ.get("E32_ITERS", "600" if MODEL == "hard" else "400"))
LIMIT = int(os.environ.get("E32_LIMIT", "0"))
INITS = ((0.05, 0.99), (0.2, 0.99), (0.05, 0.998), (0.2, 0.998))
OUT = os.environ.get("E32_OUT", f"results/e32_batched_{MODEL}.json")


# ----------------------------------------------------------------------------- fused exponential filter y_t = a y_{t-1} + s_t
@triton.jit
def expf_kernel(X, Y, N, T, a, REV: tl.constexpr, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN); m = n < N
    y = tl.zeros([BN], dtype=tl.float32)
    for i in range(T):
        t = T - 1 - i if REV else i
        y = a * y + tl.load(X + t * N + n, mask=m, other=0.0)
        tl.store(Y + t * N + n, y, mask=m)


class ExpFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        x = x.contiguous(); y = torch.empty_like(x); T, N = x.shape
        expf_kernel[(triton.cdiv(N, 128),)](x, y, N, T, a, REV=False, BN=128)
        ctx.a = a
        return y

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous(); out = torch.empty_like(g); T, N = g.shape
        expf_kernel[(triton.cdiv(N, 128),)](g, out, N, T, ctx.a, REV=True, BN=128)
        return out, None


def load_db():
    base = {}
    for f in glob.glob("results/e15all_allen_w*.json"):
        for r in json.load(open(f)): base[r["cell"]["specimen"]] = r
    cells = []
    for path in sorted(glob.glob(f"{DATA}/*.npz")):
        sid = int(os.path.basename(path)[:-4])
        if sid not in base: continue
        meta, sw = load_cell(path)
        if len(sw["Noise 1"]) < 2 or len(sw["Noise 2"]) < 2: continue
        cells.append((sid, meta, sw))
    return cells[:LIMIT] if LIMIT else cells, base


def build_batch(group, units_per_cell):
    """group: list of (sid, meta, sw); returns X [T, ncol], S [T, ncol], col_unit, unit_cell, per-cell info."""
    Xs, Ss, col_unit, unit_cell = [], [], [], []
    u = 0
    for ci, (sid, meta, sw) in enumerate(group):
        X, S = stack(sw["Noise 1"])
        for k in range(units_per_cell):
            Xs.append(X); Ss.append(S); col_unit += [u] * X.shape[1]; unit_cell.append(ci); u += 1
    X = torch.cat(Xs, 1).contiguous(); S = torch.cat(Ss, 1).contiguous()
    return X, S, torch.tensor(col_unit, device=dev), torch.tensor(unit_cell, device=dev)


def unit_mean(v, col_unit, n_units, counts):
    """v: [T, ncol] -> per-unit mean over its columns and time."""
    return torch.zeros(n_units, device=dev).index_add_(0, col_unit, v.sum(0)) / counts


def batched_fit(group, base):
    T = min(stack(g[2]["Noise 1"])[0].shape[0] for g in group)
    upc = 3 if MODEL == "hard" else len(INITS)
    X, S, col_unit, unit_cell = build_batch(group, upc)
    assert X.shape[0] == T
    n_units = len(unit_cell); ncol = X.shape[1]
    counts = torch.zeros(n_units, device=dev).index_add_(0, col_unit, torch.ones(ncol, device=dev)) * T
    tau = TAU_VR_MS / DT_MS; a = math.exp(-1.0 / tau)
    Y = ExpFilter.apply(S, a); target = unit_mean(S, col_unit, n_units, counts)
    ynorm = unit_mean(Y * Y, col_unit, n_units, counts)
    u0 = torch.zeros(ncol, device=dev)
    if MODEL == "hard":
        rs = torch.arange(n_units, device=dev) % upc
        beta0 = torch.tensor([0.95 + 0.045 * torch.rand(1, generator=torch.Generator().manual_seed(1000 + k)).item() for k in range(upc)], device=dev)[rs]
        # rate-matched gain (bias 0, u_r 0), batched bisection as in E15
        lo = torch.full((n_units,), math.log(1e-4), device=dev); hi = torch.zeros(n_units, device=dev)
        with torch.no_grad():
            for _ in range(22):
                mid = 0.5 * (lo + hi)
                Sm = B.lif_vec(torch.exp(mid)[col_unit] * X, u0, beta0[col_unit], torch.zeros(ncol, device=dev))
                r = unit_mean(Sm, col_unit, n_units, counts)
                hi = torch.where(r > target, mid, hi); lo = torch.where(r > target, lo, mid)
        P = dict(zb=torch.log(beta0 / (1 - beta0)), lg=0.5 * (lo + hi), bias=torch.zeros(n_units, device=dev), ur=torch.zeros(n_units, device=dev))
    else:
        hb = [base[g[0]]["models"]["hard"] for g in group]
        rep = lambda key: torch.tensor([h[key] for h in hb], device=dev).repeat_interleave(upc)
        beta = rep("beta"); ur = rep("ur")
        q0 = torch.tensor([i[0] for i in INITS], device=dev).repeat(len(group)); r0 = torch.tensor([i[1] for i in INITS], device=dev).repeat(len(group))
        P = dict(zb=torch.log(beta / (1 - beta)), lg=torch.log(rep("gain")), bias=rep("bias"), lq=torch.log(q0), zr=torch.log(r0 / (1 - r0)))
        ur_fixed = ur
    for v in P.values(): v.requires_grad_(True)
    opt = torch.optim.Adam(list(P.values()), lr=0.005)
    losses = torch.zeros(n_units, device=dev); it_times = []
    z = torch.zeros(ncol, device=dev)
    for it in range(ITERS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        I = torch.exp(P["lg"])[col_unit] * X + P["bias"][col_unit]
        if MODEL == "hard":
            Sm = B.lif_vec(I, u0, torch.sigmoid(P["zb"])[col_unit], P["ur"][col_unit])
        else:
            Sm, _, _ = B.alif_vec(I, z, z, torch.sigmoid(P["zb"])[col_unit], torch.sigmoid(P["zr"])[col_unit],
                                  torch.exp(P["lq"])[col_unit], ur_fixed[col_unit])
        err = unit_mean((ExpFilter.apply(Sm, a) - Y) ** 2, col_unit, n_units, counts) / ynorm
        rate = unit_mean(Sm, col_unit, n_units, counts)
        lu = err + ((rate - target) / target) ** 2
        opt.zero_grad(); lu.sum().backward(); opt.step()
        losses = lu.detach()
        torch.cuda.synchronize(); it_times.append(time.perf_counter() - t0)
    # best unit per cell (lowest final training loss, finite), held-out F1 on Noise 2
    res = []
    L = losses.view(len(group), upc).clone(); L[~torch.isfinite(L)] = float("inf")
    best = L.argmin(1)
    win = int(WIN_MS / DT_MS)
    for ci, (sid, meta, sw) in enumerate(group):
        k = ci * upc + int(best[ci])
        if not math.isfinite(float(L[ci, best[ci]])):
            res.append(dict(cell=sid, failed=True)); continue
        Xte, Ste = stack(sw["Noise 2"])
        with torch.no_grad():
            Ite = torch.exp(P["lg"][k]) * Xte + P["bias"][k]; nte = Xte.shape[1]; zz = torch.zeros(nte, device=dev)
            if MODEL == "hard":
                Sh = B.lif_vec(Ite.contiguous(), zz, torch.sigmoid(P["zb"][k]).repeat(nte), P["ur"][k].repeat(nte))
            else:
                Sh, _, _ = B.alif_vec(Ite.contiguous(), zz, zz, torch.sigmoid(P["zb"][k]).repeat(nte), torch.sigmoid(P["zr"][k]).repeat(nte),
                                      torch.exp(P["lq"][k]).repeat(nte), ur_fixed[k].repeat(nte))
        res.append(dict(cell=sid, f1_test=f1(Sh, Ste, win), train_loss=float(L[ci, best[ci]]), unit=int(best[ci]),
                        beta=float(torch.sigmoid(P["zb"][k]))))
    return res, it_times, ncol


if __name__ == "__main__":
    cells, base = load_db()
    by_T = {}
    for c in cells:
        by_T.setdefault(stack(c[2]["Noise 1"])[0].shape[0], []).append(c)
    out = dict(model=MODEL, iters=ITERS, cells=len(cells), groups=[], results=[])
    torch.cuda.synchronize(); t_all = time.perf_counter()
    for T, grp in sorted(by_T.items(), key=lambda kv: -len(kv[1])):
        # split a length group into kernel calls of at most GROUP columns
        upc = 3 if MODEL == "hard" else len(INITS)
        chunks, cur, ncur = [], [], 0
        for c in grp:
            k = len(c[2]["Noise 1"]) * upc
            if cur and ncur + k > GROUP: chunks.append(cur); cur, ncur = [], 0
            cur.append(c); ncur += k
        if cur: chunks.append(cur)
        for ch in chunks:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            res, it_times, ncol = batched_fit(ch, base)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            out["groups"].append(dict(T=T, cells=len(ch), columns=ncol, seconds=dt, ms_per_iter=1000 * float(np.median(it_times[5:]))))
            out["results"] += res
            print(f"T={T} cells={len(ch)} columns={ncol}: {dt:.1f} s, {1000 * np.median(it_times[5:]):.1f} ms/iter", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
    torch.cuda.synchronize(); out["seconds_total"] = time.perf_counter() - t_all
    print(f"database ({len(cells)} cells, {MODEL}): {out['seconds_total']:.1f} s", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
