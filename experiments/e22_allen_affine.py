"""E22: learn the reset rule.  Affine-reset LIF u <- gamma*v + delta (gamma in [0,1] via a sigmoid),
which contains hard (gamma=0) and soft (gamma=1, delta=-1) reset, fitted with the exact affine scan to
the same Allen cells, split and protocol as E15 (rate-matched init, van Rossum + rate loss, 400 Adam
steps, best of 3 restarts).  Also times one iteration with serial vs scan kernels."""
import sys, os, json, glob, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
_argv = sys.argv[:]
sys.argv = [sys.argv[0], "0.2", "1", "1"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e15_allen_fit.py")).read().split("rows = []")[0])
sys.argv = _argv
from lifscan.triton_affine import lif_affine

ITERS, RESTARTS = 400, 3
WORKER, NW = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
CELLS = set(); BASE = {}
for f in glob.glob("results/e15b_allen_w*.json") + glob.glob("results/e15c_allen_w*.json"):
    for r in json.load(open(f)):
        CELLS.add(r["cell"]["specimen"]); BASE[r["cell"]["specimen"]] = r
WARM = os.environ.get("E22_WARM", "0") == "1"
TAG = "warm" if WARM else "cold"


def sim(X, p, mode="scan"):
    I = torch.exp(p["lg"]) * X + p["bias"]
    return lif_affine(I, torch.zeros(X.shape[1], device=dev), torch.sigmoid(p["zb"]), torch.sigmoid(p["zg"]), p["delta"],
                      mode=mode, C=128, K=32)


def rate_gain(X, target, beta0, g0, d0):
    lo, hi = math.log(1e-4), math.log(1.0)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        p = dict(zb=torch.tensor(math.log(beta0 / (1 - beta0)), device=dev), lg=torch.tensor(mid, device=dev),
                 bias=torch.tensor(0.0, device=dev), zg=torch.tensor(math.log(g0 / (1 - g0)), device=dev), delta=torch.tensor(d0, device=dev))
        with torch.no_grad():
            r = sim(X, p).mean().item()
        if r > target: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


def fit(Xtr, Str, seed, mode="scan", iters=ITERS, warm=None):
    target = Str.mean().item()
    if warm is not None:   # warm start from a fitted hard (gamma ~ 0) or soft (gamma ~ 1) solution
        beta0, g0, d0, lg0, b0 = warm
    else:
        g = torch.Generator().manual_seed(seed)
        beta0 = 0.95 + 0.045 * torch.rand(1, generator=g).item()
        g0 = 0.2 + 0.6 * torch.rand(1, generator=g).item()
        d0 = -0.5 * torch.rand(1, generator=g).item()
        lg0 = rate_gain(Xtr, target, beta0, g0, d0); b0 = 0.0
    p = {k: torch.tensor(v, device=dev, requires_grad=True) for k, v in
         dict(zb=math.log(beta0 / (1 - beta0)), lg=lg0, bias=b0, zg=math.log(g0 / (1 - g0)), delta=d0).items()}
    lr = 0.005
    opt = torch.optim.Adam(list(p.values()), lr=lr)
    tau = TAU_VR_MS / DT_MS; Ytr = vr(Str, tau); times, losses = [], []
    good = {k: v.detach().clone() for k, v in p.items()}
    for it in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        S = sim(Xtr, p, mode)
        loss = ((vr(S, tau) - Ytr) ** 2).mean() / (Ytr ** 2).mean() + ((S.mean() - target) / target) ** 2
        opt.zero_grad(); loss.backward()
        finite = torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in p.values())
        if finite:
            torch.nn.utils.clip_grad_norm_(list(p.values()), 10.0)
            opt.step()
        if not finite or not all(torch.isfinite(v).all() for v in p.values()):
            with torch.no_grad():          # revert to the last finite parameters and halve the step size
                for k in p: p[k].copy_(good[k])
            lr *= 0.5
            for gparam in opt.param_groups: gparam["lr"] = lr
        else:
            good = {k: v.detach().clone() for k, v in p.items()}
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0); losses.append(loss.item() if torch.isfinite(loss) else float("nan"))
    losses[-1] = losses[-1] if np.isfinite(losses[-1]) else min([l for l in losses if np.isfinite(l)], default=float("nan"))
    return p, losses, times


rows = []
paths = [p for p in sorted(glob.glob("data/allen/*.npz")) if int(os.path.basename(p)[:-4]) in CELLS][WORKER::NW]
for path in paths:
    meta, sw = load_cell(path)
    Xtr, Str = stack(sw["Noise 1"]); Xte, Ste = stack(sw["Noise 2"])
    best = None
    if WARM:
        b = BASE[meta["specimen"]]["models"]
        starts = [(b["hard"]["beta"], 0.002, b["hard"]["ur"], math.log(b["hard"]["gain"]), b["hard"]["bias"]),
                  (b["soft"]["beta"], 0.998, -1.0, math.log(b["soft"]["gain"]), b["soft"]["bias"])]
    else:
        starts = [None] * RESTARTS
    for rs, w in enumerate(starts):
        p, losses, times = fit(Xtr, Str, 2000 + rs, warm=w)
        if np.isfinite(losses[-1]) and (best is None or losses[-1] < best[1][-1]):
            best = (p, losses, times)
    row = dict(cell=meta, T=Xtr.shape[0], N=Xtr.shape[1])
    if best is None:
        row["failed"] = True
    else:
        p, losses, times = best
        with torch.no_grad():
            Sh = sim(Xte, p, "serial")
        row.update(f1_test=f1(Sh, Ste, int(WIN_MS / DT_MS)), beta=torch.sigmoid(p["zb"]).item(), gamma=torch.sigmoid(p["zg"]).item(),
                   delta=p["delta"].item(), gain=math.exp(p["lg"].item()), bias=p["bias"].item(), train_loss=losses[-1],
                   ms_scan=1000 * float(np.median(times[5:])))
        if not WARM:
            _, _, ts = fit(Xtr, Str, 7, mode="serial", iters=25)
            row["ms_serial"] = 1000 * float(np.median(ts[5:]))
    rows.append(row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "cell"} | {"cell": meta["specimen"], "type": meta["dendrite"]}), flush=True)
    json.dump(rows, open(f"results/e22_allen_affine_{TAG}_w{WORKER}.json", "w"), indent=1)
