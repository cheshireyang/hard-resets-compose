"""E15: fit LIF models to Allen Cell Types recordings (mouse V1, current clamp, noise stimuli).
Train on the 'Noise 1' sweeps, test on the 'Noise 2' sweeps (a different noise realisation).
Models: hard-reset LIF (beta, gain, bias, u_r), soft-reset LIF (beta, gain, bias), no-reset (beta, gain, bias).
Metric: spike-time F1 within +-4 ms on held-out sweeps; reference: the cell's own repeat reliability
(F1 between repeated presentations of Noise 2).  Also times one training iteration of the hard model
with serial vs auto kernels.
usage: python3 experiments/e15_allen_fit.py DT_MS ITERS RESTARTS"""
import sys, os, json, glob, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from lifscan import fast
from lifscan.fit import lif_param, lif_soft

DT_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
RESTARTS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
WORKER, NWORKERS = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
TIMING = os.environ.get("E15_TIMING", "1") == "1"
OUTNAME = os.environ.get("E15_OUT", f"results/e15_allen_fit_dt{DT_MS}.json")
WIN_MS, TAU_VR_MS = 4.0, 10.0
dev = "cuda"


def load_cell(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    sweeps = sorted({k.split("__")[0] for k in z.files if "__" in k})
    out = {"Noise 1": [], "Noise 2": []}
    for s in sweeps:
        name = str(z[f"{s}__name"]); rate = float(z[f"{s}__rate"])
        i_pa = z[f"{s}__i_pa"]; sp = z[f"{s}__spikes"]
        step = int(round(rate * DT_MS / 1000.0))
        n = len(i_pa) // step
        x = i_pa[: n * step].reshape(n, step).mean(1)
        s_bin = np.zeros(n, dtype=np.float32)
        idx = np.minimum((sp * 1000.0 / DT_MS).astype(np.int64), n - 1)
        s_bin[idx] = 1.0
        if name in out:
            out[name].append((x, s_bin))
    return meta, out


def stack(pairs):
    T = min(len(p[0]) for p in pairs)
    T = (T // 256) * 256
    X = np.stack([p[0][:T] for p in pairs], 1) / 100.0          # 100 pA units
    S = np.stack([p[1][:T] for p in pairs], 1)
    return torch.tensor(X, device=dev), torch.tensor(S, device=dev)


def vr(S, tau_steps):
    a = math.exp(-1.0 / tau_steps)
    k = a ** torch.arange(S.shape[0], device=S.device, dtype=S.dtype)
    n = 2 * S.shape[0]
    return torch.fft.irfft(torch.fft.rfft(S, n=n, dim=0) * torch.fft.rfft(k, n=n)[:, None], n=n, dim=0)[: S.shape[0]]


def f1(Sm, St, win):
    pool = lambda A: torch.nn.functional.max_pool1d(A.T[:, None, :], 2 * win + 1, stride=1, padding=win)[:, 0, :].T
    rec = (pool(Sm) * St).sum() / St.sum().clamp(min=1)
    prec = (pool(St) * Sm).sum() / Sm.sum().clamp(min=1)
    return float(2 * rec * prec / (rec + prec + 1e-12))


def simulate(kind, X, p, mode="auto"):
    I = p["gain"] * X + p["bias"]
    u0 = torch.zeros(X.shape[1], device=dev)
    beta = torch.sigmoid(p["zb"])
    if kind == "hard":
        return lif_param(I, u0, beta, torch.tensor(1.0, device=dev), p["ur"], mode=mode)
    if kind == "soft":
        return lif_soft(I, u0, beta)
    k = beta ** torch.arange(X.shape[0], device=dev, dtype=X.dtype)
    n = 2 * X.shape[0]
    v = torch.fft.irfft(torch.fft.rfft(I, n=n, dim=0) * torch.fft.rfft(k, n=n)[:, None], n=n, dim=0)[: X.shape[0]]
    return fast.SpikeFn.apply(v - 1.0, 4.0)


def rate_matched_gain(kind, Xtr, target, beta0):
    """Bisection on log-gain (bias = 0, u_r = 0) so that the model's training firing rate matches the cell's."""
    lo, hi = math.log(1e-4), math.log(1.0)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        p = dict(zb=torch.tensor(math.log(beta0 / (1 - beta0)), device=dev), gain=torch.tensor(math.exp(mid), device=dev),
                 bias=torch.tensor(0.0, device=dev), ur=torch.tensor(0.0, device=dev))
        with torch.no_grad():
            r = simulate(kind, Xtr, p).mean().item()
        if r > target: hi = mid
        else: lo = mid
    return math.exp(0.5 * (lo + hi))


def fit(kind, Xtr, Str, seed, mode="auto", iters=ITERS):
    g = torch.Generator().manual_seed(seed)
    beta0 = 0.95 + 0.045 * torch.rand(1, generator=g).item()
    target = Str.mean().item()
    gain0 = rate_matched_gain(kind, Xtr, target, beta0)
    p = dict(zb=torch.tensor(math.log(beta0 / (1 - beta0)), device=dev, requires_grad=True),
             lg=torch.tensor(math.log(gain0), device=dev, requires_grad=True),
             bias=torch.tensor(0.0, device=dev, requires_grad=True),
             ur=torch.tensor(0.0, device=dev, requires_grad=True))
    params = [p["zb"], p["lg"], p["bias"]] + ([p["ur"]] if kind == "hard" else [])
    opt = torch.optim.Adam(params, lr=0.005)
    tau = TAU_VR_MS / DT_MS
    Ytr = vr(Str, tau)
    times, losses = [], []
    for it in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        p["gain"] = torch.exp(p["lg"])
        S = simulate(kind, Xtr, p, mode)
        loss = ((vr(S, tau) - Ytr) ** 2).mean() / (Ytr ** 2).mean() + ((S.mean() - target) / target) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0); losses.append(loss.item())
    p["gain"] = torch.exp(p["lg"]).detach()
    return p, losses, times


rows = []
DONE = set()
for f in glob.glob(os.environ.get("E15_SKIP_GLOB", "results/__none__")):
    try:
        DONE |= {r["cell"]["specimen"] for r in json.load(open(f))}
    except Exception:
        pass
todo = [p for p in sorted(glob.glob("data/allen/*.npz")) if int(os.path.basename(p)[:-4]) not in DONE]
for path in todo[WORKER::NWORKERS]:
    meta, sw = load_cell(path)
    if len(sw["Noise 1"]) < 2 or len(sw["Noise 2"]) < 2:
        continue
    Xtr, Str = stack(sw["Noise 1"]); Xte, Ste = stack(sw["Noise 2"])
    win = int(WIN_MS / DT_MS)
    # repeat reliability of the cell on the held-out stimulus (all pairs of Noise 2 repeats)
    rel = [f1(Ste[:, i:i + 1], Ste[:, j:j + 1], win) for i in range(Ste.shape[1]) for j in range(Ste.shape[1]) if i < j]
    row = dict(cell=meta, T_train=Xtr.shape[0], N_train=Xtr.shape[1], N_test=Xte.shape[1],
               rate_hz=float(Str.mean().item() * 1000 / DT_MS), reliability=float(np.mean(rel)), models={})
    for kind in ("hard", "soft", "noreset"):
        best = None
        for rs in range(RESTARTS):
            p, losses, times = fit(kind, Xtr, Str, 1000 + rs)
            if best is None or losses[-1] < best[1][-1]:
                best = (p, losses, times)
        p, losses, times = best
        with torch.no_grad():
            Ste_hat = simulate(kind, Xte, p)
        row["models"][kind] = dict(f1_test=f1(Ste_hat, Ste, win), train_loss=losses[-1],
                                   beta=torch.sigmoid(p["zb"]).item(), gain=float(p["gain"]), bias=p["bias"].item(),
                                   ur=p["ur"].item() if kind == "hard" else None,
                                   ms_per_iter=1000 * float(np.median(times[5:])),
                                   rate_model_hz=float(Ste_hat.mean().item() * 1000 / DT_MS))
    # timing of the hard model: serial vs auto (same data, 30 iterations)
    for mode in (("serial", "auto") if TIMING else ()):
        _, _, times = fit("hard", Xtr, Str, 7, mode=mode, iters=30)
        row[f"hard_ms_{mode}"] = 1000 * float(np.median(times[5:]))
    rows.append(row)
    m = row["models"]
    print(f"cell {meta['specimen']} {meta['dendrite']:15s} {meta['region']:8s} T={row['T_train']} N={row['N_train']} "
          f"rate {row['rate_hz']:.1f}Hz | reliability {row['reliability']:.3f} | F1 hard {m['hard']['f1_test']:.3f} "
          f"soft {m['soft']['f1_test']:.3f} noreset {m['noreset']['f1_test']:.3f}", flush=True)
    json.dump(rows, open(OUTNAME, "w"), indent=1)
