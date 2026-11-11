"""E18: soft-reset LIF fitted to spiny Allen cells (the class where soft reset predicts spikes
better): serial soft kernels vs the exact soft scan (+ FFT adjoint).  Same initialization and
data; reports ms/iteration and agreement of the fitted parameters and test F1."""
import sys, os, json, glob, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
sys.argv = [sys.argv[0], "0.2", "1", "1"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e15_allen_fit.py")).read().split("rows = []")[0])
from lifscan.fit import lif_soft_scan

ITERS = 100

def fit_soft(Xtr, Str, mode, beta0, gain0):
    p = dict(zb=torch.tensor(math.log(beta0 / (1 - beta0)), device=dev, requires_grad=True),
             lg=torch.tensor(math.log(gain0), device=dev, requires_grad=True),
             bias=torch.tensor(0.0, device=dev, requires_grad=True))
    opt = torch.optim.Adam([p["zb"], p["lg"], p["bias"]], lr=0.005)
    tau = TAU_VR_MS / DT_MS; Ytr = vr(Str, tau); target = Str.mean().item(); times = []
    for it in range(ITERS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        I = torch.exp(p["lg"]) * Xtr + p["bias"]
        S = lif_soft_scan(I, torch.zeros(Xtr.shape[1], device=dev), torch.sigmoid(p["zb"]), mode=mode, C=128, K=32)
        loss = ((vr(S, tau) - Ytr) ** 2).mean() / (Ytr ** 2).mean() + ((S.mean() - target) / target) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    return p, times

rows = []
for path in sorted(glob.glob("data/allen/*.npz")):
    meta, sw = load_cell(path)
    if meta["dendrite"] != "spiny" or len(sw["Noise 1"]) < 2 or len(sw["Noise 2"]) < 2:
        continue
    Xtr, Str = stack(sw["Noise 1"]); Xte, Ste = stack(sw["Noise 2"])
    beta0 = 0.97
    gain0 = rate_matched_gain("soft", Xtr, Str.mean().item(), beta0)
    out = {}
    for mode in ("serial", "scan"):
        p, times = fit_soft(Xtr, Str, mode, beta0, gain0)
        with torch.no_grad():
            I = torch.exp(p["lg"]) * Xte + p["bias"]
            Sh = lif_soft_scan(I, torch.zeros(Xte.shape[1], device=dev), torch.sigmoid(p["zb"]), mode="serial")
        out[mode] = dict(ms=1000 * float(np.median(times[5:])), beta=torch.sigmoid(p["zb"]).item(),
                         gain=math.exp(p["lg"].item()), bias=p["bias"].item(), f1=f1(Sh, Ste, int(WIN_MS / DT_MS)))
    row = dict(cell=meta, N=Xtr.shape[1], T=Xtr.shape[0], **{f"{m}_{k}": v for m in out for k, v in out[m].items()})
    row["speedup"] = row["serial_ms"] / row["scan_ms"]
    row["max_param_rel_diff"] = max(abs(out["serial"][k] - out["scan"][k]) / abs(out["serial"][k]) for k in ("beta", "gain", "bias"))
    rows.append(row)
    print(f"cell {meta['specimen']} N={row['N']} T={row['T']} soft ms/iter serial {row['serial_ms']:.1f} scan {row['scan_ms']:.1f} "
          f"x{row['speedup']:.2f} | F1 serial {out['serial']['f1']:.3f} scan {out['scan']['f1']:.3f} | param rel diff {row['max_param_rel_diff']:.1e}", flush=True)
    json.dump(rows, open("results/e18_soft_allen.json", "w"), indent=1)
    if len(rows) >= 12:
        break
