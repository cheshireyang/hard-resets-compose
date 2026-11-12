"""E28: fit ALIF neurons (hard reset to a fitted u_r, spike-triggered threshold w <- rho*w + q*s) to the Allen cells that have
official GLIF models, with the exact ALIF scan in the loop: every iteration's forward pass uses the adaptation of the
previous iteration at chunk boundaries as its prediction (first iteration: serial), the backward pass two affine scans.
Same data, split, loss and protocol as E15 (Noise 1 train, Noise 2 test, van Rossum + rate loss, Adam lr 0.005).
Warm start from the cell's E15 hard-reset fit (beta, gain, bias, u_r), with four (q, rho) initialisations; best training
loss wins.  Also times 30 iterations with serial vs scan kernels.  usage: python3 experiments/e28_allen_alif.py [WORKER NW]"""
import sys, os, json, glob, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
_argv = sys.argv[:]
sys.argv = [sys.argv[0], "0.2", "1", "1"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e15_allen_fit.py")).read().split("rows = []")[0])
sys.argv = _argv
from lifscan import triton_alif as A

ITERS = int(os.environ.get("E28_ITERS", "400"))
INITS = ((0.05, 0.99), (0.2, 0.99), (0.05, 0.998), (0.2, 0.998))
C_SCAN = 128
WORKER, NW = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
BASE = {}
for f in (glob.glob(os.environ["E28_BASE_GLOB"]) if "E28_BASE_GLOB" in os.environ else glob.glob("results/e15b_allen_w*.json") + glob.glob("results/e15c_allen_w*.json")):
    for r in json.load(open(f)):
        BASE[r["cell"]["specimen"]] = r
GLIF = {m["specimen"] for m in json.load(open(os.environ.get("E28_GLIF", "results/allen_glif_models.json")))}
DATA = os.environ.get("E28_DATA", "data/allen")
TAG = os.environ.get("E28_TAG", "")
TIMING = os.environ.get("E28_TIMING", "1") == "1"
MODE = os.environ.get("E28_MODE", "scan")
LIMIT = int(os.environ.get("E28_LIMIT", "0"))


def sim(X, p, mode, Wprev=None):
    """mode: 'serial' or 'scan'; Wprev: [T, N] adaptation of the previous iterate (for the scan's prediction)."""
    I = torch.exp(p["lg"]) * X + p["bias"]
    N = X.shape[1]; z = torch.zeros(N, device=dev)
    q = torch.exp(p["lq"]); rho = torch.sigmoid(p["zr"])
    if mode == "scan" and Wprev is not None:
        Wpred = torch.cat([z[None], Wprev[C_SCAN - 1:-1:C_SCAN]], 0).contiguous()
        return A.alif(I.contiguous(), z, z, torch.sigmoid(p["zb"]), rho, q, 1.0, float(p["ur"]), fwd="scan", bwd="scan",
                      Wpred=Wpred, dw=0.3 * float(q), C=C_SCAN, Cb=64)
    return A.alif(I.contiguous(), z, z, torch.sigmoid(p["zb"]), rho, q, 1.0, float(p["ur"]), fwd="serial",
                  bwd="scan" if mode == "scan" else "serial")


def fit(Xtr, Str, b, q0, rho0, mode="scan", iters=ITERS):
    hb = b["models"]["hard"]
    p = dict(zb=torch.tensor(math.log(hb["beta"] / (1 - hb["beta"])), device=dev, requires_grad=True),
             lg=torch.tensor(math.log(hb["gain"]), device=dev, requires_grad=True),
             bias=torch.tensor(hb["bias"], device=dev, requires_grad=True),
             lq=torch.tensor(math.log(q0), device=dev, requires_grad=True),
             zr=torch.tensor(math.log(rho0 / (1 - rho0)), device=dev, requires_grad=True),
             ur=hb["ur"])
    opt = torch.optim.Adam([p["zb"], p["lg"], p["bias"], p["lq"], p["zr"]], lr=0.005)
    tau = TAU_VR_MS / DT_MS; Ytr = vr(Str, tau); target = Str.mean().item()
    Wprev = None; losses, times = [], []
    for it in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        S, U, W = sim(Xtr, p, mode, Wprev)
        loss = ((vr(S, tau) - Ytr) ** 2).mean() / (Ytr ** 2).mean() + ((S.mean() - target) / target) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        Wprev = W.detach()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0); losses.append(loss.item())
        if not math.isfinite(losses[-1]):
            break
    return p, losses, times


rows = []
cells = sorted(s for s in BASE if s in GLIF)
if LIMIT: cells = cells[:LIMIT]
for sid in cells[WORKER::NW]:
    meta, sw = load_cell(f"{DATA}/{sid}.npz")
    Xtr, Str = stack(sw["Noise 1"]); Xte, Ste = stack(sw["Noise 2"])
    win = int(WIN_MS / DT_MS)
    best = None
    for q0, rho0 in INITS:
        p, losses, times = fit(Xtr, Str, BASE[sid], q0, rho0, mode=MODE)
        if math.isfinite(losses[-1]) and (best is None or losses[-1] < best[1][-1]):
            best = (p, losses, times, (q0, rho0))
    row = dict(cell=meta, T_train=Xtr.shape[0], N_train=Xtr.shape[1])
    if best is None:
        row["failed"] = True
    else:
        p, losses, times, init = best
        with torch.no_grad():
            Ste_hat, _, _ = sim(Xte, p, "serial")
        row.update(f1_test=f1(Ste_hat, Ste, win), train_loss=losses[-1], init=init,
                   beta=torch.sigmoid(p["zb"]).item(), gain=float(torch.exp(p["lg"])), bias=p["bias"].item(), ur=p["ur"],
                   q=float(torch.exp(p["lq"])), rho=torch.sigmoid(p["zr"]).item(),
                   rate_model_hz=float(Ste_hat.mean().item() * 1000 / DT_MS), hard_f1=BASE[sid]["models"]["hard"]["f1_test"],
                   hard_train_loss=BASE[sid]["models"]["hard"]["train_loss"])
        q0, rho0 = init
        for mode in (("serial", "scan") if TIMING else ()):
            _, _, tt = fit(Xtr, Str, BASE[sid], q0, rho0, mode=mode, iters=30)
            row[f"ms_{mode}"] = 1000 * float(np.median(tt[5:]))
    rows.append(row)
    print(f"cell {sid} {meta['dendrite']:15s}: " + ("FAILED" if row.get("failed") else
          f"ALIF F1 {row['f1_test']:.3f} (hard {row['hard_f1']:.3f}) q={row['q']:.3f} rho={row['rho']:.4f}"), flush=True)
    json.dump(rows, open(f"results/e28{TAG}_allen_alif_w{WORKER}.json", "w"), indent=1)
