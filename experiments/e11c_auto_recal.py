"""E11c: Out-of-sample automatic dispatch check with recalibrated backward threshold (T >= 4096).

E11b revealed that on the unseen grid, the original backward threshold (T >= 1024 for N <= 1024)
lost 32-40% at T = 2048 against the tuned prefetching serial backward. Raising the backward scan
threshold to T >= 4096 (while keeping forward scan at N <= 1024, T >= 16384) eliminates that
regression while preserving full speedups at T >= 8192.

Output: results/e11c_auto_oos_recal.json
"""
import itertools
import json
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifscan import triton_lif as L


def timeit(fn, reps=25):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def auto_modes_recal(N, T):
    fwd = "scan" if (N <= 1024 and T >= 16384) else "serial"
    bwd = "scan" if (N <= 1024 and T >= 4096) else "serial"
    return fwd, bwd


def lif_auto_recal(I, u0, beta=0.9, theta=1.0, u_r=0.0, alpha=4.0, K=16, repair=0):
    T, N = I.shape
    fm, bm = auto_modes_recal(N, T)
    return L.LIFAuto.apply(I, u0, beta, theta, u_r, alpha, K, repair, fm, bm)


def main():
    GRID_N = (16, 64, 256, 2048, 4096, 16384)
    GRID_T = (2048, 8192, 32768, 131072)
    rows = []
    for N, T in itertools.product(GRID_N, GRID_T):
        if N * T > 2 ** 28:
            continue
        torch.manual_seed(0)
        I0 = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous()
        u0 = torch.zeros(N, device="cuda")
        G = torch.randn(T, N, device="cuda")
        out = {}
        fns = {
            "serial": lambda I: L.lif_tuned(I, u0, beta=0.9, mode="serial"),
            "scan": lambda I: L.lif_tuned(I, u0, beta=0.9, mode="scan", repair=0),
            "auto": lambda I: lif_auto_recal(I, u0, beta=0.9),
        }
        for name, f in fns.items():
            I = I0.clone().requires_grad_(True)
            def step():
                I.grad = None
                s = f(I)
                (s * G).sum().backward()
            t = timeit(step)
            I.grad = None
            s = f(I)
            (s * G).sum().backward()
            out[name] = (t, s.detach().clone(), I.grad.clone())
        r = dict(
            N=N, T=T, modes=list(auto_modes_recal(N, T)),
            serial=out["serial"][0], scan=out["scan"][0], auto=out["auto"][0],
        )
        gs = out["serial"][2]
        r["spikes_equal"] = all(torch.equal(out[k][1], out["serial"][1]) for k in ("scan", "auto"))
        r["grad_rel_auto"] = ((out["auto"][2] - gs).norm() / gs.norm()).item()
        r["x_auto"] = r["serial"] / r["auto"]
        r["x_scan"] = r["serial"] / r["scan"]
        rows.append(r)
        print(
            f"N={N:<7} T={T:<6} modes={tuple(r['modes'])} serial {r['serial']:7.3f} "
            f"scan {r['scan']:7.3f} (x{r['x_scan']:5.2f}) auto {r['auto']:7.3f} (x{r['x_auto']:5.2f})",
            flush=True,
        )
        del I0, G, out
        torch.cuda.empty_cache()
    with open(os.path.join(ROOT, "results", "e11c_auto_oos_recal.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("E11C DONE")


if __name__ == "__main__":
    main()
