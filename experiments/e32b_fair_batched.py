"""E32b: Fair full-GPU batched serial throughput baseline for Sec. 'fit' (Hard LIF + ALIF).

Diagnoses and fixes the untiled serial ExpFilter bottleneck in E32 by using the exact same
FFT van Rossum filter as the per-cell fits (E15/E28), and measures per-iteration time and
per-cell amortized time for BOTH:
  - actual 32-cell real cohort (T = 124,928, ~291 hard / ~388 ALIF columns), and
  - full-GPU saturated batch (tiled to ~2,048 columns, ~224 cells simultaneously, matching
    GROUP=2048 for whole-database fitting of all 1,044 cells in ~5 batches).
Also profiles forward/backward kernel time vs van Rossum filter time so every millisecond
is accounted for.

Output: results/e32b_fair_batched.json
"""
import glob
import json
import math
import os
import sys
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifscan import triton_batch as B

_e15 = os.path.join(ROOT, "experiments", "e15_allen_fit.py")
_ns = {"__file__": _e15}
exec(open(_e15).read().split("rows = []")[0], _ns)
load_cell = _ns["load_cell"]
stack = _ns["stack"]
TAU_VR_MS = _ns["TAU_VR_MS"]
DT_MS = _ns["DT_MS"]

DATA = os.environ.get("E32_DATA", "data/allen_all")
dev = "cuda"


def fft_vr_kernel(T, tau_steps):
    nfft = 1 << (2 * T - 1).bit_length()
    k = torch.exp(-torch.arange(T, device=dev, dtype=torch.float32) / tau_steps)
    return nfft, torch.fft.rfft(k, nfft)[:, None]


def fft_filter(S, nfft, Kf):
    return torch.fft.irfft(torch.fft.rfft(S, nfft, dim=0) * Kf, nfft, dim=0)[: S.shape[0]]


def unit_mean(v, col_unit, n_units, counts):
    return torch.zeros(n_units, device=dev).index_add_(0, col_unit, v.sum(0)) / counts


def bench_batched(model, group, tile_factor=1, iters=30):
    upc = 3 if model == "hard" else 4
    Xs, Ss, col_unit = [], [], []
    u = 0
    for _ in range(tile_factor):
        for sid, meta, sw in group:
            X, S = stack(sw["Noise 1"])
            for _k in range(upc):
                Xs.append(X)
                Ss.append(S)
                col_unit += [u] * X.shape[1]
                u += 1
    X = torch.cat(Xs, 1).contiguous()
    S = torch.cat(Ss, 1).contiguous()
    T, ncol = X.shape
    n_cells = len(group) * tile_factor
    n_units = u
    col_unit_t = torch.tensor(col_unit, device=dev)
    counts = torch.zeros(n_units, device=dev).index_add_(0, col_unit_t, torch.ones(ncol, device=dev)) * T

    tau = TAU_VR_MS / DT_MS
    nfft, Kf = fft_vr_kernel(T, tau)
    Y = fft_filter(S, nfft, Kf)
    target = unit_mean(S, col_unit_t, n_units, counts).clamp(min=1e-6)
    ynorm = unit_mean(Y * Y, col_unit_t, n_units, counts).clamp(min=1e-9)

    u0 = torch.zeros(ncol, device=dev)
    z = torch.zeros(ncol, device=dev)
    if model == "hard":
        P = dict(
            zb=torch.full((n_units,), 3.0, device=dev, requires_grad=True),
            lg=torch.full((n_units,), -3.0, device=dev, requires_grad=True),
            bias=torch.zeros(n_units, device=dev, requires_grad=True),
            ur=torch.full((n_units,), -1.0, device=dev, requires_grad=True),
        )
    else:
        P = dict(
            zb=torch.full((n_units,), 3.0, device=dev, requires_grad=True),
            lg=torch.full((n_units,), -3.0, device=dev, requires_grad=True),
            bias=torch.zeros(n_units, device=dev, requires_grad=True),
            lq=torch.full((n_units,), math.log(0.1), device=dev, requires_grad=True),
            zr=torch.full((n_units,), 4.6, device=dev, requires_grad=True),
        )
        ur_fixed = torch.full((ncol,), -1.0, device=dev)

    opt = torch.optim.Adam(list(P.values()), lr=0.005)

    # Time full iteration (with FFT van Rossum) AND neuron kernel alone
    it_ms = []
    kern_ms = []
    for it in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        ek0 = torch.cuda.Event(enable_timing=True)
        ek1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        I = torch.exp(P["lg"])[col_unit_t] * X + P["bias"][col_unit_t]
        ek0.record()
        if model == "hard":
            Sm = B.lif_vec(I, u0, torch.sigmoid(P["zb"])[col_unit_t], P["ur"][col_unit_t])
        else:
            Sm, _, _ = B.alif_vec(
                I, z, z,
                torch.sigmoid(P["zb"])[col_unit_t],
                torch.sigmoid(P["zr"])[col_unit_t],
                torch.exp(P["lq"])[col_unit_t],
                ur_fixed,
            )
        ek1.record()
        err = unit_mean((fft_filter(Sm, nfft, Kf) - Y) ** 2, col_unit_t, n_units, counts) / ynorm
        rate = unit_mean(Sm, col_unit_t, n_units, counts)
        lu = err + ((rate - target) / target) ** 2
        opt.zero_grad()
        lu.sum().backward()
        opt.step()
        e1.record()
        torch.cuda.synchronize()
        if it >= 5:
            it_ms.append(e0.elapsed_time(e1))
            kern_ms.append(ek0.elapsed_time(ek1))

    med_it = float(np.median(it_ms))
    med_kfwd = float(np.median(kern_ms))
    full_iters = 600 if model == "hard" else 400
    sec_per_cell = (med_it / 1000.0) * full_iters / n_cells
    sec_1044 = sec_per_cell * 1044
    row = dict(
        model=model, T=T, cells=n_cells, units=n_units, columns=ncol,
        ms_per_iter=med_it, ms_kernel_fwd=med_kfwd,
        ms_per_iter_per_cell=med_it / n_cells,
        sec_per_cell_fit=sec_per_cell,
        est_1044_cells_sec=sec_1044,
    )
    print(
        f"{model:4s} cells={n_cells:3d} cols={ncol:4d} T={T} | "
        f"{med_it:6.2f} ms/iter (fwd kern {med_kfwd:5.2f} ms) | "
        f"{row['ms_per_iter_per_cell']:.3f} ms/iter/cell | "
        f"fit {full_iters} iters: {sec_per_cell:.2f} s/cell -> 1044 cells ~ {sec_1044:.1f} s",
        flush=True,
    )
    del X, S, Y, Kf
    torch.cuda.empty_cache()
    return row


def main():
    base = {}
    for f in glob.glob("results/e15all_allen_w*.json"):
        for r in json.load(open(f)):
            base[r["cell"]["specimen"]] = r
    by_T = {}
    for path in sorted(glob.glob(f"{DATA}/*.npz")):
        sid = int(os.path.basename(path)[:-4])
        if sid not in base:
            continue
        meta, sw = load_cell(path)
        if len(sw["Noise 1"]) < 2 or len(sw["Noise 2"]) < 2:
            continue
        T = stack(sw["Noise 1"])[0].shape[0]
        by_T.setdefault(T, []).append((sid, meta, sw))
    # Largest common T group (typically T = 124,928 with 32 cells)
    T_main, grp = max(by_T.items(), key=lambda kv: len(kv[1]))
    print(f"Using {len(grp)} real cells at T={T_main}", flush=True)

    rows = []
    for model, target_cols in (("hard", (300, 650, 1000)), ("alif", (400, 750, 1000))):
        upc = 3 if model == "hard" else 4
        avg_sw = float(np.mean([len(sw["Noise 1"]) for _, _, sw in grp]))
        cols_per_cell = upc * avg_sw
        for tc in target_cols:
            need_cells = min(len(grp), max(1, int(round(tc / cols_per_cell))))
            rows.append(bench_batched(model, grp[:need_cells], tile_factor=1, iters=30))

    out = dict(T=T_main, base_cells=len(grp), rows=rows)
    with open(os.path.join(ROOT, "results", "e32b_fair_batched.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("E32B DONE")


if __name__ == "__main__":
    main()
