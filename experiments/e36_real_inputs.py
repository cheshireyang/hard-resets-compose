"""E36: Real temporally-correlated biological inputs (Allen Cell Types current-clamp sweeps).

Addresses the concern that synthetic speed/piece-count grids use i.i.d. Gaussian inputs.
Evaluates the hard-reset exact scan (v3) vs tuned serial kernels on real cortical stimulus
currents I_t = gain * X_t + bias from data/allen_all/*.npz with their fitted LIF parameters:
  Part A (Per-cell fitted parameters, N = actual Noise-1 sweeps, typically 2..6, T = 131,072):
    First 32 fitted cells (balanced spiny/aspiny where available). Measures firing rate,
    mean chunk-map pieces, overflow fraction, pre-repair spike flips, repair rounds,
    and tuned serial vs scan forward and forward+backward times.
  Part B (Stacked real stimulus waveforms at N in {16, 128}, T = 131,072, representative
    cortical leak beta in {0.96, 0.99}): measures piece counts, overflow, flips, repair
    rounds, and forward+backward speedups on correlated real currents at larger N.

Output: results/e36_real_inputs.json
"""
import glob
import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifscan import triton_lif as L

DATA = os.environ.get("E36_DATA", "data/allen_all")
DT_MS = 0.2
T_USE = 114688  # 112 * 1024 <= 114,944 (fits all Noise-1 sweeps; divisible by C up to 4096)
dev = "cuda"


def timeit(fn, reps=7):
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


def load_noise1(path):
    z = np.load(path, allow_pickle=False)
    sweeps = sorted({k.split("__")[0] for k in z.files if "__" in k})
    xs = []
    for s in sweeps:
        if str(z[f"{s}__name"]) != "Noise 1":
            continue
        rate = float(z[f"{s}__rate"])
        i_pa = z[f"{s}__i_pa"]
        step = int(round(rate * DT_MS / 1000.0))
        n = len(i_pa) // step
        if n >= T_USE:
            xs.append(i_pa[: T_USE * step].reshape(T_USE, step).mean(1) / 100.0)
    if not xs:
        return None
    return np.stack(xs, axis=1).astype(np.float32)  # [T_USE, N]


def bench_tensor(I, beta, u_r, reps=7):
    T, N = I.shape
    u0 = torch.zeros(N, device=dev)
    GS = torch.randn_like(I)
    GU = torch.zeros_like(I)
    U, S = L.serial_forward(I, beta, 1.0, u_r, u0)
    rate = float(S.mean().item())

    # Tuned serial forward & backward
    bns = (16,) if N <= 16 else (16, 32, 64, 128)
    sf = [(timeit(lambda: L.serial_forward(I, beta, 1.0, u_r, u0), reps), "plain")]
    for BN in bns:
        for D in (4, 8, 16, 32):
            sf.append((timeit(lambda BN=BN, D=D: L.serial_forward_tiled(I, beta, 1.0, u_r, u0, BN=BN, D=D, num_warps=1), reps), f"tiled{BN},{D}"))
    ser_fwd, ser_fwd_cfg = min(sf)

    sb = []
    for BN in bns:
        for D in (2, 4, 8, 16):
            sb.append((timeit(lambda BN=BN, D=D: L.serial_backward_tiled_pf(I, u0, U, S, GS, GU, beta, 1.0, u_r, BN=BN, D=D, num_warps=1), reps), f"pf{BN},{D}"))
    ser_bwd, ser_bwd_cfg = min(sb)

    # Tuned scan forward (no repair) + repair check
    cf = []
    for C in (128, 256, 512, 1024):
        for K in (8, 16, 32):
            st = {}
            _, Ss = L.scan_forward_v3(I, beta, 1.0, u_r, u0, C=C, K=K, BN=16 if N <= 16 else 32, nw=1, BNr=16 if N <= 16 else 32, D=16, nwr=1, repair=0, stats=st)
            flip = float((Ss != S).float().mean().item())
            t = timeit(lambda C=C, K=K: L.scan_forward_v3(I, beta, 1.0, u_r, u0, C=C, K=K, BN=16 if N <= 16 else 32, nw=1, BNr=16 if N <= 16 else 32, D=16, nwr=1, repair=0), reps)
            cf.append((t, dict(C=C, K=K, pieces=st["mean_pieces"], ovf=st["overflow_frac"], flip=flip)))
    scan_fwd, scan_cfg = min(cf, key=lambda x: x[0])

    # Check exact repair rounds at best (C, K)
    st_rep = {}
    Ur, Sr = L.scan_forward_v3(I, beta, 1.0, u_r, u0, C=scan_cfg["C"], K=scan_cfg["K"], BN=16 if N <= 16 else 32, nw=1, BNr=16 if N <= 16 else 32, D=16, nwr=1, repair=64, stats=st_rep)
    bit_exact = bool(torch.equal(Ur, U) and torch.equal(Sr, S))

    cb = []
    for C in (128, 256, 512):
        t = timeit(lambda C=C: L.scan_backward_tiled(I, u0, U, S, GS, GU, beta, 1.0, u_r, C=C, BNr=16 if N <= 16 else 32, D=4, nwr=1), reps)
        cb.append((t, C))
    scan_bwd, scan_bwd_C = min(cb)

    return dict(
        N=N, T=T, beta=float(beta), u_r=float(u_r), rate=rate,
        serial_fwd=ser_fwd, serial_bwd=ser_bwd,
        scan_fwd=scan_fwd, scan_bwd=scan_bwd,
        speedup_fwd=ser_fwd / scan_fwd,
        speedup_fb=(ser_fwd + ser_bwd) / (scan_fwd + scan_bwd),
        mean_pieces=scan_cfg["pieces"], overflow_frac=scan_cfg["ovf"],
        flip_norepair=scan_cfg["flip"],
        repair_rounds=st_rep.get("repair_iters", 0), bit_exact=bit_exact,
        scan_cfg=scan_cfg,
    )


def main():
    fits = {}
    for f in sorted(glob.glob("results/e15all_allen_w*.json")):
        for r in json.load(open(f)):
            if "models" in r and "hard" in r["models"]:
                fits[r["cell"]["specimen"]] = r

    # Pick balanced 32 cells (16 spiny + 16 aspiny/sparsely spiny) that exist in DATA
    spiny, aspiny = [], []
    for sid, r in sorted(fits.items()):
        p = f"{DATA}/{sid}.npz"
        if not os.path.exists(p):
            continue
        d = r["cell"].get("dendrite", "")
        if d == "spiny":
            spiny.append((sid, r))
        else:
            aspiny.append((sid, r))
    chosen = spiny[:16] + aspiny[:16]

    per_cell = []
    stacked_cols = []
    for sid, r in chosen:
        X = load_noise1(f"{DATA}/{sid}.npz")
        if X is None:
            continue
        m = r["models"]["hard"]
        I_np = m["gain"] * X + m["bias"]
        for c in range(I_np.shape[1]):
            stacked_cols.append(I_np[:, c])
        I = torch.from_numpy(I_np).to(dev).contiguous()
        res = bench_tensor(I, m["beta"], m["ur"], reps=5)
        res["specimen"] = sid
        res["dendrite"] = r["cell"].get("dendrite", "")
        per_cell.append(res)
        print(
            f"cell {sid} ({res['dendrite'][:6]:6s}) N={res['N']} beta={res['beta']:.3f} rate={100*res['rate']:4.2f}% | "
            f"pieces={res['mean_pieces']:.2f} ovf={res['overflow_frac']:.4f} flip={res['flip_norepair']:.1e} rep={res['repair_rounds']} | "
            f"fwd={res['speedup_fwd']:.2f}x fb={res['speedup_fb']:.2f}x",
            flush=True,
        )

    # Stacked real currents at N = 16, 128 with cortical leak beta in {0.96, 0.99}
    stacked_rows = []
    if len(stacked_cols) >= 16:
        mat = np.stack(stacked_cols, axis=1)  # [T_USE, total_cols]
        for N in (16, 128):
            idx = np.arange(N) % mat.shape[1]
            I = torch.from_numpy(mat[:, idx].copy()).to(dev).contiguous()
            for beta in (0.96, 0.99):
                res = bench_tensor(I, beta, -0.5, reps=5)
                stacked_rows.append(res)
                print(
                    f"stacked N={N:3d} beta={beta} rate={100*res['rate']:4.2f}% | "
                    f"pieces={res['mean_pieces']:.2f} ovf={res['overflow_frac']:.4f} flip={res['flip_norepair']:.1e} rep={res['repair_rounds']} | "
                    f"fwd={res['speedup_fwd']:.2f}x fb={res['speedup_fb']:.2f}x",
                    flush=True,
                )

    out = dict(T=T_USE, per_cell=per_cell, stacked=stacked_rows)
    with open(os.path.join(ROOT, "results", "e36_real_inputs.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("E36 DONE:", len(per_cell), "cells,", len(stacked_rows), "stacked configs")


if __name__ == "__main__":
    main()
