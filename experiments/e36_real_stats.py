"""E36 (deterministic): Hard-reset chunk-map sizes, overflow, spike flips, and repair rounds on
real temporally-correlated biological currents (Allen Cell Types current-clamp recordings).

Purely deterministic (no wall-clock timing, unaffected by concurrent GPU workloads).
Evaluates every downloaded cell in data/allen_all/*.npz with its fitted hard-reset LIF parameters
(from results/e15all_allen_w*.json) on all Noise-1 sweeps (truncated to the largest multiple of
C=256 <= T, typically T=114,688 or 124,672), for K in {8, 16}.

Output: results/e36_real_stats.json
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
C = 256
dev = "cuda"


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
        xs.append(i_pa[: n * step].reshape(n, step).mean(1) / 100.0)
    if not xs:
        return None
    T = (min(len(x) for x in xs) // C) * C
    if T < C:
        return None
    return np.stack([x[:T] for x in xs], axis=1).astype(np.float32)


def main():
    fits = {}
    for f in sorted(glob.glob("results/e15all_allen_w*.json")):
        for r in json.load(open(f)):
            if "models" in r and "hard" in r["models"]:
                fits[r["cell"]["specimen"]] = r

    rows = []
    for path in sorted(glob.glob(f"{DATA}/*.npz")):
        sid = int(os.path.basename(path)[:-4])
        if sid not in fits:
            continue
        X = load_noise1(path)
        if X is None:
            continue
        r = fits[sid]
        m = r["models"]["hard"]
        I_np = m["gain"] * X + m["bias"]
        I = torch.from_numpy(I_np).to(dev).contiguous()
        T, N = I.shape
        u0 = torch.zeros(N, device=dev)
        U, S = L.serial_forward(I, m["beta"], 1.0, m["ur"], u0)
        rate = float(S.mean().item())

        row = dict(
            specimen=sid, dendrite=r["cell"].get("dendrite", ""),
            N=N, T=T, beta=float(m["beta"]), ur=float(m["ur"]), rate=rate,
        )
        for K in (8, 16):
            st0 = {}
            _, Ss = L.scan_forward_v3(I, m["beta"], 1.0, m["ur"], u0, C=C, K=K, BN=16, nw=1, BNr=16, D=16, nwr=1, repair=0, stats=st0)
            flip0 = float((Ss != S).float().mean().item())
            st1 = {}
            Ur, Sr = L.scan_forward_v3(I, m["beta"], 1.0, m["ur"], u0, C=C, K=K, BN=16, nw=1, BNr=16, D=16, nwr=1, repair=64, stats=st1)
            row[f"K{K}"] = dict(
                mean_pieces=st0["mean_pieces"], max_pieces=st0["max_pieces"],
                overflow_frac=st0["overflow_frac"], flip_norepair=flip0,
                repair_iters=st1["repair_iters"],
                bit_exact=bool(torch.equal(Ur, U) and torch.equal(Sr, S)),
            )
        rows.append(row)
        print(
            f"cell {sid} ({row['dendrite'][:6]:6s}) N={N} T={T} beta={row['beta']:.3f} rate={100*rate:4.2f}% | "
            f"K16: pcs={row['K16']['mean_pieces']:.2f} (max {row['K16']['max_pieces']:2d}) ovf={row['K16']['overflow_frac']:.4f} "
            f"flip={row['K16']['flip_norepair']:.1e} rep={row['K16']['repair_iters']} exact={row['K16']['bit_exact']}",
            flush=True,
        )

    summ = dict(
        cells=len(rows), C=C,
        K16_mean_pieces_avg=float(np.mean([r["K16"]["mean_pieces"] for r in rows])),
        K16_mean_pieces_max=float(np.max([r["K16"]["mean_pieces"] for r in rows])),
        K16_max_overflow=float(np.max([r["K16"]["overflow_frac"] for r in rows])),
        K16_mean_overflow=float(np.mean([r["K16"]["overflow_frac"] for r in rows])),
        K16_max_flip_norepair=float(np.max([r["K16"]["flip_norepair"] for r in rows])),
        K16_zero_flip_cells=sum(r["K16"]["flip_norepair"] == 0.0 for r in rows),
        K16_bit_exact_cells=sum(r["K16"]["bit_exact"] for r in rows),
        K16_median_repair=float(np.median([r["K16"]["repair_iters"] for r in rows])),
        K8_max_overflow=float(np.max([r["K8"]["overflow_frac"] for r in rows])),
        K8_mean_overflow=float(np.mean([r["K8"]["overflow_frac"] for r in rows])),
    )
    print("SUMMARY:", summ, flush=True)
    with open(os.path.join(ROOT, "results", "e36_real_stats.json"), "w") as f:
        json.dump(dict(summary=summ, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
