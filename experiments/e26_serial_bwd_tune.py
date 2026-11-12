"""E26: tuning of the serial backward kernels (the baseline every backward speedup is measured against).
For LIF (hard reset) and soft reset: the plain kernel, the old time-tiled kernel (used in E6c/E9/E11 and the
first arXiv version) and the time-tiled kernel with one-tile prefetch, each over (BN, D, num_warps).
Outputs must be identical across kernels (same arithmetic, same order).
usage: python3 experiments/e26_serial_bwd_tune.py"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L


def tm(fn, reps=5):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


rows = []
GRID = [(BN, D, nw) for BN in (16, 32, 64, 128) for D in (2, 4, 8, 16) for nw in (1, 2, 4)]
for N, T in itertools.product((4, 16, 128, 1024, 8192, 32768), (1024, 8192, 65536)):
    if N * T > 2 ** 28: continue
    torch.manual_seed(0)
    I = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda")
    U, S = L.serial_forward(I, 0.9, 1.0, 0.0, u0); GS = torch.randn(T, N, device="cuda"); GU = torch.zeros_like(GS)
    ref = L.serial_backward(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0)
    r = dict(N=N, T=T, hard={}, soft={})
    r["hard"]["plain"] = tm(lambda: L.serial_backward(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0))
    same = True
    for BN, D, nw in GRID:
        if BN > max(16, N) or T % D: continue
        r["hard"][f"old_{BN}_{D}_{nw}"] = tm(lambda: L.serial_backward_tiled(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0, BN=BN, D=D, num_warps=nw))
        r["hard"][f"pf_{BN}_{D}_{nw}"] = tm(lambda: L.serial_backward_tiled_pf(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0, BN=BN, D=D, num_warps=nw))
        same &= torch.equal(L.serial_backward_tiled_pf(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0, BN=BN, D=D, num_warps=nw), ref)
    r["hard_identical"] = bool(same)
    Us, Ss = L.soft_serial_forward(I, 0.9, 1.0, u0)
    refs = L.soft_serial_backward(I, u0, Us, GS, 0.9, 1.0)
    r["soft"]["plain"] = tm(lambda: L.soft_serial_backward(I, u0, Us, GS, 0.9, 1.0))
    same = True
    for BN, D, nw in GRID:
        if BN > max(16, N) or T % D: continue
        r["soft"][f"pf_{BN}_{D}_{nw}"] = tm(lambda: L.soft_serial_backward_pf(I, u0, Us, GS, 0.9, 1.0, BN=BN, D=D, num_warps=nw))
        same &= torch.equal(L.soft_serial_backward_pf(I, u0, Us, GS, 0.9, 1.0, BN=BN, D=D, num_warps=nw), refs)
    r["soft_identical"] = bool(same)
    def best(d, pre): k = min((k for k in d if k.startswith(pre)), key=d.get); return k, d[k]
    ko, to = best(r["hard"], "old"); kp, tp = best(r["hard"], "pf"); ksp, tsp = best(r["soft"], "pf")
    print(f"N={N:6d} T={T:6d} | hard: plain {r['hard']['plain']:8.2f} old-best {to:8.2f} ({ko}) pf-best {tp:8.2f} ({kp}) old/pf {to / tp:4.2f} "
          f"same {r['hard_identical']} | soft: plain {r['soft']['plain']:8.2f} pf-best {tsp:8.2f} ({ksp}) plain/pf {r['soft']['plain'] / tsp:5.2f} same {r['soft_identical']}", flush=True)
    rows.append(r)
    json.dump(rows, open("results/e26_serial_bwd_tune.json", "w"), indent=1)
