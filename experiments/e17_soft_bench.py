"""E17: soft-reset LIF forward on RTX 5090: tuned serial (plain / time-tiled, full grid) vs two-level soft scan."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grids import best_serial
from lifscan import triton_soft as S

def timeit(fn, reps=9):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]

rows = []
for beta in (0.95, 0.97):
    for N, T in itertools.product((4, 128, 1024, 8192), (16384, 65536, 262144)):
        if N * T > 2 ** 28: continue
        torch.manual_seed(0)
        I = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda")
        Ur, Sr = S.soft_serial_forward(I, beta, 1.0, u0)
        ser_t, ser_cfg = best_serial(timeit, lambda: S.soft_serial_forward(I, beta, 1.0, u0),
                                     lambda BN, D, nw: S.soft_serial_forward_tiled(I, beta, 1.0, u0, BN=BN, D=D, num_warps=nw), T)
        ser = [ser_t]
        best = None
        for C in (64, 128, 256):
            for K in (16, 32):
                st = {}
                _, Sp = S.soft_scan_forward(I, beta, 1.0, u0, C=C, K=K, stats=st)
                t = timeit(lambda: S.soft_scan_forward(I, beta, 1.0, u0, C=C, K=K))
                cand = dict(t=t, C=C, K=K, ovf=st["overflow_frac"], pieces=st["mean_pieces"], flip=(Sp != Sr).float().mean().item())
                if best is None or t < best["t"]: best = cand
        r = dict(beta=beta, N=N, T=T, serial=min(ser), serial_cfg=ser_cfg, scan=best["t"], cfg=best, speedup=min(ser) / best["t"])
        rows.append(r)
        print(f"beta={beta} N={N:<5} T={T:<7} serial {r['serial']:8.3f} scan {r['scan']:8.3f} x{r['speedup']:5.2f} "
              f"(C={best['C']},K={best['K']},ovf={best['ovf']:.3f},pieces={best['pieces']:.1f},flip={best['flip']:.0e})", flush=True)
        json.dump(rows, open("results/e17_soft_bench.json", "w"), indent=1)
