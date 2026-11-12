"""E25d: exact ALIF scan (predicted adaptation box) vs tuned serial ALIF kernels, forward pass, RTX 5090.
Full tuning grids on both sides (supersedes the first E25 run, whose serial grid lacked deep tiles).

Inputs: free membrane potential is an OU process with stationary mean m and std 0.5 (in units of theta0 = 1).
Predictions of the entry adaptation of every chunk:
  prev   -- the previous iterate of an optimisation loop, emulated by the trajectory for inputs scaled by 1.005
            (free in a fitting loop: the replay of the previous iteration stores w at chunk boundaries);
  warm4  -- simulate the 4 preceding chunks from (u_r, 0) (no prior; its cost is included in the timing).
Every implementation is tuned per cell (best median time); spikes are compared with the serial kernel.
usage: python3 experiments/e25_alif_scan.py OUT.json [quick]"""
import sys, os, math, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_alif as A

OUT = sys.argv[1]
QUICK = len(sys.argv) > 2 and sys.argv[2] == "quick"


def tm(fn, reps=7):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


NEURONS = [("lsnn", 0.95, 0.995, 0.9), ("glif2", 0.99, 0.96, 0.2), ("appD-a", 0.9, 0.95, 0.5), ("appD-b", 0.99, 0.99, 0.25)]
MS = (0.5, 2.0) if QUICK else (0.5, 1.0, 2.0)
NS = (16, 128, 1024)
TS = (65536,) if QUICK else (65536, 262144)
GRID = list(itertools.product((128, 256), (0.1, 0.3), ((16, 32), (32, 32)), (4, 8), (4, 16)))
rows = []
for (name, beta, rho, q), mfree, N, T in itertools.product(NEURONS, MS, NS, TS):
    torch.manual_seed(0)
    I = ((1 - beta) * mfree + 0.5 * math.sqrt(1 - beta * beta) * torch.randn(T, N, device="cuda")).contiguous()
    u0 = torch.zeros(N, device="cuda"); w0 = torch.zeros(N, device="cuda")
    ser = {}
    ser["plain"] = tm(lambda: A.alif_serial_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, BN=32, D=0))
    for BN, D, nw in itertools.product((16, 32, 64, 128), (2, 4, 8, 16, 32), (1, 2, 4)):
        ser[f"BN{BN}_D{D}_nw{nw}"] = tm(lambda: A.alif_serial_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, BN=BN, D=D, num_warps=nw))
    sbest = min(ser, key=ser.get)
    _, Wr, Sr = A.alif_serial_forward(I, beta, rho, q, 1.0, 0.0, u0, w0)
    _, Wp, _ = A.alif_serial_forward((1.005 * I).contiguous(), beta, rho, q, 1.0, 0.0, u0, w0)
    r = dict(name=name, beta=beta, rho=rho, q=q, m=mfree, N=N, T=T, rate=Sr.mean().item(), ms_serial=ser[sbest], serial_cfg=sbest)
    for pred in ("prev", "warm4"):
        best = None
        for C, dwq, (K1, K2), BN, Dr in GRID:
            if pred == "prev":
                Wpred = torch.cat([w0[None], Wp[C - 1:-1:C]], 0).contiguous()
                f = lambda: A.alif_scan_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, Wpred, dwq * q, C=C, K1=K1, K2=K2, BN=BN, BNc=1, D=Dr)
            else:
                f = lambda: A.alif_scan_forward(I, beta, rho, q, 1.0, 0.0, u0, w0,
                                                A.alif_warm_predict(I, beta, rho, q, 1.0, 0.0, u0, w0, C, Wn=4),
                                                dwq * q, C=C, K1=K1, K2=K2, BN=BN, BNc=1, D=Dr)
            t = tm(f, reps=5)
            if best is None or t < best[0]:
                best = (t, (C, dwq, K1, K2, BN, Dr))
        C, dwq, K1, K2, BN, Dr = best[1]
        Wpred = (torch.cat([w0[None], Wp[C - 1:-1:C]], 0) if pred == "prev" else A.alif_warm_predict(I, beta, rho, q, 1.0, 0.0, u0, w0, C, Wn=4)).contiguous()
        st = {}; ph = {}
        _, _, S = A.alif_scan_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, Wpred, dwq * q, C=C, K1=K1, K2=K2, BN=BN, BNc=1, D=Dr, stats=st, phases=ph)
        r[f"ms_{pred}"] = best[0]; r[f"cfg_{pred}"] = dict(C=C, dwq=dwq, K1=K1, K2=K2, BN=BN, D=Dr)
        r[f"stats_{pred}"] = st; r[f"phases_{pred}"] = ph
        r[f"flips_{pred}"] = (S != Sr).float().mean().item()
    rows.append(r)
    print(f"{name:6s} m={mfree} N={N:5d} T={T}: rate {100 * r['rate']:.2f}% serial {r['ms_serial']:7.2f} | prev {r['ms_prev']:7.2f} ({r['ms_serial'] / r['ms_prev']:4.2f}x, miss {100 * r['stats_prev']['miss_frac']:4.1f}%) "
          f"| warm4 {r['ms_warm4']:7.2f} ({r['ms_serial'] / r['ms_warm4']:4.2f}x, miss {100 * r['stats_warm4']['miss_frac']:4.1f}%) | flips {r['flips_prev']:.0e}/{r['flips_warm4']:.0e}", flush=True)
    json.dump(rows, open(OUT, "w"), indent=1)
