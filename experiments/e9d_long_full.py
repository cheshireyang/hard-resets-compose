"""E9d: long sequences, full tuning grids on both sides (supersedes E9; see E6d for the grids):
tuned serial vs two-level (v3) vs three-level (v4) exact forward, plus the affine-scan backward.
usage: python3 experiments/e9d_long_full.py BETA"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
dev = "cuda"
SF = [(BN, D, nw) for BN in (16, 32, 64, 128) for D in (2, 4, 8, 16, 32) for nw in (1, 2, 4)]
SB = [(BN, D, nw) for BN in (16, 32, 64, 128) for D in (2, 4, 8, 16) for nw in (1, 2, 4)]
REPLAY = ((16, 4, 1), (32, 8, 1), (64, 4, 2), (64, 16, 1), (64, 32, 1), (128, 16, 1))
CBW = [(BN, D, nw) for BN in (16, 32, 64, 128) for D in (2, 4, 8, 16) for nw in (1, 2)]

def timeit(fn, reps=5):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]

rows = []
for N, T in itertools.product((128, 1024, 8192), (16384, 65536, 262144)):
    if N * T > 2 ** 28: continue
    torch.manual_seed(0)
    I = (0.15 + 0.4 * torch.randn(T, N, device=dev)).contiguous(); u0 = torch.zeros(N, device=dev)
    U, S = L.serial_forward(I, beta, 1., 0., u0)
    GS = torch.randn_like(I); GU = torch.zeros_like(I)
    r = dict(N=N, T=T, beta=beta)
    sf = [timeit(lambda: L.serial_forward(I, beta, 1., 0., u0))]
    sb = [timeit(lambda: L.serial_backward(I, u0, U, S, GS, GU, beta, 1., 0.))]
    for BN, D, nw in SF:
        sf.append(timeit(lambda: L.serial_forward_tiled(I, beta, 1., 0., u0, BN=BN, D=D, num_warps=nw)))
    for BN, D, nw in SB:
        sb.append(timeit(lambda: L.serial_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., BN=BN, D=D, num_warps=nw)))
        sb.append(timeit(lambda: L.serial_backward_tiled_pf(I, u0, U, S, GS, GU, beta, 1., 0., BN=BN, D=D, num_warps=nw)))
    r["serial_fwd"], r["serial_bwd"] = min(sf), min(sb)
    v3 = []
    for C in (64, 128, 256):
        for K in (8, 16, 32):
            st = {}; _, Ss = L.scan_forward_v3(I, beta, 1., 0., u0, C=C, K=K, stats=st)
            for (BN, nw), (BNr, D, nwr) in itertools.product(((32, 1), (128, 4)), REPLAY):
                if C % D: continue
                t = timeit(lambda: L.scan_forward_v3(I, beta, 1., 0., u0, C=C, K=K, BN=BN, nw=nw, BNr=BNr, D=D, nwr=nwr))
                v3.append((t, dict(C=C, K=K, BN=BN, nw=nw, BNr=BNr, D=D, nwr=nwr, ovf=st["overflow_frac"], flip=(Ss != S).float().mean().item())))
    r["v3_fwd"], r["v3_cfg"] = min(v3, key=lambda x: x[0])
    v4 = []
    for C in (32, 64, 128):
        for G in (8, 16, 32):
            if (T // C) % G or T // C // G < 2: continue
            for K, K2 in ((8, 16), (16, 32)):
                st = {}; _, Ss = L.scan_forward_v4(I, beta, 1., 0., u0, C=C, G=G, K=K, K2=K2, stats=st)
                t = timeit(lambda: L.scan_forward_v4(I, beta, 1., 0., u0, C=C, G=G, K=K, K2=K2))
                v4.append((t, dict(C=C, G=G, K=K, K2=K2, ovf=st["overflow_frac"], sovf=st["super_overflow_frac"],
                                   super_pieces=st["super_mean_pieces"], flip=(Ss != S).float().mean().item())))
    r["v4_fwd"], r["v4_cfg"] = min(v4, key=lambda x: x[0])
    cb = []
    for C in (32, 64, 128):
        for BN, D, nw in CBW:
            if C % D: continue
            cb.append(timeit(lambda: L.scan_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., C=C, BNr=BN, D=D, nwr=nw)))
    r["scan_bwd"] = min(cb)
    r["x_v3"] = r["serial_fwd"] / r["v3_fwd"]; r["x_v4"] = r["serial_fwd"] / r["v4_fwd"]
    r["x_fb_v4"] = (r["serial_fwd"] + r["serial_bwd"]) / (min(r["v3_fwd"], r["v4_fwd"]) + r["scan_bwd"])
    rows.append(r)
    print(f"N={N:<5} T={T:<7} serial f {r['serial_fwd']:7.3f} | v3 {r['v3_fwd']:7.3f} (x{r['x_v3']:5.2f}, C={r['v3_cfg']['C']},K={r['v3_cfg']['K']}) "
          f"| v4 {r['v4_fwd']:7.3f} (x{r['x_v4']:5.2f}, C={r['v4_cfg']['C']},G={r['v4_cfg']['G']},K={r['v4_cfg']['K']},sovf={r['v4_cfg']['sovf']:.3f},flip={r['v4_cfg']['flip']:.0e}) "
          f"| bwd serial {r['serial_bwd']:7.3f} scan {r['scan_bwd']:7.3f} | f+b x{r['x_fb_v4']:5.1f}", flush=True)
    json.dump(rows, open(f"results/e9d_long_full_beta{beta}.json", "w"), indent=1)
    del I, U, S, GS, GU; torch.cuda.empty_cache()
