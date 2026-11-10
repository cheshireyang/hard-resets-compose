"""E9: long sequences on the second RTX 5090 node: tuned serial vs two-level (v3) vs three-level (v4)
exact forward, plus the affine-scan backward.  Every implementation tuned per cell.
usage: python3 experiments/e9_threelevel.py BETA"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
dev = "cuda"
TILE = ((32, 4, 1), (32, 8, 1), (64, 4, 2), (128, 4, 4))

def timeit(fn, reps=7):
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
    for BN, D, nw in TILE:
        sf.append(timeit(lambda: L.serial_forward_tiled(I, beta, 1., 0., u0, BN=BN, D=D, num_warps=nw)))
        sb.append(timeit(lambda: L.serial_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., BN=BN, D=D, num_warps=nw)))
    r["serial_fwd"], r["serial_bwd"] = min(sf), min(sb)
    v3 = []
    for C in (64, 128, 256):
        for K in (8, 16, 32):
            st = {}; _, Ss = L.scan_forward_v3(I, beta, 1., 0., u0, C=C, K=K, stats=st)
            t = timeit(lambda: L.scan_forward_v3(I, beta, 1., 0., u0, C=C, K=K))
            v3.append((t, dict(C=C, K=K, ovf=st["overflow_frac"], flip=(Ss != S).float().mean().item())))
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
    for C in (64, 128, 256):
        for BN, D, nw in TILE:
            if C % D: continue
            cb.append(timeit(lambda: L.scan_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., C=C, BNr=BN, D=D, nwr=nw)))
    r["scan_bwd"] = min(cb)
    r["x_v3"] = r["serial_fwd"] / r["v3_fwd"]; r["x_v4"] = r["serial_fwd"] / r["v4_fwd"]
    r["x_fb_v4"] = (r["serial_fwd"] + r["serial_bwd"]) / (min(r["v3_fwd"], r["v4_fwd"]) + r["scan_bwd"])
    rows.append(r)
    print(f"N={N:<5} T={T:<7} serial f {r['serial_fwd']:7.3f} | v3 {r['v3_fwd']:7.3f} (x{r['x_v3']:5.2f}, C={r['v3_cfg']['C']},K={r['v3_cfg']['K']}) "
          f"| v4 {r['v4_fwd']:7.3f} (x{r['x_v4']:5.2f}, C={r['v4_cfg']['C']},G={r['v4_cfg']['G']},K={r['v4_cfg']['K']},sovf={r['v4_cfg']['sovf']:.3f},flip={r['v4_cfg']['flip']:.0e}) "
          f"| bwd serial {r['serial_bwd']:7.3f} scan {r['scan_bwd']:7.3f} | f+b x{r['x_fb_v4']:5.1f}", flush=True)
    json.dump(rows, open(f"results/e9_threelevel_beta{beta}.json", "w"), indent=1)
    del I, U, S, GS, GU; torch.cuda.empty_cache()
