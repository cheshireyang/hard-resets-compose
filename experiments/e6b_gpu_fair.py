"""E6b: best-vs-best benchmark on RTX 5090.  Every implementation is tuned over its own
config grid per (N, T); we report the best time of each.
usage: python3 experiments/e6b_gpu_fair.py BETA"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L
from spikingjelly.activation_based import functional as sjf, surrogate

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
KS = (8, 16, 32)
tau = 1.0 / (1.0 - beta)
dev = "cuda"
TILE = ((32, 4, 1), (64, 4, 2), (32, 8, 1), (128, 4, 4))

def timeit(fn, reps=7):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]

def best(cands):
    return min(cands, key=lambda c: c[0])

rows = []
for N, T in itertools.product((128, 1024, 8192, 32768, 65536, 262144), (1024, 4096, 16384, 65536)):
    if N * T > 2 ** 28: continue
    torch.manual_seed(0)
    I = (0.15 + 0.4 * torch.randn(T, N, device=dev)).contiguous(); u0 = torch.zeros(N, device=dev)
    GS = torch.randn_like(I); GU = torch.zeros_like(I)
    U, S = L.serial_forward(I, beta, 1., 0., u0)
    r = dict(N=N, T=T, beta=beta)
    # serial forward / backward
    sf = [(timeit(lambda: L.serial_forward(I, beta, 1., 0., u0)), "plain")]
    sb = [(timeit(lambda: L.serial_backward(I, u0, U, S, GS, GU, beta, 1., 0.)), "plain")]
    for BN, D, nw in TILE:
        sf.append((timeit(lambda: L.serial_forward_tiled(I, beta, 1., 0., u0, BN=BN, D=D, num_warps=nw)), f"tiled{BN},{D},{nw}"))
        sb.append((timeit(lambda: L.serial_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., BN=BN, D=D, num_warps=nw)), f"tiled{BN},{D},{nw}"))
    r["serial_fwd"], r["serial_fwd_cfg"] = best(sf)
    r["serial_bwd"], r["serial_bwd_cfg"] = best(sb)
    # spikingjelly
    sg = surrogate.Sigmoid(alpha=4.0)
    Isj = I.clone().requires_grad_(True)
    def sj_fwd():
        with torch.no_grad():
            return sjf.lif_multi_step(I, u0.clone(), tau, False, 1.0, 0.0, sg, True, False)
    def sj_fb():
        s, v, _ = sjf.lif_multi_step(Isj, u0.clone(), tau, False, 1.0, 0.0, sg, True, False); s.backward(GS)
    r["sj_fwd"] = timeit(sj_fwd); r["sj_fb"] = timeit(sj_fb)
    r["sj_flip_vs_serial"] = (sj_fwd()[0] != S).float().mean().item()
    # scan forward (v3: tight envelope bounds) and backward
    cf, cb = [], []
    for C in (32, 64, 128, 256):
        if T % C or T // C < 2: continue
        for K in KS:
            st = {}
            Us, Ss = L.scan_forward_v3(I, beta, 1., 0., u0, C=C, K=K, stats=st)
            flip = (Ss != S).float().mean().item()
            for BN, nw in ((32, 1), (128, 4)):
                for BNr, D, nwr in ((32, 8, 1), (64, 4, 2)):
                    t = timeit(lambda: L.scan_forward_v3(I, beta, 1., 0., u0, C=C, K=K, BN=BN, nw=nw, BNr=BNr, D=D, nwr=nwr))
                    cf.append((t, dict(C=C, K=K, BN=BN, nw=nw, BNr=BNr, D=D, ovf=st["overflow_frac"],
                                       pieces=st["mean_pieces"], flip=flip)))
        for BN, D, nw in TILE:
            if C % D: continue
            t = timeit(lambda: L.scan_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., C=C, BNr=BN, D=D, nwr=nw))
            cb.append((t, dict(C=C, BNr=BN, D=D, nw=nw)))
    r["scan_fwd"], r["scan_fwd_cfg"] = best(cf)
    r["scan_bwd"], r["scan_bwd_cfg"] = best(cb)
    sfb = r["serial_fwd"] + r["serial_bwd"]; cfb = r["scan_fwd"] + r["scan_bwd"]
    r["speedup_fwd"] = r["serial_fwd"] / r["scan_fwd"]; r["speedup_bwd"] = r["serial_bwd"] / r["scan_bwd"]
    r["speedup_fb"] = sfb / cfb; r["speedup_fb_vs_sj"] = r["sj_fb"] / cfb
    rows.append(r)
    print(f"N={N:<7}T={T:<6}| serial f {r['serial_fwd']:7.3f} b {r['serial_bwd']:7.3f} | SJ f+b {r['sj_fb']:7.3f} | "
          f"scan f {r['scan_fwd']:7.3f} (C={r['scan_fwd_cfg']['C']},K={r['scan_fwd_cfg']['K']},ovf={r['scan_fwd_cfg']['ovf']:.2f},flip={r['scan_fwd_cfg']['flip']:.0e}) "
          f"b {r['scan_bwd']:7.3f} | x fwd {r['speedup_fwd']:5.2f} bwd {r['speedup_bwd']:5.2f} f+b {r['speedup_fb']:5.2f} vsSJ {r['speedup_fb_vs_sj']:5.2f}", flush=True)
    json.dump(rows, open(f"results/e6c_fair_beta{beta}.json", "w"), indent=1)
    del I, GS, GU, U, S, Isj; torch.cuda.empty_cache()
