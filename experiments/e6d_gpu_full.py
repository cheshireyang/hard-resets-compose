"""E6d: best-vs-best benchmark with full tuning grids on BOTH sides (supersedes E6c, whose serial baselines were
tuned over four tiles only and whose serial backward did not prefetch; see E26/E26b).
  serial forward : plain + time-tiled, BN x D x num_warps = {16..128} x {2..32} x {1,2,4}
  serial backward: plain + old tiled + prefetching tiled, BN x D x num_warps = {16..128} x {2..16} x {1,2,4}
  scan forward   : C x K x build(BN, nw) x replay(BNr, D, nwr)   (replay = the serial tiled kernel per chunk)
  scan backward  : C x (BNr, D, nwr)
usage: python3 experiments/e6d_gpu_full.py BETA"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L
from spikingjelly.activation_based import functional as sjf, surrogate

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
tau = 1.0 / (1.0 - beta)
dev = "cuda"
SF = [(BN, D, nw) for BN in (16, 32, 64, 128) for D in (2, 4, 8, 16, 32) for nw in (1, 2, 4)]
SB = [(BN, D, nw) for BN in (16, 32, 64, 128) for D in (2, 4, 8, 16) for nw in (1, 2, 4)]
BUILD = ((32, 1), (128, 4))
REPLAY = ((16, 4, 1), (32, 8, 1), (64, 4, 2), (64, 16, 1), (64, 32, 1), (128, 16, 1))
CBW = [(BN, D, nw) for BN in (16, 32, 64, 128) for D in (2, 4, 8, 16) for nw in (1, 2)]


def timeit(fn, reps=5):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


def best(c):
    return min(c, key=lambda x: x[0])


rows = []
for N, T in itertools.product((128, 1024, 8192, 32768, 65536, 262144), (1024, 4096, 16384, 65536)):
    if N * T > 2 ** 28: continue
    torch.manual_seed(0)
    I = (0.15 + 0.4 * torch.randn(T, N, device=dev)).contiguous(); u0 = torch.zeros(N, device=dev)
    GS = torch.randn_like(I); GU = torch.zeros_like(I)
    U, S = L.serial_forward(I, beta, 1., 0., u0)
    r = dict(N=N, T=T, beta=beta)
    sf = [(timeit(lambda: L.serial_forward(I, beta, 1., 0., u0)), "plain")]
    for BN, D, nw in SF:
        if T % D: continue
        sf.append((timeit(lambda: L.serial_forward_tiled(I, beta, 1., 0., u0, BN=BN, D=D, num_warps=nw)), f"tiled{BN},{D},{nw}"))
    sb = [(timeit(lambda: L.serial_backward(I, u0, U, S, GS, GU, beta, 1., 0.)), "plain")]
    sb_old = []
    for BN, D, nw in SB:
        if T % D: continue
        t_old = timeit(lambda: L.serial_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., BN=BN, D=D, num_warps=nw))
        sb.append((t_old, f"tiled{BN},{D},{nw}")); sb_old.append(t_old)
        sb.append((timeit(lambda: L.serial_backward_tiled_pf(I, u0, U, S, GS, GU, beta, 1., 0., BN=BN, D=D, num_warps=nw)), f"pf{BN},{D},{nw}"))
    r["serial_fwd"], r["serial_fwd_cfg"] = best(sf)
    r["serial_bwd"], r["serial_bwd_cfg"] = best(sb)
    r["serial_bwd_old_best"] = min(sb_old)
    sg = surrogate.Sigmoid(alpha=4.0)
    Isj = I.clone().requires_grad_(True)
    def sj_fwd():
        with torch.no_grad():
            return sjf.lif_multi_step(I, u0.clone(), tau, False, 1.0, 0.0, sg, True, False)
    def sj_fb():
        s, v, _ = sjf.lif_multi_step(Isj, u0.clone(), tau, False, 1.0, 0.0, sg, True, False); s.backward(GS)
    r["sj_fwd"] = timeit(sj_fwd); r["sj_fb"] = timeit(sj_fb)
    cf, cb = [], []
    for C in (32, 64, 128, 256):
        if T % C or T // C < 2: continue
        for K in (8, 16, 32):
            st = {}
            Us, Ss = L.scan_forward_v3(I, beta, 1., 0., u0, C=C, K=K, stats=st)
            flip = (Ss != S).float().mean().item()
            for (BN, nw), (BNr, D, nwr) in itertools.product(BUILD, REPLAY):
                if C % D: continue
                t = timeit(lambda: L.scan_forward_v3(I, beta, 1., 0., u0, C=C, K=K, BN=BN, nw=nw, BNr=BNr, D=D, nwr=nwr))
                cf.append((t, dict(C=C, K=K, BN=BN, nw=nw, BNr=BNr, D=D, nwr=nwr, ovf=st["overflow_frac"], pieces=st["mean_pieces"], flip=flip)))
        if C > 128: continue
        for BN, D, nw in CBW:
            if C % D: continue
            t = timeit(lambda: L.scan_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., C=C, BNr=BN, D=D, nwr=nw))
            cb.append((t, dict(C=C, BNr=BN, D=D, nw=nw)))
    r["scan_fwd"], r["scan_fwd_cfg"] = best(cf)
    r["scan_bwd"], r["scan_bwd_cfg"] = best(cb)
    sfb = r["serial_fwd"] + r["serial_bwd"]; cfb = r["scan_fwd"] + r["scan_bwd"]
    r["speedup_fwd"] = r["serial_fwd"] / r["scan_fwd"]; r["speedup_bwd"] = r["serial_bwd"] / r["scan_bwd"]
    r["speedup_fb"] = sfb / cfb; r["speedup_fb_vs_sj"] = r["sj_fb"] / cfb
    rows.append(r)
    print(f"N={N:<7}T={T:<6}| serial f {r['serial_fwd']:7.3f} ({r['serial_fwd_cfg']}) b {r['serial_bwd']:7.3f} ({r['serial_bwd_cfg']}; old {r['serial_bwd_old_best']:7.3f}) "
          f"| scan f {r['scan_fwd']:7.3f} b {r['scan_bwd']:7.3f} | x fwd {r['speedup_fwd']:5.2f} bwd {r['speedup_bwd']:5.2f} f+b {r['speedup_fb']:5.2f} vsSJ {r['speedup_fb_vs_sj']:5.2f}", flush=True)
    json.dump(rows, open(f"results/e6d_full_beta{beta}.json", "w"), indent=1)
    del I, GS, GU, U, S, Isj; torch.cuda.empty_cache()
