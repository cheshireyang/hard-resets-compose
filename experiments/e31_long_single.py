"""E31: few recurrences, very long sequences -- the regime in which nothing can be batched.

One long recording (N = 1) or a few trials (N = 4, 16), T up to 2^26 steps (3.7 h at 0.2 ms bins, 56 min at 0.05 ms).
Best vs best, each tuned per cell:
  GPU serial forward : untiled + time-tiled BN=16, D in {2..32}            (BN >= N; wider blocks only idle lanes)
  GPU serial backward: tiled and prefetching tiled, BN=16, D in {2..16}
  CPU serial         : single-threaded C (-O3 -march=native -ffast-math), forward and backward (lifscan/cpu_ref.py)
  two-level scan     : C in {256..8192}, K in {16, 32}
  three-level scan   : C in {128, 256, 512}, G in {16, 32, 64}, (K, K2) = (16, 32)
  affine-scan bwd    : C in {128..4096}
Spike flips of every scan configuration against the serial kernel are recorded.
usage: python3 experiments/e31_long_single.py BETA"""
import sys, os, json, itertools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from lifscan import triton_lif as L
from lifscan import cpu_ref

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
dev = "cuda"
OUT = f"results/e31_long_single_beta{beta}.json"


def timeit(fn, reps):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


def cpu_time(fn, reps):
    fn(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(1000 * (time.perf_counter() - t0))
    ts.sort(); return ts[len(ts) // 2]


rows = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {(r["N"], r["T"]) for r in rows}
for N, T in itertools.product((1, 4, 16), (2 ** 20, 2 ** 22, 2 ** 24, 2 ** 26)):
    if N * T > 2 ** 28 or (N, T) in done: continue
    reps = 5 if T <= 2 ** 22 else 3
    torch.manual_seed(0)
    I = (0.15 + 0.4 * torch.randn(T, N, device=dev)).contiguous(); u0 = torch.zeros(N, device=dev)
    GS = torch.randn_like(I); GU = torch.zeros_like(I)
    U, S = L.serial_forward(I, beta, 1., 0., u0)
    r = dict(N=N, T=T, beta=beta, rate=S.mean().item())
    sf = [(timeit(lambda: L.serial_forward(I, beta, 1., 0., u0), reps), "plain")]
    for D in (2, 4, 8, 16, 32):
        sf.append((timeit(lambda: L.serial_forward_tiled(I, beta, 1., 0., u0, BN=16, D=D, num_warps=1), reps), f"tiled16,{D}"))
    sb = []
    for D in (2, 4, 8, 16):
        sb.append((timeit(lambda: L.serial_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., BN=16, D=D, num_warps=1), reps), f"tiled16,{D}"))
        sb.append((timeit(lambda: L.serial_backward_tiled_pf(I, u0, U, S, GS, GU, beta, 1., 0., BN=16, D=D, num_warps=1), reps), f"pf16,{D}"))
    (r["serial_fwd"], r["serial_fwd_cfg"]), (r["serial_bwd"], r["serial_bwd_cfg"]) = min(sf), min(sb)
    # CPU, single thread
    In, u0n, GSn = I.cpu().numpy(), u0.cpu().numpy(), GS.cpu().numpy()
    Un, Sn = cpu_ref.forward(In, u0n, beta)
    r["cpu_fwd"] = cpu_time(lambda: cpu_ref.forward(In, u0n, beta), reps)
    r["cpu_bwd"] = cpu_time(lambda: cpu_ref.backward(In, Un, Sn, GSn, u0n, beta), reps)
    r["cpu_flip"] = float((torch.from_numpy(Sn).to(dev) != S).float().mean())
    # two-level scan
    cf = []
    for C, K in itertools.product((256, 512, 1024, 2048, 4096, 8192), (16, 32)):
        if T % C or T // C < 2: continue
        st = {}; _, Ss = L.scan_forward_v3(I, beta, 1., 0., u0, C=C, K=K, BN=16, nw=1, BNr=16, D=16, nwr=1, stats=st)
        flip = (Ss != S).float().mean().item()
        for D in (8, 16, 32):
            t = timeit(lambda: L.scan_forward_v3(I, beta, 1., 0., u0, C=C, K=K, BN=16, nw=1, BNr=16, D=D, nwr=1), reps)
            cf.append((t, dict(C=C, K=K, D=D, ovf=st["overflow_frac"], pieces=st["mean_pieces"], flip=flip)))
    r["v3_fwd"], r["v3_cfg"] = min(cf, key=lambda x: x[0])
    # three-level scan
    c4 = []
    for C, G in itertools.product((128, 256, 512), (16, 32, 64)):
        if (T // C) % G or T // C // G < 2: continue
        st = {}; _, Ss = L.scan_forward_v4(I, beta, 1., 0., u0, C=C, G=G, K=16, K2=32, BN=16, BNg=16, BNc=16, BNr=16, D=16, stats=st)
        t = timeit(lambda: L.scan_forward_v4(I, beta, 1., 0., u0, C=C, G=G, K=16, K2=32, BN=16, BNg=16, BNc=16, BNr=16, D=16), reps)
        c4.append((t, dict(C=C, G=G, ovf=st["overflow_frac"], sovf=st["super_overflow_frac"], super_pieces=st["super_mean_pieces"],
                           flip=(Ss != S).float().mean().item())))
    r["v4_fwd"], r["v4_cfg"] = min(c4, key=lambda x: x[0]) if c4 else (None, None)
    # affine-scan backward
    cb = []
    Lref = L.serial_backward_tiled_pf(I, u0, U, S, GS, GU, beta, 1., 0., BN=16, D=4)
    for C in (128, 256, 512, 1024, 2048, 4096):
        if T % C or T // C < 2: continue
        Lc = L.scan_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., C=C, BNr=16, D=2, nwr=1)
        rel = ((Lc - Lref).abs().max() / Lref.abs().max()).item()
        for D in (2, 4, 8):
            t = timeit(lambda: L.scan_backward_tiled(I, u0, U, S, GS, GU, beta, 1., 0., C=C, BNr=16, D=D, nwr=1), reps)
            cb.append((t, dict(C=C, D=D, rel=rel)))
    r["scan_bwd"], r["scan_bwd_cfg"] = min(cb, key=lambda x: x[0])
    fwd = min(x for x in (r["v3_fwd"], r["v4_fwd"]) if x is not None)
    r["scan_fwd"] = fwd
    r["speedup_fwd_gpu"] = r["serial_fwd"] / fwd; r["speedup_fwd_cpu"] = r["cpu_fwd"] / fwd
    r["speedup_bwd_gpu"] = r["serial_bwd"] / r["scan_bwd"]; r["speedup_bwd_cpu"] = r["cpu_bwd"] / r["scan_bwd"]
    r["speedup_fb_gpu"] = (r["serial_fwd"] + r["serial_bwd"]) / (fwd + r["scan_bwd"])
    r["speedup_fb_cpu"] = (r["cpu_fwd"] + r["cpu_bwd"]) / (fwd + r["scan_bwd"])
    r["speedup_fb_best_serial"] = min(r["serial_fwd"] + r["serial_bwd"], r["cpu_fwd"] + r["cpu_bwd"]) / (fwd + r["scan_bwd"])
    rows.append(r)
    print(f"N={N:<3}T={T:<9}| GPU serial f {r['serial_fwd']:9.2f} b {r['serial_bwd']:9.2f} | CPU f {r['cpu_fwd']:9.2f} b {r['cpu_bwd']:9.2f} "
          f"| scan f {r['v3_fwd']:7.3f} (v4 {r['v4_fwd'] if r['v4_fwd'] is None else round(r['v4_fwd'], 3)}) b {r['scan_bwd']:7.3f} "
          f"| f+b x{r['speedup_fb_gpu']:6.1f} vs GPU, x{r['speedup_fb_cpu']:6.1f} vs CPU | flips {r['v3_cfg']['flip']:.1e}", flush=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    del I, GS, GU, U, S; torch.cuda.empty_cache()
