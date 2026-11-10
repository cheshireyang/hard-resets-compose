"""E6: RTX 5090 benchmark.  Fused serial (ours, Triton) / SpikingJelly 2.0 fused / map-scan (Triton).
usage: python3 experiments/e6_gpu_bench.py [beta] [out.json]"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as tl_
from spikingjelly.activation_based import functional as sjf, surrogate

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
out = sys.argv[2] if len(sys.argv) > 2 else f"results/e6_gpu_beta{beta}.json"
tau = 1.0 / (1.0 - beta)
KS = (8, 16, 32) if beta <= 0.9 else (16, 32, 64)
dev = "cuda"

def timeit(fn, reps=10):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]

rows = []
Ns = (128, 1024, 8192, 65536, 262144)
Ts = (1024, 4096, 16384, 65536)
for N, T in itertools.product(Ns, Ts):
    if N * T > 2 ** 28: continue
    torch.manual_seed(0)
    I = (0.15 + 0.4 * torch.randn(T, N, device=dev)).contiguous()
    u0 = torch.zeros(N, device=dev)
    GS = torch.randn_like(I); GU = torch.zeros_like(I)
    r = dict(N=N, T=T, beta=beta)
    # ours serial
    r["fwd_serial"] = timeit(lambda: tl_.serial_forward(I, beta, 1.0, 0.0, u0))
    U, S = tl_.serial_forward(I, beta, 1.0, 0.0, u0)
    r["bwd_serial"] = timeit(lambda: tl_.serial_backward(I, u0, U, S, GS, GU, beta, 1.0, 0.0))
    # spikingjelly fused
    Isj = I.clone().requires_grad_(True)
    sg = surrogate.Sigmoid(alpha=4.0)
    def sj_fwd():
        with torch.no_grad():
            return sjf.lif_multi_step(I, u0.clone(), tau, False, 1.0, 0.0, sg, True, False)
    def sj_fb():
        s, v, _ = sjf.lif_multi_step(Isj, u0.clone(), tau, False, 1.0, 0.0, sg, True, False)
        s.backward(GS)
    try:
        r["fwd_sj"] = timeit(sj_fwd); r["fb_sj"] = timeit(sj_fb)
        ssj = sj_fwd()[0]
        r["sj_spikes_equal_serial"] = bool(torch.equal(ssj, S))
        r["sj_flip_vs_serial"] = (ssj != S).float().mean().item()
    except Exception as e:
        r["sj_error"] = repr(e)[:200]
    # scan: sweep C, K
    best = None
    for C in (16, 32, 64, 128, 256):
        if T % C or T // C < 2: continue
        for K in KS:
            st = {}
            Us, Ss = tl_.scan_forward(I, beta, 1.0, 0.0, u0, C=C, K=K, stats=st)
            tf = timeit(lambda: tl_.scan_forward(I, beta, 1.0, 0.0, u0, C=C, K=K))
            tb = timeit(lambda: tl_.scan_backward(I, u0, U, S, GS, GU, beta, 1.0, 0.0, C=C))
            cand = dict(C=C, K=K, fwd=tf, bwd=tb, ovf=st["overflow_frac"], flip=(Ss != S).float().mean().item())
            if best is None or tf + tb < best["fwd"] + best["bwd"]:
                best = cand
            r.setdefault("sweep", []).append(cand)
    r["scan_best"] = best
    rows.append(r)
    sp = lambda a, b: a / b
    print(f"N={N:<7} T={T:<6} | serial fwd {r['fwd_serial']:8.3f} bwd {r['bwd_serial']:8.3f} ms | "
          f"SJ fwd {r.get('fwd_sj', float('nan')):8.3f} f+b {r.get('fb_sj', float('nan')):8.3f} | "
          f"scan(C={best['C']},K={best['K']}) fwd {best['fwd']:8.3f} bwd {best['bwd']:8.3f} ovf {best['ovf']:.3f} flip {best['flip']:.1e} | "
          f"speedup f+b vs serial x{(r['fwd_serial']+r['bwd_serial'])/(best['fwd']+best['bwd']):.2f} "
          f"vs SJ x{r.get('fb_sj', float('nan'))/(best['fwd']+best['bwd']):.2f}", flush=True)
    json.dump(rows, open(out, "w"), indent=1)
    del I, GS, GU, U, S, Isj; torch.cuda.empty_cache()
