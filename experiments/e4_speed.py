"""E4: wall-clock on Apple M4 (CPU / MPS), PyTorch eager, no custom kernels.
Forward:  serial loop | two-level scan (capacity C) | pruned scan (K=16)
Backward: serial autograd | hand-written serial adjoint (fair) | affine scan"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fast import *

def sync(dev):
    if dev == "mps": torch.mps.synchronize()

def bench(fn, dev, reps=3):
    fn(); sync(dev); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(dev); ts.append(time.perf_counter() - t0)
    return min(ts)

beta = 0.9
rows = []
DEVS = sys.argv[1:] or ["cpu", "mps"]
for dev in DEVS:
    for N in (64, 1024, 8192):
        for T in (256, 1024, 4096, 16384):
            if N * T > 8192 * 4096: continue
            C = max(16, int(2 ** round(math.log2(math.sqrt(T)))))
            torch.manual_seed(0)
            I = (0.15 + 0.4 * torch.randn(T, N)).to(dev); u0 = torch.zeros(N, device=dev)
            us, ss, vs = serial_forward_nograd(I, beta, 1.0, 0.0, u0)
            gs = torch.randn_like(I); gu = torch.zeros_like(I)
            st = {}
            scan_forward_pruned(I, beta, 1.0, 0.0, u0, C=C, K=16, stats=st)
            r = dict(dev=dev, N=N, T=T, C=C, overflow=st["overflow_frac"])
            r["fwd_serial"] = bench(lambda: serial_forward_nograd(I, beta, 1.0, 0.0, u0), dev)
            r["fwd_scan"] = bench(lambda: scan_forward(I, beta, 1.0, 0.0, u0, C=C), dev)
            r["fwd_pruned"] = bench(lambda: scan_forward_pruned(I, beta, 1.0, 0.0, u0, C=C, K=16), dev)
            r["bwd_serial_adjoint"] = bench(lambda: serial_backward(vs, ss, gs, gu, beta, 1.0, 0.0), dev)
            def bwd_scan():
                sg = surrogate_grad(vs - 1.0, 4.0); J = 1 - ss
                a = torch.zeros_like(vs); a[:-1] = beta * J[:-1]
                return reverse_affine_scan(a, gs * sg + J * gu)
            r["bwd_scan"] = bench(bwd_scan, dev)
            if N * T <= 1024 * 4096:
                Ig = I.clone().requires_grad_(True)
                def fb():
                    u, s = serial_lif(Ig, beta, 1.0, 0.0, u0); (gs * s).sum().backward()
                r["fwdbwd_autograd"] = bench(fb, dev, reps=1)
            rows.append(r)
            print(f"{dev} N={N:<5} T={T:<6} C={C:<4} ovf={r['overflow']:.3f} | fwd serial {r['fwd_serial']*1e3:7.1f} "
                  f"scan {r['fwd_scan']*1e3:7.1f} pruned {r['fwd_pruned']*1e3:7.1f} ms | bwd adjoint {r['bwd_serial_adjoint']*1e3:7.1f} "
                  f"scan {r['bwd_scan']*1e3:7.1f} ms | autograd fwd+bwd {r.get('fwdbwd_autograd', float('nan'))*1e3:8.1f}", flush=True)
            json.dump(rows, open(f"results/e4_speed_{'_'.join(DEVS)}.json", "w"), indent=1)
