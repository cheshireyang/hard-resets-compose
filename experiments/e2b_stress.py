"""E2b: near-threshold stress test.  With prob p, I_t is chosen so that the (float64) serial
pre-reset potential lands within +-eps of theta; measure float32 spike flips of the scan."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fast import *
rows = []
T, N = 4096, 512
for beta in (0.9, 0.99):
    for eps in (1e-5, 1e-6, 1e-7):
        torch.manual_seed(0)
        I = 0.15 + 0.4 * torch.randn(T, N, dtype=torch.float64)
        hit = torch.rand(T, N) < 0.3
        u = torch.zeros(N, dtype=torch.float64)
        for t in range(T):
            tgt = 1.0 + eps * (2 * torch.rand(N, dtype=torch.float64) - 1)
            I[t] = torch.where(hit[t], tgt - beta * u, I[t])
            v = beta * u + I[t]; u = torch.where(v >= 1.0, torch.zeros_like(v), v)
        I32 = I.float(); u0 = torch.zeros(N)
        _, s32, _ = serial_forward_nograd(I32, beta, 1.0, 0.0, u0)
        _, s64, _ = serial_forward_nograd(I, beta, 1.0, 0.0, u0.double())
        st = {}
        _, sc, _ = scan_forward_pruned(I32, beta, 1.0, 0.0, u0, C=64, K=64, stats=st)
        st2 = {}
        _, sc2, _ = scan_forward_pruned(I32, beta, 1.0, 0.0, u0, C=64, K=64, repair=10**6, stats=st2)
        r = dict(beta=beta, eps=eps, flip_scan32_vs_ser32=(sc != s32).float().mean().item(),
                 neurons_diverged=(sc != s32).any(0).float().mean().item(),
                 flip_ser32_vs_ser64=(s32 != s64.float()).float().mean().item(),
                 repair_iters=st2["repair_iters"], bitexact_after_repair=bool(torch.equal(sc2, s32)))
        rows.append(r); print(r, flush=True)
json.dump(rows, open("results/e2b_stress.json", "w"), indent=1)
