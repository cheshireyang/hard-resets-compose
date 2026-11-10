"""E1c: hard-reset pieces per chunk map when pruning with the tight per-chunk envelope bound
l_{mC} (l_t = min(u_r, beta*l_{t-1}+I_t)) vs. the loose global bound, as a function of chunk
length L.  float64, CPU."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fast import build_chunk_maps
torch.manual_seed(0)
rows = []
T, N = 16384, 64
for beta in (0.9, 0.99, 0.999, 1.0):
    for mu, sd in ((0.15, 0.3), (0.0, 0.3), (-0.1, 0.5)):
        I = mu + sd * torch.randn(T, N, dtype=torch.float64)
        u0 = torch.zeros(N, dtype=torch.float64)
        # envelope
        l = torch.empty(T + 1, N, dtype=torch.float64); l[0] = u0
        for t in range(T):
            l[t + 1] = torch.minimum(torch.zeros(N, dtype=torch.float64), beta * l[t] + I[t])
        hi = 1.0
        lo_loose = (torch.minimum(u0, torch.zeros_like(u0)) + I.min(0).values.clamp(max=0) / (1 - beta)) if beta < 1 \
            else (u0.clamp(max=0) + I.clamp(max=0).sum(0))
        line = []
        for L in (16, 64, 256, 1024, 4096):
            M = T // L
            A, B, xd, cd, n = build_chunk_maps(I.reshape(M, L, N).transpose(0, 1).contiguous(), beta, 1.0, 0.0)
            K = xd.shape[-1]
            valid = torch.arange(K) < n[..., None]
            lo_t = l[torch.arange(M) * L][..., None]          # [M, N, 1]
            tight = (valid & (xd >= lo_t) & (xd <= hi)).sum(-1) + 1
            loose = (valid & (xd >= lo_loose[None, :, None]) & (xd <= hi)).sum(-1) + 1
            allp = n + 1
            rows.append(dict(beta=beta, mu=mu, sd=sd, L=L, tight_mean=tight.double().mean().item(),
                             tight_max=int(tight.max()), tight_p99=tight.double().quantile(0.99).item(),
                             loose_mean=loose.double().mean().item(), all_mean=allp.double().mean().item()))
            line.append(f"L={L}: {rows[-1]['tight_mean']:.1f} (p99 {rows[-1]['tight_p99']:.0f}, max {rows[-1]['tight_max']}) / {rows[-1]['loose_mean']:.1f} / {rows[-1]['all_mean']:.0f}")
        print(f"beta={beta} mu={mu}: tight / loose / all  ", " | ".join(line), flush=True)
json.dump(rows, open("results/e1c_tight.json", "w"), indent=1)
