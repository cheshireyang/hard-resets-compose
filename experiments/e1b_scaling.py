"""E1b: growth of hard-reset piece count with segment length (float64, chunk-map builder)."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fast import build_chunk_maps
torch.manual_seed(0)
rows = []
for beta in (0.9, 0.99, 0.999, 1.0):
    for mu, sd in ((0.0, 0.3), (0.05, 0.3), (0.15, 0.3), (-0.1, 0.5)):
        line = []
        for L in (16, 64, 256, 1024, 4096):
            N = 256 if L <= 1024 else 64
            I = (mu + sd * torch.randn(L, 1, N, dtype=torch.float64))
            A, B, xd, cd, n = build_chunk_maps(I, beta, 1.0, 0.0)
            # count only breakpoints in the reachable domain [lo, theta)
            lo = min(0.0, I.min().item() / max(1e-12, 1 - beta)) if beta < 1 else -math.inf
            K = xd.shape[-1]
            valid = (torch.arange(K) < n[..., None]) & (xd >= lo) & (xd < 1.0)
            p_all = (n + 1).double(); p_dom = (valid.sum(-1) + 1).double()
            rows.append(dict(beta=beta, mu=mu, sd=sd, L=L, mean_all=p_all.mean().item(), max_all=p_all.max().item(),
                             mean_dom=p_dom.mean().item(), max_dom=p_dom.max().item()))
            line.append(f"L={L}:{p_all.mean().item():.1f}/{p_dom.mean().item():.1f}")
        print(f"beta={beta:<6} mu={mu:<5} sd={sd}  (mean pieces all / in-domain) ", "  ".join(line), flush=True)
json.dump(rows, open("results/e1b_scaling.json", "w"), indent=1)
